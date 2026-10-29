// benchmark.cpp — эксперимент и отчёт.
#include "benchmark.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <ios>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace radix {

namespace {

// Время выполнения f() в секундах (std::chrono::steady_clock — монотонные часы).
template <class F>
double time_seconds(F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// Многопоточное копирование src -> dst. Нужно только режиму fill с несколькими запусками, чтобы вернуть
// вход после сортировки на месте (в измеряемое время не входит). Многопоточный memcpy заметно ускоряет
// восстановление гигабайтных массивов.
void parallel_copy(const std::vector<Key>& src, std::vector<Key>& dst, int threads) {
    const std::size_t n = src.size();
    if (dst.size() != n) dst.resize(n);
    if (n == 0) return;
    constexpr std::size_t kParallelThreshold = std::size_t{1} << 20;
    if (threads <= 1 || n < kParallelThreshold) {
        std::memcpy(dst.data(), src.data(), n * sizeof(Key));
        return;
    }
    const Key* const s = src.data();
    Key* const d = dst.data();
#pragma omp parallel num_threads(threads)
    {
        const std::size_t tid = static_cast<std::size_t>(omp_get_thread_num());
        const std::size_t nt = static_cast<std::size_t>(omp_get_num_threads());
        const std::size_t lo = block_begin(tid, nt, n);
        const std::size_t hi = block_begin(tid + 1, nt, n);
        std::memcpy(d + lo, s + lo, (hi - lo) * sizeof(Key));
    }
}

void finalize(SeriesStats& s) {
    if (s.samples.empty()) return;
    s.best = *std::min_element(s.samples.begin(), s.samples.end());
    double sum = 0.0;
    for (const double x : s.samples) sum += x;
    s.mean = sum / static_cast<double>(s.samples.size());
    if (s.samples.size() > 1) {
        double acc = 0.0;
        for (const double x : s.samples) acc += (x - s.mean) * (x - s.mean);
        s.stddev = std::sqrt(acc / static_cast<double>(s.samples.size() - 1));
    }
}

}  // namespace

RunSettings default_run_settings(std::size_t n) noexcept {
    RunSettings s;
    if (n >= kLargeInputThreshold) {
        s.repeats = 1;
        s.warmup = false;
    }
    return s;
}

int count_sorts(const RunConfig& c) noexcept {
    const int per_round = (c.run_serial ? 1 : 0) + (c.run_parallel ? 1 : 0);
    return per_round * (c.repeats + (c.warmup ? 1 : 0));
}

std::uint64_t estimate_experiment_memory(std::size_t n, const RunConfig& c) {
    const std::uint64_t array_bytes = static_cast<std::uint64_t>(n) * sizeof(Key);
    const bool scatter = (c.algorithm == Algorithm::Scatter);
    // scatter: вход + выходной буфер (восстановление — обменом указателей). fill: 1 массив при единственной
    // сортировке, иначе 2 (рабочий + эталонная копия для восстановления).
    const std::uint64_t arrays = scatter ? 2 : (count_sorts(c) > 1 ? 2 : 1);
    std::uint64_t total = arrays * array_bytes;
    if (c.run_serial) total += (kKeyRange + 1) * sizeof(Count);
    if (c.run_parallel) {
        const MemoryEstimate m = estimate_memory(n, c.threads, c.algorithm, true);
        total += m.histogram_bytes + m.aux_bytes;
    }
    if (c.verify) {
        total += 2 * kKeyRange * sizeof(std::uint64_t);        // отпечатки входа и результата
        if (n <= kReferenceSortLimit) total += array_bytes;    // эталон std::sort (только для небольших массивов)
    }
    return total;
}

RunResult run_experiment(std::vector<Key>&& input, const RunConfig& cfg, std::vector<Key>* sorted_out) {
    if (!cfg.run_serial && !cfg.run_parallel) throw std::invalid_argument("benchmark: nothing to run");
    if (cfg.repeats < 1) throw std::invalid_argument("benchmark: repeats must be >= 1");
    if (cfg.threads < 1 || cfg.threads > kMaxThreads) {
        throw std::invalid_argument("benchmark: threads must be in [1, " + std::to_string(kMaxThreads) + "]");
    }
    auto say = [&cfg](const std::string& msg) {
        if (cfg.progress) cfg.progress(msg);
    };

    std::vector<Key> a(std::move(input));  // перед каждым запуском здесь лежит ИСХОДНЫЙ массив
    const std::size_t n = a.size();
    const bool scatter = (cfg.algorithm == Algorithm::Scatter);
    const bool need_copy = !scatter && count_sorts(cfg) > 1;

    RunResult r;
    r.config = cfg;
    r.n = n;
    r.restore_bytes = need_copy ? static_cast<std::uint64_t>(n) * sizeof(Key) : 0;

    // ---- Данные для проверки: снимаются с ИСХОДНОГО массива до любых сортировок ----
    Fingerprint fingerprint;
    std::vector<Key> reference;
    bool have_reference = false;
    if (cfg.verify) {
        say("verify: taking checksums and the exact histogram of the input");
        fingerprint = make_fingerprint(a);
        if (n <= kReferenceSortLimit) {
            reference = make_reference_sorted(a);
            have_reference = true;
        }
    }

    // ---- Всё выделяется ЗАРАНЕЕ, один раз (в замеры аллокации не попадают) ----
    std::vector<Key> b;
    ParallelRadixWorkspace workspace;
    if (cfg.run_parallel && n > 0) {
        workspace.reserve(static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(cfg.threads), n)));
    }
    if (scatter) {
        if (n > 0) say("allocating the output buffer (" + format_bytes(static_cast<std::uint64_t>(n) * sizeof(Key)) + ")");
        b.resize(n);
    } else if (need_copy) {
        say("making a pristine copy of the input to restore it between runs (" +
            format_bytes(static_cast<std::uint64_t>(n) * sizeof(Key)) + ")");
        b = a;
    }

    // Приведение массива `a` к исходному состоянию перед очередным запуском (в замер не входит):
    //   scatter — сортировка не изменяет вход: после неё исходные данные целиком лежат в `b`,
    //             поэтому достаточно поменять `a` и `b` местами (O(1), без копирования);
    //   fill    — сортировка на месте портит вход: копируем эталон из `b` обратно в `a`.
    bool dirty = false;  // `a` содержит результат предыдущего запуска
    auto restore = [&] {
        if (!dirty) return;
        if (scatter) a.swap(b);
        else parallel_copy(b, a, cfg.threads);
        dirty = false;
    };
    auto sort_serial = [&](PhaseTimes* pt) {
        if (scatter) serial_radix_sort(a, b, pt);
        else serial_histogram_fill_sort(a, pt);
    };
    auto sort_parallel = [&](PhaseTimes* pt) {
        if (scatter) parallel_radix_sort(a, b, cfg.threads, workspace, pt);
        else parallel_histogram_fill_sort(a, cfg.threads, workspace, pt);
    };
    auto inspect = [&] {
        if (cfg.inspect_input) cfg.inspect_input(a);
    };
    auto check = [&](const char* who) {
        say(std::string("verify: checking the ") + who + " result");
        Verification v;
        const double sec = time_seconds([&] { v = verify_result(a, fingerprint, have_reference ? &reference : nullptr); });
        say(std::string("verify: ") + who + (v.passed() ? " result is correct" : " result is WRONG") + " (checked in " +
            format_seconds(sec) + ")");
        return v;
    };

    // ---- Прогрев (не учитывается) ----
    if (cfg.warmup) {
        if (cfg.run_serial) {
            say("warm-up: serial sort (not timed)");
            restore();
            inspect();
            sort_serial(nullptr);
            dirty = true;
        }
        if (cfg.run_parallel) {
            say("warm-up: parallel sort (not timed)");
            restore();
            inspect();
            sort_parallel(nullptr);
            dirty = true;
        }
    }

    // ---- Измеряемые повторы ----
    double best_serial = std::numeric_limits<double>::infinity();
    double best_parallel = std::numeric_limits<double>::infinity();
    for (int rep = 0; rep < cfg.repeats; ++rep) {
        const bool last = (rep == cfg.repeats - 1);
        const std::string tag = "run " + std::to_string(rep + 1) + "/" + std::to_string(cfg.repeats) + ": ";
        if (cfg.run_serial) {
            restore();  // вне замера
            inspect();
            say(tag + "serial sort of " + std::to_string(n) + " keys...");
            PhaseTimes pt;
            const double s = time_seconds([&] { sort_serial(&pt); });
            dirty = true;
            say(tag + "serial sort done in " + format_seconds(s));
            r.serial.samples.push_back(s);
            if (s < best_serial) {
                best_serial = s;
                r.serial_phases = pt;
            }
            if (cfg.verify && last) r.serial_check = check("serial");
        }
        if (cfg.run_parallel) {
            restore();  // вне замера
            inspect();
            say(tag + "parallel sort of " + std::to_string(n) + " keys (" + std::to_string(cfg.threads) + " threads requested)...");
            PhaseTimes pt;
            const double s = time_seconds([&] { sort_parallel(&pt); });
            dirty = true;
            say(tag + "parallel sort done in " + format_seconds(s));
            r.parallel.samples.push_back(s);
            if (s < best_parallel) {
                best_parallel = s;
                r.parallel_phases = pt;
            }
            if (cfg.verify && last) r.parallel_check = check("parallel");
        }
    }
    finalize(r.serial);
    finalize(r.parallel);

    r.serial_memory = estimate_memory(n, 1, cfg.algorithm, false);
    r.parallel_memory = estimate_memory(n, r.parallel_phases.threads_used, cfg.algorithm, true);

    if (sorted_out != nullptr) sorted_out->swap(a);  // отсортированный результат последнего запуска
    return r;
}

// ---------------------------------------------------------------------------
// Форматирование
// ---------------------------------------------------------------------------

std::string format_bytes(std::uint64_t bytes) {
    static const char* const units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    char buf[64];
    if (bytes < 1024) {
        std::snprintf(buf, sizeof buf, "%llu B", static_cast<unsigned long long>(bytes));
        return buf;
    }
    double v = static_cast<double>(bytes);
    int u = 0;
    while (v >= 1024.0 && u < 4) {
        v /= 1024.0;
        ++u;
    }
    std::snprintf(buf, sizeof buf, "%.2f %s", v, units[u]);
    return buf;
}

std::string format_seconds(double s) {
    char buf[48];
    // Для очень коротких времён (маленькие N) три знака после запятой дали бы «0.000 s».
    if (s == 0.0 || s >= 0.01) std::snprintf(buf, sizeof buf, "%.3f s", s);
    else std::snprintf(buf, sizeof buf, "%.6f s", s);
    return buf;
}

namespace {

std::string format_ratio(double x) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.2fx", x);
    return buf;
}

std::string format_percent(double x) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f%%", x);
    return buf;
}

std::string series_suffix(const SeriesStats& s, int repeats) {
    if (repeats <= 1) return " (single run)";
    return " (best of " + std::to_string(repeats) + "; mean " + format_seconds(s.mean) + ", sd " +
           format_seconds(s.stddev) + ")";
}

const char* algorithm_description(Algorithm a) {
    return a == Algorithm::Scatter ? "scatter (stable counting sort, single 16-bit pass, output buffer)"
                                   : "fill (histogram + range fill, in place, no output buffer)";
}

std::string phases_line(const PhaseTimes& p, const char* merge_name, const char* move_name) {
    return std::string("count ") + format_seconds(p.count_s) + " | " + merge_name + " " +
           format_seconds(p.prefix_s) + " | " + move_name + " " + format_seconds(p.move_s);
}

}  // namespace

void print_report(std::ostream& os, const RunResult& r, const ReportContext& ctx) {
    const RunConfig& c = r.config;
    auto kv = [&os](const std::string& k, const std::string& v) {
        os << std::left << std::setw(15) << k << v << '\n';
    };
    auto sub = [&os](const std::string& k, const std::string& v) {
        os << "  " << std::left << std::setw(13) << k << v << '\n';
    };

    os << ctx.title << '\n' << std::string(ctx.title.size(), '=') << "\n\n";

    kv("Elements:", std::to_string(r.n));
    kv("Data type:", "uint16_t");
    kv("Key range:", "0..65535");
    kv("Algorithm:", algorithm_description(c.algorithm));
    if (!ctx.input_description.empty()) kv("Input:", ctx.input_description);
    if (!ctx.load_label.empty() && ctx.load_seconds >= 0.0) {
        kv(ctx.load_label + ":", format_seconds(ctx.load_seconds) + " (not included in sorting time)");
    }
    if (!ctx.output_description.empty()) {
        kv("Result file:", ctx.output_description + " (written in " + format_seconds(ctx.write_seconds) +
                               ", not included in sorting time)");
    }
    if (c.run_parallel) {
        kv("Threads:", std::to_string(c.threads) + " requested, " + std::to_string(r.parallel_phases.threads_used) +
                           " used (hardware threads: " + std::to_string(ctx.hardware_threads) + ")");
    } else {
        kv("Threads:", "1 (serial mode)");
    }
    kv("Repeats:", std::to_string(c.repeats) + (c.warmup ? " (+1 untimed warm-up)" : " (no warm-up)") +
                       (ctx.repeats_note.empty() ? std::string() : "; " + ctx.repeats_note));

    // ---- Корректность ----
    os << '\n';
    if (c.verify) {
        using Flag = bool Verification::*;
        auto all = [&](Flag f) {
            return (!c.run_serial || r.serial_check.*f) && (!c.run_parallel || r.parallel_check.*f);
        };
        auto yn = [](bool b) { return std::string(b ? "YES" : "NO"); };
        os << "Correctness:\n";
        sub("Sorted:", yn(all(&Verification::sorted)));
        sub("Count check:", yn(all(&Verification::count_ok)));
        sub("Checksum:", yn(all(&Verification::checksum_ok)));
        sub("Histogram:", yn(all(&Verification::histogram_ok)) + " (exact multiset equality)");
        if (r.n <= kReferenceSortLimit) {
            sub("std::sort:", yn(all(&Verification::reference_ok)) + " (result compared with std::sort, N <= " +
                                  std::to_string(kReferenceSortLimit) + ")");
        } else {
            sub("std::sort:", "skipped (N > " + std::to_string(kReferenceSortLimit) + ")");
        }
        const bool serial_bad = c.run_serial && !r.serial_check.passed();
        const bool parallel_bad = c.run_parallel && !r.parallel_check.passed();
        if (serial_bad || parallel_bad) {
            sub("FAILED in:", std::string(serial_bad ? "serial " : "") + (parallel_bad ? "parallel" : ""));
        }
    } else {
        os << "Correctness:\n";
        sub("Not checked", "(use --verify)");
    }

    // ---- Производительность ----
    os << "\nPerformance:\n";
    if (r.n == 0) {
        sub("Note:", "empty array, nothing to measure");
    }
    if (c.run_serial) sub("Serial:", format_seconds(r.serial.best) + series_suffix(r.serial, c.repeats));
    if (c.run_parallel) sub("Parallel:", format_seconds(r.parallel.best) + series_suffix(r.parallel, c.repeats));
    if (c.run_serial && c.run_parallel) {
        if (r.serial.best > 0.0 && r.parallel.best > 0.0 && r.n > 0) {
            const int used = std::max(1, r.parallel_phases.threads_used);
            const double speedup = r.serial.best / r.parallel.best;
            const double speedup_mean = (r.parallel.mean > 0.0) ? r.serial.mean / r.parallel.mean : 0.0;
            sub("Speedup:", format_ratio(speedup) + " (best / best), " + format_ratio(speedup_mean) +
                                " (mean / mean)");
            sub("Efficiency:", format_percent(speedup / static_cast<double>(used) * 100.0) + " (speedup / " +
                                   std::to_string(used) + " thread" + (used == 1 ? "" : "s") + ")");
            if (used == 1) {
                sub("Note:", "1 thread: the ratio shows the overhead of the parallel algorithm vs serial");
            }
        } else {
            sub("Speedup:", "n/a (time is too small to measure)");
            sub("Efficiency:", "n/a");
        }
    }

    // ---- Фазы ----
    if (r.n > 0) {
        const char* move_name = (c.algorithm == Algorithm::Scatter) ? "scatter" : "fill";
        os << "\nPhase breakdown (fastest run, omp_get_wtime):\n";
        if (c.run_serial) sub("Serial:", phases_line(r.serial_phases, "prefix", move_name));
        if (c.run_parallel) sub("Parallel:", phases_line(r.parallel_phases, "merge+prefix", move_name));
    }

    // ---- Память ----
    const MemoryEstimate& m = c.run_parallel ? r.parallel_memory : r.serial_memory;
    os << "\nMemory (" << (c.run_parallel ? "parallel" : "serial") << " version):\n";
    sub("Input:", format_bytes(m.input_bytes));
    sub("Output:", m.output_bytes != 0 ? format_bytes(m.output_bytes) : std::string("none (in place)"));
    if (c.run_parallel) {
        sub("Histogram:", format_bytes(m.histogram_bytes) + " (" + std::to_string(r.parallel_phases.threads_used) +
                              " x 65536 x " + std::to_string(sizeof(Count)) + " B)");
        sub("Aux tables:", format_bytes(m.aux_bytes));
    } else {
        sub("Histogram:", format_bytes(m.histogram_bytes) + " (65537 x " + std::to_string(sizeof(Count)) + " B)");
    }
    sub("Total:", format_bytes(m.total()));
    if (r.restore_bytes != 0) {
        sub("Restore copy:", "+" + format_bytes(r.restore_bytes) + " (pristine input to restore between runs)");
    } else if (c.algorithm == Algorithm::Scatter && count_sorts(c) > 1) {
        sub("Restore:", "no extra copy (the sort leaves its input intact; arrays are swapped)");
    }
}

}  // namespace radix
