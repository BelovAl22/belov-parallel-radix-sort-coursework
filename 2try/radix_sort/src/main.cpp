// main.cpp — консольное приложение sort_app: последовательная и OpenMP-параллельная цифровая
// сортировка массива uint16_t, бенчмарк, генерация тестовых данных.
//
// Вывод: итоговый отчёт — в stdout; статусные строки с прогрессом («reading…», «run 1/1…») — в stderr
// (их можно отключить флагом --quiet). Благодаря этому на огромных массивах всегда видно, что программа
// работает, а отчёт можно перенаправить в файл или в grep.
//
// Коды возврата: 0 — успех; 1 — ошибка выполнения (ввод-вывод, память, проверка не пройдена);
//                2 — неверные аргументы командной строки.
#include <omp.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "benchmark.hpp"
#include "data_gen.hpp"
#include "io.hpp"
#include "radix_sort.hpp"
#include "sysinfo.hpp"
#include "verify.hpp"

namespace {

using namespace radix;

class UsageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class Mode { Serial, Parallel, Benchmark, Generate };

struct Options {
    Mode mode = Mode::Benchmark;
    std::string input_path;
    std::string output_path;
    bool have_size = false;
    std::size_t size = 0;
    Distribution distribution = Distribution::Uniform;
    std::uint64_t seed = 12345;
    int threads = 0;   // 0 = значение по умолчанию OpenMP (OMP_NUM_THREADS или число ядер)
    int repeats = 0;   // 0 = автоматически (3, а для N >= 250 000 000 — 1)
    int warmup = -1;   // -1 = автоматически (включён, а для N >= 250 000 000 — выключен); 0 / 1 — явно
    bool verify = false;
    bool quiet = false;
    bool force = false;
    Algorithm algorithm = Algorithm::Scatter;
    FileFormat format = FileFormat::Auto;
    bool help = false;
};

const char* const kUsage =
    R"(sort_app - serial vs OpenMP-parallel radix sort of uint16_t arrays (0..65535)

Usage:
  sort_app --mode serial    --input input.txt [--verify]
  sort_app --mode parallel  --input input.txt --threads 8 [--verify]
  sort_app --mode benchmark --generate 10000000 --threads 16 [--repeats 3] [--verify]
  sort_app --mode generate  --size 10000000 --output input.txt [--seed 123]

Modes (--mode):
  serial        run the serial reference sort
  parallel      run the OpenMP parallel sort
  benchmark     run both on identical data, report speedup and efficiency (default)
  generate      write a generated test array to --output and exit

Data source (serial / parallel / benchmark):
  --input FILE        read keys from FILE (one integer 0..65535 per line, or raw uint16 for *.bin)
  --generate N        generate N keys in memory instead of reading a file (alias: --size N)
                      N accepts decimal suffixes k, M, G: 1M = 1000000, 1G = 1000000000
  --distribution D    uniform | random | sorted | reverse | few-values | zeros | max | two-values
                      (default: uniform; "random" is bell-shaped, "few-values" has 16 distinct values)
  --seed S            generator seed, makes runs reproducible (default: 12345)
  --format F          text | binary | auto  (default: auto; *.bin = binary, anything else = text)

Execution:
  --threads T         number of OpenMP threads of the parallel version, 1..1024
                      (default: OMP_NUM_THREADS, otherwise all hardware threads)
  --repeats R         number of timed repetitions (default: 3; but 1 when N >= 250000000)
  --warmup            force the untimed warm-up run (default: on; but off when N >= 250000000)
  --no-warmup         skip the untimed warm-up run
  --algorithm A       scatter (default) - stable counting sort with an output buffer
                      fill              - in-place "histogram + range fill" (extension, faster, less memory)
  --verify            verify results: sorted, element count, checksums, exact histogram,
                      and comparison with std::sort for N <= 10000000

Output:
  --output FILE       serial/parallel/benchmark: write the sorted array to FILE (not timed)
                      generate: destination of the generated array
  --quiet, -q         do not print progress lines (stderr); the final report is always printed
  --force             start even if the memory pre-check says there is not enough free RAM
  --help, -h          show this help

Progress and warnings go to stderr, the final report goes to stdout.

Memory: sorting keeps two arrays of N * 2 bytes (input + output buffer; 1 array for a single
--algorithm fill run). For N = 10^9 that is about 3.7 GiB (1.9 GiB). Before allocating anything the
program compares this with the free RAM and refuses to start if it does not fit: once the OS starts
swapping, the random writes of the sort turn a seconds-long run into hours.

Timing covers only the sort call itself; generation, file I/O, restoring the input between
repetitions and verification are never included.

Exit codes: 0 success, 1 runtime error / failed verification, 2 invalid arguments.
)";

// ---------------------------------------------------------------------------
// Разбор аргументов
// ---------------------------------------------------------------------------

// Разбор целого без знака. allow_suffix: суффиксы k/M/G (десятичные). Иначе допускается префикс 0x.
std::uint64_t parse_u64(const std::string& text, const std::string& option, bool allow_suffix) {
    std::string_view s = text;
    std::uint64_t mult = 1;
    if (allow_suffix && !s.empty()) {
        switch (s.back()) {
            case 'k': case 'K': mult = 1000ull; break;
            case 'm': case 'M': mult = 1000000ull; break;
            case 'g': case 'G': mult = 1000000000ull; break;
            default: break;
        }
        if (mult != 1) s.remove_suffix(1);
    }
    int base = 10;
    if (!allow_suffix && s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        base = 16;
        s.remove_prefix(2);
    }
    std::uint64_t value = 0;
    const char* const end = s.data() + s.size();
    const auto res = std::from_chars(s.data(), end, value, base);
    if (s.empty() || res.ec != std::errc() || res.ptr != end) {
        throw UsageError("invalid value for " + option + ": '" + text + "'");
    }
    if (mult != 1 && value > UINT64_MAX / mult) {
        throw UsageError("value for " + option + " is too large: '" + text + "'");
    }
    return value * mult;
}

Options parse_args(int argc, char** argv) {
    Options o;
    const std::vector<std::string> args(argv + 1, argv + argc);
    for (std::size_t i = 0; i < args.size(); ++i) {
        std::string arg = args[i];
        std::string inline_value;
        bool has_inline = false;
        if (arg.rfind("--", 0) == 0) {
            const std::size_t eq = arg.find('=');
            if (eq != std::string::npos) {
                inline_value = arg.substr(eq + 1);
                arg.resize(eq);
                has_inline = true;
            }
        }
        // Значение опции: из «--opt=value» либо из следующего аргумента.
        auto value = [&]() -> std::string {
            if (has_inline) return inline_value;
            if (i + 1 >= args.size()) throw UsageError("option " + arg + " requires a value");
            return args[++i];
        };
        auto flag = [&]() {
            if (has_inline) throw UsageError("option " + arg + " does not take a value");
        };

        if (arg == "--help" || arg == "-h") {
            flag();
            o.help = true;
        } else if (arg == "--mode") {
            const std::string v = value();
            if (v == "serial") o.mode = Mode::Serial;
            else if (v == "parallel") o.mode = Mode::Parallel;
            else if (v == "benchmark") o.mode = Mode::Benchmark;
            else if (v == "generate") o.mode = Mode::Generate;
            else throw UsageError("invalid --mode '" + v + "' (expected serial|parallel|benchmark|generate)");
        } else if (arg == "--input") {
            o.input_path = value();
        } else if (arg == "--output") {
            o.output_path = value();
        } else if (arg == "--generate" || arg == "--size") {
            const std::uint64_t n = parse_u64(value(), arg, true);
            if (n > std::vector<Key>().max_size()) throw UsageError("size is too large for this platform");
            o.size = static_cast<std::size_t>(n);
            o.have_size = true;
        } else if (arg == "--distribution") {
            const std::string v = value();
            if (!parse_distribution(v, o.distribution)) {
                throw UsageError("invalid --distribution '" + v + "' (expected " + distribution_names() + ")");
            }
        } else if (arg == "--seed") {
            o.seed = parse_u64(value(), arg, false);
        } else if (arg == "--threads") {
            const std::uint64_t t = parse_u64(value(), arg, false);
            if (t < 1 || t > static_cast<std::uint64_t>(kMaxThreads)) {
                throw UsageError("--threads must be in [1, " + std::to_string(kMaxThreads) + "]");
            }
            o.threads = static_cast<int>(t);
        } else if (arg == "--repeats") {
            const std::uint64_t r = parse_u64(value(), arg, false);
            if (r < 1 || r > 1000) throw UsageError("--repeats must be in [1, 1000]");
            o.repeats = static_cast<int>(r);
        } else if (arg == "--warmup") {
            flag();
            o.warmup = 1;
        } else if (arg == "--no-warmup") {
            flag();
            o.warmup = 0;
        } else if (arg == "--verify") {
            flag();
            o.verify = true;
        } else if (arg == "--quiet" || arg == "-q") {
            flag();
            o.quiet = true;
        } else if (arg == "--force") {
            flag();
            o.force = true;
        } else if (arg == "--algorithm") {
            const std::string v = value();
            if (v == "scatter") o.algorithm = Algorithm::Scatter;
            else if (v == "fill") o.algorithm = Algorithm::Fill;
            else throw UsageError("invalid --algorithm '" + v + "' (expected scatter|fill)");
        } else if (arg == "--format") {
            const std::string v = value();
            if (!parse_file_format(v, o.format)) throw UsageError("invalid --format '" + v + "' (expected text|binary|auto)");
        } else {
            throw UsageError("unknown option '" + args[i] + "'");
        }
    }
    return o;
}

void validate(const Options& o) {
    if (o.mode == Mode::Generate) {
        if (!o.have_size) throw UsageError("--mode generate requires --size N");
        if (o.output_path.empty()) throw UsageError("--mode generate requires --output FILE");
        if (!o.input_path.empty()) throw UsageError("--input cannot be used with --mode generate");
        return;
    }
    if (o.input_path.empty() && !o.have_size) {
        throw UsageError("specify the data source: --input FILE or --generate N");
    }
    if (!o.input_path.empty() && o.have_size) {
        throw UsageError("--input and --generate/--size are mutually exclusive");
    }
}

// ---------------------------------------------------------------------------
// Журнал прогресса (stderr)
// ---------------------------------------------------------------------------

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

class Log {
public:
    explicit Log(bool quiet) : quiet_(quiet), t0_(std::chrono::steady_clock::now()) {}

    // Статусная строка с временем от старта программы. Выводится сразу (std::cerr не буферизуется).
    void info(const std::string& msg) const {
        if (quiet_) return;
        char prefix[32];
        std::snprintf(prefix, sizeof prefix, "[%7.1f s] ", seconds_since(t0_));
        std::cerr << prefix << msg << '\n';
    }
    // Предупреждения печатаются всегда, даже с --quiet.
    void warn(const std::string& msg) const { std::cerr << "warning: " << msg << '\n'; }

private:
    bool quiet_;
    std::chrono::steady_clock::time_point t0_;
};

std::string percent(double done, double total) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "%.0f%%", total > 0 ? 100.0 * done / total : 0.0);
    return buf;
}

// ---------------------------------------------------------------------------
// Предварительная проверка памяти
// ---------------------------------------------------------------------------

// Сравнивает потребность в памяти с доступной ОЗУ. Если не хватает — отказывается стартовать
// (бросает исключение), потому что при вытеснении в swap случайные записи сортировки превращают
// секунды в часы. `slack` > 1 делает проверку мягче (используется для оценок, а не точных значений).
void preflight(const Log& log, const MemoryInfo& mem, std::uint64_t required, double slack, bool force,
               const std::string& what) {
    if (!mem.known() || required == 0) return;
    if (static_cast<double>(required) > static_cast<double>(mem.available) * slack) {
        std::string msg = what + " needs about " + format_bytes(required) + " of RAM, but only " +
                          format_bytes(mem.available) + " is available (installed: " + format_bytes(mem.total) +
                          "). The sort scatters elements randomly over the whole output array, so once memory runs "
                          "out and the OS starts swapping, a run of seconds turns into hours. ";
        if (force) {
            log.warn(msg + "Continuing because of --force.");
            return;
        }
        throw std::runtime_error(
            msg + "Options: use --algorithm fill with --mode serial or --mode parallel (it needs only one array), "
                  "reduce the size, close other programs, or add --force to try anyway.");
    }
    if (static_cast<double>(required) > 0.85 * static_cast<double>(mem.available)) {
        log.warn("memory is tight: the run needs about " + format_bytes(required) + " of the " +
                 format_bytes(mem.available) + " available; other programs may push it into swap");
    }
}

// Настройки запуска; повторы и прогрев берутся по умолчанию в зависимости от N, если не заданы явно.
RunConfig make_config(const Options& o, int threads, std::size_t n, std::string* note) {
    const RunSettings def = default_run_settings(n);
    RunConfig cfg;
    cfg.algorithm = o.algorithm;
    cfg.run_serial = (o.mode != Mode::Parallel);
    cfg.run_parallel = (o.mode != Mode::Serial);
    cfg.threads = threads;
    cfg.repeats = (o.repeats > 0) ? o.repeats : def.repeats;
    cfg.warmup = (o.warmup >= 0) ? (o.warmup != 0) : def.warmup;
    cfg.verify = o.verify;
    if (note != nullptr && n >= kLargeInputThreshold && (o.repeats == 0 || o.warmup < 0)) {
        *note = "defaults for N >= " + std::to_string(kLargeInputThreshold) + "; see --repeats / --warmup";
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Режимы
// ---------------------------------------------------------------------------

int run_generate(const Options& o) {
    const Log log(o.quiet);
    const FileFormat fmt = resolve_format(o.format, o.output_path);
    const std::size_t n = o.size;
    constexpr std::size_t kChunk = std::size_t{1} << 24;  // 16M ключей = 32 МиБ за раз
    std::vector<Key> chunk(std::min(n, kChunk));

    log.info("generating " + std::to_string(n) + " keys (" + to_string(o.distribution) + ", seed " +
             std::to_string(o.seed) + ") into '" + o.output_path + "' as " + to_string(fmt));
    const auto t0 = std::chrono::steady_clock::now();
    auto last_report = t0;
    KeyWriter writer(o.output_path, fmt);
    for (std::size_t first = 0; first < n; first += chunk.size()) {
        const std::size_t count = std::min(chunk.size(), n - first);
        generate_keys(chunk.data(), first, count, n, o.distribution, o.seed);
        writer.write(chunk.data(), count);
        if (seconds_since(last_report) >= 2.0) {
            last_report = std::chrono::steady_clock::now();
            log.info("generating: " + std::to_string(first + count) + " / " + std::to_string(n) + " keys (" +
                     percent(static_cast<double>(first + count), static_cast<double>(n)) + ")");
        }
    }
    writer.close();
    const double elapsed = seconds_since(t0);

    std::cout << "Generated " << n << " keys\n"
              << "  Distribution: " << to_string(o.distribution) << " (seed " << o.seed << ")\n"
              << "  Format:       " << to_string(fmt) << '\n'
              << "  File:         " << o.output_path << " (" << format_bytes(writer.bytes_written()) << ")\n"
              << "  Time:         " << format_seconds(elapsed) << " (generation + writing)\n";
    return 0;
}

// Запись результата порциями — с прогрессом (текстовый файл на 10^9 чисел — это ~6 ГБ).
void write_with_progress(const std::string& path, FileFormat format, const std::vector<Key>& v, const Log& log) {
    KeyWriter writer(path, format);
    constexpr std::size_t kChunk = std::size_t{1} << 24;
    auto last_report = std::chrono::steady_clock::now();
    for (std::size_t first = 0; first < v.size(); first += kChunk) {
        const std::size_t count = std::min(kChunk, v.size() - first);
        writer.write(v.data() + first, count);
        if (seconds_since(last_report) >= 2.0) {
            last_report = std::chrono::steady_clock::now();
            log.info("writing: " + percent(static_cast<double>(first + count), static_cast<double>(v.size())));
        }
    }
    writer.close();
}

int run_sort(const Options& o, int threads) {
    const Log log(o.quiet);
    ReportContext ctx;
    ctx.hardware_threads = omp_get_num_procs();
    const std::size_t kMaxVector = std::vector<Key>().max_size();

    std::vector<Key> data;

    // N, не помещающееся в счётчики, отвергаем сразу — до генерации/чтения гигабайт данных.
    auto check_counter_capacity = [](std::uint64_t n) {
        if (n > max_supported_elements()) {
            throw std::runtime_error("N = " + std::to_string(n) + " exceeds the limit of this build (" +
                                     std::to_string(max_supported_elements()) +
                                     " keys, " + std::to_string(sizeof(Count) * 8) +
                                     "-bit counters); rebuild with -DRADIX_WIDE_COUNTERS=ON");
        }
    };

    // ---- Подготовка данных (в измеряемое время не входит) ----
    if (!o.input_path.empty()) {
        const FileFormat fmt = resolve_format(o.format, o.input_path);
        // Размер файла известен заранее: проверяем память ДО многоминутного чтения. Число ключей в текстовом
        // файле — оценка (по трём фрагментам), поэтому проверка мягкая (+15 %).
        const std::uint64_t estimate = estimate_key_count(o.input_path, fmt);
        if (fmt == FileFormat::Binary) check_counter_capacity(estimate);  // для binary число ключей точное
        if (estimate > 0 && estimate <= kMaxVector) {
            const std::size_t n_est = static_cast<std::size_t>(estimate);
            const RunConfig probe = make_config(o, threads, n_est, nullptr);
            preflight(log, query_memory(), estimate_experiment_memory(n_est, probe), 1.15, o.force,
                      "sorting about " + std::to_string(n_est) + " keys");
        }
        log.info("reading '" + o.input_path + "' (" + to_string(fmt) + ")...");
        const auto t0 = std::chrono::steady_clock::now();
        const ProgressFn progress = [&](std::uint64_t done, std::uint64_t total) {
            const double sec = seconds_since(t0);
            const double rate = sec > 0.0 ? static_cast<double>(done) / sec : 0.0;
            const double eta = (rate > 0.0 && total > done) ? static_cast<double>(total - done) / rate : 0.0;
            char tail[64];
            std::snprintf(tail, sizeof tail, ", about %.0f s left", eta);
            log.info("reading: " + format_bytes(done) + " of " + format_bytes(total) + " (" +
                     percent(static_cast<double>(done), static_cast<double>(total)) + "), " +
                     format_bytes(static_cast<std::uint64_t>(rate)) + "/s" + tail);
        };
        const ReadInfo info = read_keys(o.input_path, o.format, data, progress);
        ctx.load_seconds = seconds_since(t0);
        ctx.load_label = "Reading";
        ctx.input_description = "file '" + o.input_path + "' (" + to_string(info.format) + ", " +
                                format_bytes(info.file_bytes) + ")";
        log.info("read " + std::to_string(data.size()) + " keys in " + format_seconds(ctx.load_seconds));
    } else {
        // Размер известен точно: проверяем лимит счётчиков и память до выделения чего бы то ни было.
        check_counter_capacity(o.size);
        const RunConfig probe = make_config(o, threads, o.size, nullptr);
        preflight(log, query_memory(), estimate_experiment_memory(o.size, probe), 1.0, o.force,
                  "sorting " + std::to_string(o.size) + " keys");
        log.info("generating " + std::to_string(o.size) + " keys (" + to_string(o.distribution) + ", seed " +
                 std::to_string(o.seed) + ")...");
        const auto t0 = std::chrono::steady_clock::now();
        generate_keys(data, o.size, o.distribution, o.seed);
        ctx.load_seconds = seconds_since(t0);
        ctx.load_label = "Generation";
        ctx.input_description = std::string(to_string(o.distribution)) + ", seed " + std::to_string(o.seed) + " (generated)";
        log.info("generated in " + format_seconds(ctx.load_seconds));
    }

    // ---- Настройки запуска и точная проверка памяти (теперь N известно) ----
    const std::size_t n = data.size();
    check_counter_capacity(n);
    RunConfig cfg = make_config(o, threads, n, &ctx.repeats_note);
    if (!o.input_path.empty()) {
        // Массив данных уже занимает память — требуется только то, чего ещё не хватает.
        const std::uint64_t need = estimate_experiment_memory(n, cfg);
        const std::uint64_t have = static_cast<std::uint64_t>(data.capacity()) * sizeof(Key);
        preflight(log, query_memory(), need > have ? need - have : 0, 1.0, o.force,
                  "sorting " + std::to_string(n) + " keys (in addition to the loaded array)");
    }
    cfg.progress = [&log](const std::string& msg) { log.info(msg); };

    switch (o.mode) {
        case Mode::Serial: ctx.title = "Serial Radix Sort"; break;
        case Mode::Parallel: ctx.title = "Parallel Radix Sort"; break;
        default: ctx.title = "Parallel Radix Sort (benchmark: serial vs parallel)"; break;
    }

    // ---- Эксперимент ----
    std::vector<Key> sorted;
    const RunResult result = run_experiment(std::move(data), cfg, o.output_path.empty() ? nullptr : &sorted);

    // ---- Запись результата (тоже вне измерения) ----
    if (!o.output_path.empty()) {
        log.info("writing the sorted array to '" + o.output_path + "'...");
        const auto t0 = std::chrono::steady_clock::now();
        write_with_progress(o.output_path, resolve_format(o.format, o.output_path), sorted, log);
        ctx.write_seconds = seconds_since(t0);
        ctx.output_description = "'" + o.output_path + "'";
    }

    log.info("done; the report follows");
    print_report(std::cout, result, ctx);

    if (o.verify) {
        const bool serial_bad = cfg.run_serial && !result.serial_check.passed();
        const bool parallel_bad = cfg.run_parallel && !result.parallel_check.passed();
        if (serial_bad || parallel_bad) {
            std::cerr << "VERIFICATION FAILED\n";
            return 1;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse_args(argc, argv);
        if (o.help) {
            std::cout << kUsage;
            return 0;
        }
        validate(o);

        const int threads = (o.threads > 0) ? o.threads : std::max(1, omp_get_max_threads());
        omp_set_num_threads(threads);  // по умолчанию и для генерации данных / копирования

        return (o.mode == Mode::Generate) ? run_generate(o) : run_sort(o, threads);
    } catch (const UsageError& e) {
        std::cerr << "error: " << e.what() << "\nTry 'sort_app --help' for usage.\n";
        return 2;
    } catch (const std::bad_alloc&) {
        std::cerr << "error: not enough memory for the requested array size\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
