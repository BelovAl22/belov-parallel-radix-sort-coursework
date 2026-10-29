// test_radix_sort.cpp — самодостаточный набор тестов (без внешних библиотек).
//
// Запуск:   radix_tests            — быстрый набор (размеры до 10^7)
//           radix_tests --large    — дополнительно массив из 10^8 элементов (~400 МБ памяти)
//
// Что проверяется:
//   * все четыре алгоритма (serial/parallel × scatter/fill) против std::sort на крайних случаях:
//     N = 0, 1, 2, N < T, N = T, N > T; все 0; все 65535; одно значение; два значения;
//     отсортированный; обратный; случайный; несколько значений;
//   * инварианты диапазонов записи параллельной версии (белый ящик: курсоры в workspace);
//   * повторное использование workspace / буферов между вызовами с разными N и T;
//   * размеры 10^3, 10^4, 10^6, 10^7 (и 10^8 с --large);
//   * детерминированность генератора, ввод-вывод (в т. ч. числа, разрезанные границей блока чтения),
//     модуль проверки, вспомогательные функции;
//   * экономия памяти бенчмарка: scatter не портит вход, а КАЖДЫЙ запуск (прогрев, serial, parallel,
//     все повторы) стартует ровно с исходных данных; оценка памяти, настройки по умолчанию для огромных N.
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <iterator>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#include "benchmark.hpp"
#include "data_gen.hpp"
#include "io.hpp"
#include "radix_sort.hpp"
#include "sysinfo.hpp"
#include "verify.hpp"

using namespace radix;

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failures;                                                        \
            std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                        \
    } while (0)

#define CHECK_MSG(cond, msg)                                                                 \
    do {                                                                                     \
        ++g_checks;                                                                          \
        if (!(cond)) {                                                                       \
            ++g_failures;                                                                    \
            std::fprintf(stderr, "  FAIL %s:%d: %s [%s]\n", __FILE__, __LINE__, #cond, std::string(msg).c_str()); \
        }                                                                                    \
    } while (0)

#define CHECK_THROWS(expr, ExcType)                                                \
    do {                                                                           \
        ++g_checks;                                                                \
        bool thrown_ = false;                                                      \
        try {                                                                      \
            expr;                                                                  \
        } catch (const ExcType&) {                                                 \
            thrown_ = true;                                                        \
        } catch (...) {                                                            \
        }                                                                          \
        if (!thrown_) {                                                            \
            ++g_failures;                                                          \
            std::fprintf(stderr, "  FAIL %s:%d: expected %s from %s\n", __FILE__, __LINE__, #ExcType, #expr); \
        }                                                                          \
    } while (0)

// ---------------------------------------------------------------------------
// Вспомогательное
// ---------------------------------------------------------------------------

enum Pattern { kZeros, kMax, kSameValue, kTwoValues, kSorted, kReverse, kRandom, kFewValues, kAlternating, kPatternCount };

const char* pattern_name(int p) {
    static const char* names[] = {"zeros", "max", "same", "two-values", "sorted", "reverse", "random", "few-values", "alternating"};
    return names[p];
}

std::vector<Key> make_pattern(int kind, std::size_t n, std::mt19937_64& rng) {
    std::vector<Key> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t ramp = n ? (static_cast<std::uint64_t>(i) * 65536u) / n : 0;
        switch (kind) {
            case kZeros: v[i] = 0; break;
            case kMax: v[i] = 65535; break;
            case kSameValue: v[i] = 31337; break;
            case kTwoValues: v[i] = (rng() & 1) ? Key{65535} : Key{0}; break;
            case kSorted: v[i] = static_cast<Key>(ramp); break;
            case kReverse: v[i] = static_cast<Key>(65535 - ramp); break;
            case kRandom: v[i] = static_cast<Key>(rng()); break;
            case kFewValues: v[i] = static_cast<Key>((rng() % 16) * 4097); break;
            default: v[i] = (i % 2 == 0) ? Key{0} : Key{65535}; break;  // kAlternating
        }
    }
    return v;
}

// Прогоняет ВСЕ алгоритмы на одном входе и сверяет с std::sort.
void check_all_algorithms(const std::vector<Key>& input, const std::vector<int>& thread_counts,
                          ParallelRadixWorkspace& ws, const std::string& label) {
    std::vector<Key> expected = input;
    std::sort(expected.begin(), expected.end());

    {  // serial scatter; буфер «грязный» и неподходящего размера — функция обязана справиться
        std::vector<Key> d = input;
        std::vector<Key> b(input.size() / 2 + 3, Key{0xABCD});
        serial_radix_sort(d, b);
        CHECK_MSG(d == expected, label + " serial scatter");
    }
    {  // serial fill
        std::vector<Key> d = input;
        serial_histogram_fill_sort(d);
        CHECK_MSG(d == expected, label + " serial fill");
    }
    for (const int t : thread_counts) {
        const std::string tl = label + " T=" + std::to_string(t);
        {
            std::vector<Key> d = input;
            std::vector<Key> b;
            PhaseTimes pt;
            parallel_radix_sort(d, b, t, ws, &pt);
            CHECK_MSG(d == expected, tl + " parallel scatter");
            CHECK_MSG(pt.threads_used >= 1 && pt.threads_used <= t, tl + " threads_used in [1,T]");
            CHECK_MSG(input.empty() || static_cast<std::size_t>(pt.threads_used) <= input.size(), tl + " threads_used <= N");
        }
        {
            std::vector<Key> d = input;
            PhaseTimes pt;
            parallel_histogram_fill_sort(d, t, ws, &pt);
            CHECK_MSG(d == expected, tl + " parallel fill");
        }
    }
}

struct TempFile {
    std::filesystem::path path;
    explicit TempFile(const std::string& suffix) {
        static int counter = 0;
        path = std::filesystem::temp_directory_path() /
               ("radix_test_" + std::to_string(static_cast<long long>(std::chrono::steady_clock::now().time_since_epoch().count() % 1000000007LL)) +
                "_" + std::to_string(counter++) + suffix);
    }
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    std::string str() const { return path.string(); }
};

void write_raw(const TempFile& f, const std::string& bytes) {
    std::ofstream out(f.path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// Ожидаем IoError, текст которого содержит `needle`.
bool read_fails_with(const TempFile& f, const std::string& needle) {
    std::vector<Key> v;
    try {
        read_keys(f.str(), FileFormat::Text, v);
    } catch (const IoError& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Тесты
// ---------------------------------------------------------------------------

void test_block_begin() {
    for (std::size_t parts = 1; parts <= 40; ++parts) {
        for (const std::size_t n : {0u, 1u, 2u, 3u, 39u, 40u, 41u, 1000u, 65537u}) {
            CHECK(block_begin(0, parts, n) == 0);
            CHECK(block_begin(parts, parts, n) == n);
            std::size_t min_len = n, max_len = 0;
            for (std::size_t i = 0; i < parts; ++i) {
                const std::size_t len = block_begin(i + 1, parts, n) - block_begin(i, parts, n);
                min_len = std::min(min_len, len);
                max_len = std::max(max_len, len);
            }
            CHECK(max_len - min_len <= 1);  // блоки различаются не более чем на один элемент
        }
    }
    // Нет переполнения на больших N: наивная формула i * n / parts переполнила бы 32 бита.
    const std::size_t big = static_cast<std::size_t>(max_supported_elements());
    CHECK(block_begin(1024, 1024, big) == big);
    CHECK(block_begin(1023, 1024, big) < big);
}

void test_edge_sizes_and_patterns() {
    std::mt19937_64 rng(2024);
    ParallelRadixWorkspace ws;  // ОДНА рабочая область на весь тест: заодно проверяет повторное использование
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 33, 64, 100, 1000, 65535, 65536, 65537, 100000};
    const std::vector<int> threads = {1, 2, 3, 4, 7, 8, 16, 33, 64};  // в т. ч. T > N и T = N
    for (const std::size_t n : sizes) {
        for (int p = 0; p < kPatternCount; ++p) {
            const std::vector<Key> input = make_pattern(p, n, rng);
            check_all_algorithms(input, threads, ws, std::string(pattern_name(p)) + " N=" + std::to_string(n));
        }
    }
}

void test_threads_used_and_n_less_than_t() {
    ParallelRadixWorkspace ws;
    std::mt19937_64 rng(5);
    {
        std::vector<Key> d = make_pattern(kRandom, 3, rng), b;
        PhaseTimes pt;
        parallel_radix_sort(d, b, 16, ws, &pt);
        CHECK(pt.threads_used <= 3);  // N < T: лишние потоки не создаются
        CHECK(std::is_sorted(d.begin(), d.end()));
    }
    {
        std::vector<Key> d, b;
        PhaseTimes pt;
        parallel_radix_sort(d, b, 8, ws, &pt);  // N = 0
        CHECK(d.empty());
        CHECK(pt.threads_used == 1);
        CHECK(ws.capacity() >= 3);  // область не сжимается
    }
    {
        std::vector<Key> d = make_pattern(kRandom, 1000, rng), b;
        PhaseTimes pt;
        parallel_radix_sort(d, b, 1, ws, &pt);  // T = 1 — вырождение в один блок
        CHECK(pt.threads_used == 1);
        CHECK(std::is_sorted(d.begin(), d.end()));
    }
    {  // перегрузка без workspace
        std::vector<Key> d = make_pattern(kRandom, 5000, rng), b;
        std::vector<Key> e = d;
        std::sort(e.begin(), e.end());
        parallel_radix_sort(d, b, 5);
        CHECK(d == e);
    }
}

// Белый ящик: после parallel_radix_sort в строке t workspace лежит КОНЕЦ диапазона записи потока t
// для каждого значения v. По определению он равен global_start[v] + Σ_{t' <= t} local_count[t'][v].
// Совпадение с независимым расчётом доказывает: диапазоны потоков для v идут подряд в порядке
// номеров потоков (стабильность), не пересекаются и покрывают [start[v], start[v+1]) без дыр.
void check_write_ranges(const std::vector<Key>& input, int threads, const std::string& label) {
    ParallelRadixWorkspace ws;
    std::vector<Key> data = input, buf;
    PhaseTimes pt;
    parallel_radix_sort(data, buf, threads, ws, &pt);
    const std::size_t n = input.size();
    if (n == 0) return;
    const std::size_t nt = static_cast<std::size_t>(pt.threads_used);

    std::vector<std::vector<std::uint64_t>> cnt(nt, std::vector<std::uint64_t>(kKeyRange, 0));
    for (std::size_t t = 0; t < nt; ++t) {
        for (std::size_t i = block_begin(t, nt, n); i < block_begin(t + 1, nt, n); ++i) ++cnt[t][input[i]];
    }
    std::uint64_t global_start = 0;
    bool ok = true;
    for (std::size_t v = 0; v < kKeyRange && ok; ++v) {
        std::uint64_t running = global_start;
        for (std::size_t t = 0; t < nt; ++t) {
            running += cnt[t][v];
            if (ws.row(static_cast<int>(t))[v] != running) ok = false;
        }
        global_start = running;  // после последнего потока = начало значения v + 1
    }
    CHECK_MSG(ok, label + " write ranges are contiguous, ordered by thread, disjoint");
    CHECK_MSG(global_start == n, label + " ranges cover exactly [0, N)");
}

void test_write_range_invariants() {
    std::mt19937_64 rng(77);
    for (const int t : {1, 2, 3, 4, 7, 16}) {
        for (const std::size_t n : {1u, 5u, 100u, 5000u, 100000u}) {
            for (const int p : {kZeros, kTwoValues, kSorted, kRandom, kFewValues}) {
                check_write_ranges(make_pattern(p, n, rng), t,
                                   std::string(pattern_name(p)) + " N=" + std::to_string(n) + " T=" + std::to_string(t));
            }
        }
    }
}

void test_workspace_and_buffer_reuse() {
    std::mt19937_64 rng(31);
    ParallelRadixWorkspace ws;
    std::vector<Key> data, buffer;  // одни и те же объекты на всех итерациях (как в бенчмарке)
    struct Step { std::size_t n; int t; };
    const Step steps[] = {{1000, 4}, {100000, 2}, {5, 8}, {0, 3}, {300000, 16}, {1, 1}, {65537, 64}, {70000, 3}, {200000, 16}};
    for (int round = 0; round < 2; ++round) {
        for (const Step& s : steps) {
            const std::vector<Key> input = make_pattern(kRandom, s.n, rng);
            std::vector<Key> expected = input;
            std::sort(expected.begin(), expected.end());

            data = input;
            parallel_radix_sort(data, buffer, s.t, ws);
            CHECK_MSG(data == expected, "reuse parallel scatter N=" + std::to_string(s.n) + " T=" + std::to_string(s.t));

            data = input;
            serial_radix_sort(data, buffer);
            CHECK_MSG(data == expected, "reuse serial scatter N=" + std::to_string(s.n));

            data = input;
            parallel_histogram_fill_sort(data, s.t, ws);
            CHECK_MSG(data == expected, "reuse parallel fill N=" + std::to_string(s.n) + " T=" + std::to_string(s.t));
        }
    }
}

void test_invalid_arguments() {
    std::vector<Key> d = {3, 1, 2}, b;
    ParallelRadixWorkspace ws;
    CHECK_THROWS(parallel_radix_sort(d, d, 2), std::invalid_argument);          // data и buffer — один объект
    CHECK_THROWS(serial_radix_sort(d, d), std::invalid_argument);
    CHECK_THROWS(parallel_radix_sort(d, b, 0), std::invalid_argument);
    CHECK_THROWS(parallel_radix_sort(d, b, -3), std::invalid_argument);
    CHECK_THROWS(parallel_radix_sort(d, b, kMaxThreads + 1), std::invalid_argument);
    CHECK_THROWS(parallel_histogram_fill_sort(d, 0, ws), std::invalid_argument);
    CHECK_THROWS(ws.reserve(0), std::invalid_argument);
    CHECK((d == std::vector<Key>{3, 1, 2}));  // при ошибке данные не тронуты
}

void test_sizes(bool large) {
    struct Case { std::size_t n; bool compare_std_sort; };
    std::vector<Case> cases = {{1000, true}, {10000, true}, {1000000, true}, {10000000, false}};
    if (large) cases.push_back({100000000, false});
    ParallelRadixWorkspace ws;
    for (const Case& c : cases) {
        for (const Distribution dist : {Distribution::Uniform, Distribution::FewValues, Distribution::Reverse}) {
            if (c.n >= 100000000 && dist != Distribution::Uniform) continue;  // экономим время
            std::vector<Key> original;
            generate_keys(original, c.n, dist, 12345);
            const Fingerprint fp = make_fingerprint(original);
            std::vector<Key> reference;
            if (c.compare_std_sort) reference = make_reference_sorted(original);
            const std::vector<Key>* ref = c.compare_std_sort ? &reference : nullptr;
            const std::string label = std::string(to_string(dist)) + " N=" + std::to_string(c.n);

            std::vector<Key> data, buffer;
            for (const int t : {1, 4}) {
                data = original;
                parallel_radix_sort(data, buffer, t, ws);
                CHECK_MSG(verify_result(data, fp, ref).passed(), label + " parallel scatter T=" + std::to_string(t));
            }
            data = original;
            serial_radix_sort(data, buffer);
            CHECK_MSG(verify_result(data, fp, ref).passed(), label + " serial scatter");

            data = original;
            parallel_histogram_fill_sort(data, 4, ws);
            CHECK_MSG(verify_result(data, fp, ref).passed(), label + " parallel fill");
        }
    }
}

void test_generator() {
    const Distribution all[] = {Distribution::Uniform, Distribution::Random, Distribution::Sorted, Distribution::Reverse,
                                Distribution::FewValues, Distribution::Zeros, Distribution::Max, Distribution::TwoValues};
    for (const Distribution d : all) {
        const std::size_t n = 300000;
        std::vector<Key> a, b, chunked(n);
        omp_set_num_threads(1);
        generate_keys(a, n, d, 99);
        omp_set_num_threads(4);
        generate_keys(b, n, d, 99);
        CHECK_MSG(a == b, std::string("generator independent of thread count: ") + to_string(d));

        // Порционная генерация (как в --mode generate) даёт тот же массив.
        for (std::size_t first = 0; first < n; first += 70001) {
            const std::size_t cnt = std::min<std::size_t>(70001, n - first);
            generate_keys(chunked.data() + first, first, cnt, n, d, 99);
        }
        CHECK_MSG(a == chunked, std::string("chunked generation equals one-shot: ") + to_string(d));

        std::vector<Key> other;
        generate_keys(other, n, d, 100);  // другое зерно -> другие данные (для случайных распределений)
        if (d == Distribution::Uniform || d == Distribution::Random || d == Distribution::FewValues ||
            d == Distribution::TwoValues) {
            CHECK_MSG(a != other, std::string("different seed gives different data: ") + to_string(d));
        }
    }

    std::vector<Key> v;
    generate_keys(v, 1000000, Distribution::Sorted, 1);
    CHECK(std::is_sorted(v.begin(), v.end()));
    CHECK(v.front() == 0 && v.back() >= 65535 - 1);
    generate_keys(v, 1000000, Distribution::Reverse, 1);
    CHECK(std::is_sorted(v.rbegin(), v.rend()));
    CHECK(v.front() == 65535);
    generate_keys(v, 1000000, Distribution::Zeros, 1);
    CHECK(std::all_of(v.begin(), v.end(), [](Key x) { return x == 0; }));
    generate_keys(v, 1000000, Distribution::Max, 1);
    CHECK(std::all_of(v.begin(), v.end(), [](Key x) { return x == 65535; }));
    generate_keys(v, 1000000, Distribution::TwoValues, 1);
    CHECK(std::all_of(v.begin(), v.end(), [](Key x) { return x == 0 || x == 65535; }));
    generate_keys(v, 1000000, Distribution::FewValues, 1);
    {
        std::vector<Key> distinct = v;
        std::sort(distinct.begin(), distinct.end());
        distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
        CHECK(distinct.size() == 16);
        CHECK(distinct.front() == 0 && distinct.back() == 65535);
    }
    generate_keys(v, 1000000, Distribution::Uniform, 1);
    {
        const Fingerprint fp = make_fingerprint(v);
        std::size_t distinct = 0;
        for (const std::uint64_t c : fp.histogram) distinct += (c != 0);
        CHECK(distinct > 64000);  // равномерное: почти все значения встречаются
        const double mean = static_cast<double>(fp.sum) / 1e6;
        CHECK(mean > 32768.0 - 300.0 && mean < 32768.0 + 300.0);
    }
    generate_keys(v, 1000000, Distribution::Random, 1);
    {
        const Fingerprint fp = make_fingerprint(v);
        const double mean = static_cast<double>(fp.sum) / 1e6;
        CHECK(mean > 32768.0 - 300.0 && mean < 32768.0 + 300.0);
        const std::size_t middle = static_cast<std::size_t>(std::count_if(v.begin(), v.end(), [](Key x) { return x > 16384 && x < 49152; }));
        CHECK(middle > 800000);  // «колокол»: заметно больше половины значений в средней половине диапазона
    }

    Distribution d{};
    CHECK(parse_distribution("FEW_VALUES", d) && d == Distribution::FewValues);
    CHECK(parse_distribution("few", d) && d == Distribution::FewValues);
    CHECK(parse_distribution("Uniform", d) && d == Distribution::Uniform);
    CHECK(!parse_distribution("gaussian", d));
}

void test_io_roundtrip() {
    std::mt19937_64 rng(11);
    for (const FileFormat fmt : {FileFormat::Text, FileFormat::Binary}) {
        for (const std::size_t n : {0u, 1u, 2u, 1000u, 3000000u}) {  // 3·10^6 текстовых чисел — несколько блоков чтения
            const std::vector<Key> original = make_pattern(kRandom, n, rng);
            TempFile f(fmt == FileFormat::Text ? ".txt" : ".bin");
            const std::uint64_t written = write_keys(f.str(), fmt, original);
            CHECK(written == std::filesystem::file_size(f.path));
            std::vector<Key> back = {1, 2, 3};  // старое содержимое должно быть заменено
            const ReadInfo info = read_keys(f.str(), fmt, back);
            CHECK_MSG(back == original, std::string("roundtrip ") + to_string(fmt) + " N=" + std::to_string(n));
            CHECK(info.file_bytes == written);
        }
    }
    {  // Auto: расширение определяет формат
        CHECK(resolve_format(FileFormat::Auto, "x.BIN") == FileFormat::Binary);
        CHECK(resolve_format(FileFormat::Auto, "x.txt") == FileFormat::Text);
        CHECK(resolve_format(FileFormat::Auto, "noext") == FileFormat::Text);
        CHECK(resolve_format(FileFormat::Text, "x.bin") == FileFormat::Text);
    }
    {  // Потоковая запись порциями == запись целиком
        const std::vector<Key> original = make_pattern(kRandom, 100003, rng);
        TempFile f1(".txt"), f2(".txt");
        write_keys(f1.str(), FileFormat::Text, original);
        KeyWriter w(f2.str(), FileFormat::Text);
        for (std::size_t i = 0; i < original.size(); i += 777) {
            w.write(original.data() + i, std::min<std::size_t>(777, original.size() - i));
        }
        w.close();
        std::ifstream a(f1.path, std::ios::binary), b(f2.path, std::ios::binary);
        const std::string sa((std::istreambuf_iterator<char>(a)), std::istreambuf_iterator<char>());
        const std::string sb((std::istreambuf_iterator<char>(b)), std::istreambuf_iterator<char>());
        CHECK(sa == sb);
        CHECK(!sa.empty() && sa.back() == '\n');
    }
}

void test_text_parsing_formats() {
    TempFile f(".txt");
    std::vector<Key> v;
    auto parse = [&](const std::string& text) {
        write_raw(f, text);
        read_keys(f.str(), FileFormat::Text, v);
    };
    parse("3\n1\n2");  // нет перевода строки в конце
    CHECK((v == std::vector<Key>{3, 1, 2}));
    parse("3\r\n1\r\n\r\n 2 \t\r\n");  // CRLF, пустые строки, пробелы
    CHECK((v == std::vector<Key>{3, 1, 2}));
    parse("\xEF\xBB\xBF" "7\n0\n65535\n");  // BOM UTF-8, граничные значения
    CHECK((v == std::vector<Key>{7, 0, 65535}));
    parse("0000065535\n000\n");  // ведущие нули
    CHECK((v == std::vector<Key>{65535, 0}));
    parse("1 2\t3\n4");  // любые пробельные разделители
    CHECK((v == std::vector<Key>{1, 2, 3, 4}));
    parse("");
    CHECK(v.empty());
    parse("\n\n  \n");
    CHECK(v.empty());

    write_raw(f, "1\n2\nabc\n4\n");
    CHECK(read_fails_with(f, "line 3"));
    write_raw(f, "10\n65536\n");
    CHECK(read_fails_with(f, "line 2"));
    CHECK(read_fails_with(f, "out of range"));
    write_raw(f, "99999999999999999999\n");  // очень длинное число: не переполняет накопитель
    CHECK(read_fails_with(f, "out of range"));
    write_raw(f, "5\n-7\n");
    CHECK(read_fails_with(f, "negative"));
    write_raw(f, "1.5\n");
    CHECK(read_fails_with(f, "unexpected character"));
    write_raw(f, std::string("1\n2\n\0003\n", 7));  // нулевой байт внутри
    CHECK(read_fails_with(f, "unexpected byte"));

    CHECK_THROWS(read_keys("definitely_missing_file_xyz.txt", FileFormat::Text, v), IoError);
    CHECK_THROWS(read_keys(std::filesystem::temp_directory_path().string(), FileFormat::Text, v), IoError);  // каталог

    TempFile bin(".bin");
    write_raw(bin, "abc");  // 3 байта — не кратно 2
    CHECK_THROWS(read_keys(bin.str(), FileFormat::Binary, v), IoError);
    CHECK_THROWS(KeyWriter("/definitely/not/existing/dir/out.txt", FileFormat::Text), IoError);
}

// Число, разрезанное границей блока чтения (4 МиБ — внутренняя константа io.cpp), разбирается верно.
void test_chunk_boundary() {
    constexpr std::size_t kChunk = std::size_t{1} << 22;
    for (std::size_t offset = 0; offset <= 7; ++offset) {
        // Префикс из kChunk - 6 + offset байт: число «65535» начинается за 6 - offset байт до границы блока.
        const std::size_t prefix_len = kChunk - 6 + offset;
        std::string text;
        text.reserve(prefix_len + 32);
        std::size_t ones = 0;
        while (text.size() + 2 <= prefix_len) {
            text += "1\n";
            ++ones;
        }
        text.append(prefix_len - text.size(), '\n');  // добиваем пустыми строками до точной длины
        text += "65535\r\n7\n";
        TempFile f(".txt");
        write_raw(f, text);
        std::vector<Key> v;
        read_keys(f.str(), FileFormat::Text, v);
        CHECK_MSG(v.size() == ones + 2, "chunk boundary offset " + std::to_string(offset));
        CHECK_MSG(v.size() == ones + 2 && v[ones] == 65535 && v[ones + 1] == 7 && v[0] == 1,
                  "chunk boundary values offset " + std::to_string(offset));
    }
}

void test_verify_module() {
    std::mt19937_64 rng(3);
    const std::vector<Key> original = make_pattern(kRandom, 5000, rng);
    const Fingerprint fp = make_fingerprint(original);
    const std::vector<Key> sorted = make_reference_sorted(original);
    CHECK(verify_result(sorted, fp, &sorted).passed());
    CHECK(verify_result(sorted, fp, nullptr).passed());

    CHECK(!verify_result(original, fp, nullptr).sorted);  // не отсортирован
    std::vector<Key> corrupted = sorted;
    corrupted[100] = static_cast<Key>(corrupted[100] == 65535 ? 65534 : corrupted[100] + 1);  // подмена значения: мультимножество меняется
    {
        const Verification v = verify_result(corrupted, fp, nullptr);
        CHECK(!v.passed());
        CHECK(!v.histogram_ok);
    }
    std::vector<Key> shorter = sorted;
    shorter.pop_back();
    {
        const Verification v = verify_result(shorter, fp, nullptr);
        CHECK(!v.passed());
        CHECK(!v.count_ok);
    }
    {  // перестановка с тем же мультимножеством, но неотсортированная
        std::vector<Key> perm = sorted;
        std::reverse(perm.begin(), perm.end());
        const Verification v = verify_result(perm, fp, &sorted);
        CHECK(v.histogram_ok && v.checksum_ok && v.count_ok);
        CHECK(!v.sorted && !v.reference_ok && !v.passed());
    }
}

void test_memory_estimate_and_limits() {
    const MemoryEstimate m = estimate_memory(1000000000u, 16, Algorithm::Scatter, true);
    CHECK(m.input_bytes == 2000000000ull);
    CHECK(m.output_bytes == 2000000000ull);
    CHECK(m.histogram_bytes == 16ull * kKeyRange * sizeof(Count));
#ifndef RADIX_WIDE_COUNTERS
    CHECK(m.histogram_bytes == 4ull * 1024 * 1024);  // 16 потоков * 65536 * 4 байта = 4 МиБ
    CHECK(max_supported_elements() == 4294967295ull || sizeof(std::size_t) < 8);
#endif
    const MemoryEstimate f = estimate_memory(1000000000u, 16, Algorithm::Fill, true);
    CHECK(f.output_bytes == 0);
    const MemoryEstimate s = estimate_memory(0, 8, Algorithm::Scatter, true);
    CHECK(s.input_bytes == 0 && s.histogram_bytes == sizeof(Histogram));  // N = 0: потоков не больше одного

    ParallelRadixWorkspace ws(4);
    CHECK(ws.capacity() == 4);
    CHECK(ws.bytes() >= 4ull * kKeyRange * sizeof(Count));
    ws.reserve(2);  // не сжимается
    CHECK(ws.capacity() == 4);
    ws.reserve(8);
    CHECK(ws.capacity() == 8);
}


// Гарантия, на которой держится экономия памяти бенчмарка: scatter НЕ портит вход —
// после вызова исходные данные целиком лежат в buffer (восстановление = обмен data/buffer).
void test_scatter_leaves_input_intact() {
    std::mt19937_64 rng(123);
    ParallelRadixWorkspace ws;
    for (const std::size_t n : {0u, 1u, 2u, 100u, 70000u, 300000u}) {
        for (const int p : {kRandom, kSorted, kReverse, kFewValues, kZeros}) {
            const std::vector<Key> input = make_pattern(p, n, rng);
            std::vector<Key> expected = input;
            std::sort(expected.begin(), expected.end());
            const std::string label = std::string(pattern_name(p)) + " N=" + std::to_string(n);
            {
                std::vector<Key> d = input, b;
                serial_radix_sort(d, b);
                CHECK_MSG(d == expected, label + " serial sorted");
                CHECK_MSG(b == input, label + " serial: buffer holds the unchanged input");
            }
            for (const int t : {1, 3, 8}) {
                std::vector<Key> d = input, b;
                parallel_radix_sort(d, b, t, ws);
                CHECK_MSG(d == expected, label + " parallel sorted T=" + std::to_string(t));
                CHECK_MSG(b == input, label + " parallel: buffer holds the unchanged input T=" + std::to_string(t));
            }
        }
    }
}

void test_default_run_settings_and_counts() {
    CHECK(default_run_settings(0).repeats == 3 && default_run_settings(0).warmup);
    CHECK(default_run_settings(kLargeInputThreshold - 1).repeats == 3);
    CHECK(default_run_settings(kLargeInputThreshold - 1).warmup);
    const RunSettings big = default_run_settings(kLargeInputThreshold);
    CHECK(big.repeats == 1 && !big.warmup);  // миллиард элементов: один замер без прогрева
    CHECK(default_run_settings(1000000000u).repeats == 1);

    RunConfig c;
    c.run_serial = c.run_parallel = true;
    c.repeats = 3;
    c.warmup = true;
    CHECK(count_sorts(c) == 8);
    c.run_parallel = false;
    CHECK(count_sorts(c) == 4);
    c.repeats = 1;
    c.warmup = false;
    CHECK(count_sorts(c) == 1);
}

void test_experiment_memory_estimate() {
    const std::size_t n = 1000000000u;
    const std::uint64_t arr = 2000000000ull;  // один массив из 10^9 ключей
    const std::uint64_t slack = 64ull << 20;  // таблицы гистограмм и т. п. — единицы МиБ
    RunConfig c;
    c.threads = 16;
    c.algorithm = Algorithm::Scatter;
    c.repeats = 3;
    c.warmup = true;
    std::uint64_t m = estimate_experiment_memory(n, c);
    CHECK(m >= 2 * arr && m < 2 * arr + slack);  // scatter: два массива, без копии оригинала
    c.repeats = 1;
    c.warmup = false;
    m = estimate_experiment_memory(n, c);
    CHECK(m >= 2 * arr && m < 2 * arr + slack);

    c.algorithm = Algorithm::Fill;
    c.run_serial = false;  // одна параллельная сортировка на месте -> ОДИН массив
    m = estimate_experiment_memory(n, c);
    CHECK(m >= arr && m < arr + slack);
    c.run_serial = true;  // две сортировки -> нужна эталонная копия -> два массива
    m = estimate_experiment_memory(n, c);
    CHECK(m >= 2 * arr && m < 2 * arr + slack);

    c.verify = true;  // проверка добавляет лишь отпечатки (1 МиБ); эталона std::sort для N > 10^7 нет
    CHECK(estimate_experiment_memory(n, c) < 2 * arr + slack);
    CHECK(estimate_experiment_memory(0, c) < slack);
}

void test_run_experiment_fairness_and_memory() {
    std::mt19937_64 rng(2025);
    for (const Algorithm algo : {Algorithm::Scatter, Algorithm::Fill}) {
        for (const int mode : {0, 1, 2}) {  // 0 — только serial, 1 — только parallel, 2 — оба
            for (const std::size_t n : {0u, 1u, 7u, 100000u}) {
                const std::vector<Key> input = make_pattern(kRandom, n, rng);
                std::vector<Key> expected = input;
                std::sort(expected.begin(), expected.end());

                RunConfig cfg;
                cfg.algorithm = algo;
                cfg.run_serial = (mode != 1);
                cfg.run_parallel = (mode != 0);
                cfg.threads = 3;
                cfg.repeats = 3;
                cfg.warmup = true;
                cfg.verify = true;
                int inspected = 0;
                int progress_lines = 0;
                bool all_pristine = true;
                // Каждый запуск (прогрев, serial, parallel, все повторы) обязан начинаться с ИСХОДНЫХ данных.
                cfg.inspect_input = [&](const std::vector<Key>& a) {
                    ++inspected;
                    if (a != input) all_pristine = false;
                };
                cfg.progress = [&](const std::string&) { ++progress_lines; };

                std::vector<Key> sorted;
                const RunResult r = run_experiment(std::vector<Key>(input), cfg, &sorted);

                const std::string label = std::string(to_string(algo)) + " mode=" + std::to_string(mode) +
                                          " N=" + std::to_string(n);
                CHECK_MSG(all_pristine, label + ": every run starts from the original data");
                CHECK_MSG(inspected == count_sorts(cfg), label + ": number of sort calls");
                CHECK_MSG(sorted == expected, label + ": sorted output");
                CHECK_MSG(progress_lines > 0, label + ": progress lines are reported");
                if (cfg.run_serial) {
                    CHECK_MSG(r.serial_check.passed(), label + ": serial verified");
                    CHECK_MSG(r.serial.samples.size() == 3, label + ": serial samples");
                }
                if (cfg.run_parallel) {
                    CHECK_MSG(r.parallel_check.passed(), label + ": parallel verified");
                    CHECK_MSG(r.parallel.samples.size() == 3, label + ": parallel samples");
                }
                // Лишняя копия данных нужна только fill при нескольких запусках; scatter восстанавливает вход обменом.
                const std::uint64_t expected_restore = (algo == Algorithm::Fill && count_sorts(cfg) > 1) ? n * sizeof(Key) : 0;
                CHECK_MSG(r.restore_bytes == expected_restore, label + ": restore copy size");
            }
        }
    }
    {  // Единственный запуск: ничего восстанавливать и копировать не нужно
        RunConfig cfg;
        cfg.algorithm = Algorithm::Fill;
        cfg.run_serial = false;
        cfg.run_parallel = true;
        cfg.threads = 2;
        cfg.repeats = 1;
        cfg.warmup = false;
        cfg.verify = true;
        const std::vector<Key> input = make_pattern(kRandom, 5000, rng);
        const RunResult r = run_experiment(std::vector<Key>(input), cfg);
        CHECK(r.restore_bytes == 0 && r.parallel_check.passed());
    }
    {  // Недопустимые настройки
        RunConfig cfg;
        cfg.run_serial = cfg.run_parallel = false;
        CHECK_THROWS(run_experiment(std::vector<Key>{1, 2, 3}, cfg), std::invalid_argument);
        cfg.run_serial = true;
        cfg.repeats = 0;
        CHECK_THROWS(run_experiment(std::vector<Key>{1, 2, 3}, cfg), std::invalid_argument);
        cfg.repeats = 1;
        cfg.threads = 0;
        CHECK_THROWS(run_experiment(std::vector<Key>{1, 2, 3}, cfg), std::invalid_argument);
    }
}

void test_query_memory() {
    const MemoryInfo m = query_memory();
#if defined(__linux__) || defined(_WIN32)
    CHECK(m.known());
    CHECK(m.total >= m.available);
    CHECK(m.total > (64ull << 20));
#else
    (void)m;  // на прочих системах сведений о памяти может не быть
#endif
}

void test_estimate_key_count_and_reserve() {
    const std::size_t n = 3000000;
    for (const Distribution d : {Distribution::Uniform, Distribution::Sorted, Distribution::Reverse, Distribution::FewValues}) {
        std::vector<Key> v;
        generate_keys(v, n, d, 7);
        TempFile f(".txt");
        write_keys(f.str(), FileFormat::Text, v);
        const std::uint64_t est = estimate_key_count(f.str(), FileFormat::Text);
        const std::string label = std::string("estimate for ") + to_string(d) + " = " + std::to_string(est);
        CHECK_MSG(est >= n && est <= n * 14 / 10, label);  // не меньше N (иначе вектор «растёт») и без большого перерасхода

        std::vector<Key> back;
        read_keys(f.str(), FileFormat::Text, back);
        CHECK_MSG(back == v, label + ": content");
        CHECK_MSG(back.capacity() <= n * 14 / 10 + 2048, label + ": capacity is not excessive");
    }
    {  // binary: точно; небольшой текст: безопасная верхняя граница; несуществующий файл: 0
        std::vector<Key> v;
        generate_keys(v, 12345, Distribution::Uniform, 3);
        TempFile bin(".bin"), txt(".txt");
        write_keys(bin.str(), FileFormat::Binary, v);
        write_keys(txt.str(), FileFormat::Text, v);
        CHECK(estimate_key_count(bin.str(), FileFormat::Binary) == v.size());
        CHECK(estimate_key_count(txt.str(), FileFormat::Text) >= v.size());
        CHECK(estimate_key_count("definitely_missing_file_xyz.txt", FileFormat::Text) == 0);
    }
}

// Бинарное чтение идёт порциями по 2^25 элементов: проверяем границу между порциями.
void test_binary_read_piece_boundary() {
    const std::size_t n = (std::size_t{1} << 25) + 3;
    std::vector<Key> v;
    generate_keys(v, n, Distribution::Uniform, 99);
    TempFile f(".bin");
    write_keys(f.str(), FileFormat::Binary, v);
    std::vector<Key> back;
    std::uint64_t progress_calls = 0;
    read_keys(f.str(), FileFormat::Binary, back, [&](std::uint64_t, std::uint64_t) { ++progress_calls; });
    CHECK(back == v);
    (void)progress_calls;  // обратный вызов троттлится по времени (раз в 2 с): число вызовов не фиксировано
}


// Разбор вывода `vm_stat` (macOS): проверяется на любой ОС, потому что это чистый разбор строки.
void test_vm_stat_parser() {
    const std::string sample =
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                               5324.\n"
        "Pages active:                           212345.\n"
        "Pages inactive:                         198765.\n"
        "Pages speculative:                        1234.\n"
        "Pages throttled:                             0.\n"
        "Pages wired down:                       100000.\n"
        "Pages purgeable:                          2000.\n"
        "\"Translation faults\":                 12345678.\n"
        "Pages copy-on-write:                    123456.\n";
    // free + inactive + speculative + purgeable = 5324 + 198765 + 1234 + 2000 страниц по 16384 байта
    CHECK(detail::parse_vm_stat_available(sample) == (5324ull + 198765ull + 1234ull + 2000ull) * 16384ull);

    const std::string small_pages =
        "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
        "Pages free:      100.\n"
        "Pages inactive:  200.\n";
    CHECK(detail::parse_vm_stat_available(small_pages) == 300ull * 4096ull);  // speculative / purgeable необязательны

    CHECK(detail::parse_vm_stat_available("") == 0);                                   // пустой вывод
    CHECK(detail::parse_vm_stat_available("Pages free: 5.\nPages inactive: 5.\n") == 0);  // нет размера страницы
    CHECK(detail::parse_vm_stat_available(
              "Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 5.\n") == 0);  // нет inactive
    CHECK(detail::parse_vm_stat_available("command not found\n") == 0);
}

struct TestEntry {
    const char* name;
    std::function<void()> fn;
};

}  // namespace

int main(int argc, char** argv) {
    bool large = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--large") == 0) large = true;
        else {
            std::fprintf(stderr, "usage: radix_tests [--large]\n");
            return 2;
        }
    }

    const std::vector<TestEntry> tests = {
        {"block_begin", test_block_begin},
        {"memory_estimate_and_limits", test_memory_estimate_and_limits},
        {"edge_sizes_and_patterns", test_edge_sizes_and_patterns},
        {"threads_used_and_n_less_than_t", test_threads_used_and_n_less_than_t},
        {"write_range_invariants", test_write_range_invariants},
        {"workspace_and_buffer_reuse", test_workspace_and_buffer_reuse},
        {"invalid_arguments", test_invalid_arguments},
        {"generator", test_generator},
        {"io_roundtrip", test_io_roundtrip},
        {"text_parsing_formats", test_text_parsing_formats},
        {"chunk_boundary", test_chunk_boundary},
        {"verify_module", test_verify_module},
        {"scatter_leaves_input_intact", test_scatter_leaves_input_intact},
        {"default_run_settings_and_counts", test_default_run_settings_and_counts},
        {"experiment_memory_estimate", test_experiment_memory_estimate},
        {"run_experiment_fairness_and_memory", test_run_experiment_fairness_and_memory},
        {"query_memory", test_query_memory},
        {"vm_stat_parser", test_vm_stat_parser},
        {"estimate_key_count_and_reserve", test_estimate_key_count_and_reserve},
        {"binary_read_piece_boundary", test_binary_read_piece_boundary},
        {"sizes", [large] { test_sizes(large); }},
    };

    std::printf("radix_tests: %zu test groups%s, OpenMP max threads = %d\n", tests.size(),
                large ? " (+ large N = 10^8)" : "", omp_get_max_threads());
    int failed_groups = 0;
    for (const TestEntry& t : tests) {
        const int before = g_failures;
        const auto t0 = std::chrono::steady_clock::now();
        try {
            t.fn();
        } catch (const std::exception& e) {
            ++g_failures;
            std::fprintf(stderr, "  FAIL: unexpected exception in %s: %s\n", t.name, e.what());
        }
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const bool ok = (g_failures == before);
        if (!ok) ++failed_groups;
        std::printf("[%s] %-34s %.2f s\n", ok ? " OK " : "FAIL", t.name, sec);
    }
    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
