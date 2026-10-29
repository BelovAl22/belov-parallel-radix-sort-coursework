// gbench_radix_sort.cpp — НЕОБЯЗАТЕЛЬНЫЙ бенчмарк на Google Benchmark.
//
// Собирается только при -DENABLE_GOOGLE_BENCHMARK=ON и наличии установленной библиотеки
// (find_package(benchmark)). Основное приложение sort_app и тесты от неё не зависят.
//
// Запуск:   ./radix_gbench
//           ./radix_gbench --benchmark_filter=Parallel --benchmark_min_time=0.5s
//
// Во всех бенчмарках восстановление исходных данных выполняется между PauseTiming()/ResumeTiming(),
// то есть не входит во время сортировки; буферы и рабочая область создаются один раз до цикла.
// UseRealTime() обязателен: параллельные версии нужно сравнивать по настенному, а не по CPU-времени.
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "data_gen.hpp"
#include "radix_sort.hpp"

using namespace radix;

namespace {

// Исходные данные генерируются один раз для каждого размера и затем только копируются.
const std::vector<Key>& original_data(std::size_t n) {
    static std::size_t cached_n = static_cast<std::size_t>(-1);
    static std::vector<Key> cached;
    if (cached_n != n) {
        generate_keys(cached, n, Distribution::Uniform, 12345);
        cached_n = n;
    }
    return cached;
}

void set_throughput(benchmark::State& state, std::size_t n) {
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(n));
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(n * sizeof(Key)));
}

void BM_SerialRadix(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::vector<Key>& original = original_data(n);
    std::vector<Key> data(n), buffer(n);
    for (auto _ : state) {
        state.PauseTiming();
        data = original;
        state.ResumeTiming();
        serial_radix_sort(data, buffer);
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
    set_throughput(state, n);
}

void BM_ParallelRadix(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const int threads = static_cast<int>(state.range(1));
    const std::vector<Key>& original = original_data(n);
    std::vector<Key> data(n), buffer(n);
    ParallelRadixWorkspace workspace(threads);  // создаётся один раз: аллокаций в цикле нет
    for (auto _ : state) {
        state.PauseTiming();
        data = original;
        state.ResumeTiming();
        parallel_radix_sort(data, buffer, threads, workspace);
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
    state.counters["threads"] = threads;
    set_throughput(state, n);
}

void BM_SerialFill(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::vector<Key>& original = original_data(n);
    std::vector<Key> data(n);
    for (auto _ : state) {
        state.PauseTiming();
        data = original;
        state.ResumeTiming();
        serial_histogram_fill_sort(data);
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
    set_throughput(state, n);
}

void BM_ParallelFill(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const int threads = static_cast<int>(state.range(1));
    const std::vector<Key>& original = original_data(n);
    std::vector<Key> data(n);
    ParallelRadixWorkspace workspace(threads);
    for (auto _ : state) {
        state.PauseTiming();
        data = original;
        state.ResumeTiming();
        parallel_histogram_fill_sort(data, threads, workspace);
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
    state.counters["threads"] = threads;
    set_throughput(state, n);
}

// Справочная линия: std::sort на тех же данных (только для небольших размеров, он медленный).
void BM_StdSort(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::vector<Key>& original = original_data(n);
    std::vector<Key> data(n);
    for (auto _ : state) {
        state.PauseTiming();
        data = original;
        state.ResumeTiming();
        std::sort(data.begin(), data.end());
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
    set_throughput(state, n);
}

}  // namespace

BENCHMARK(BM_SerialRadix)->RangeMultiplier(4)->Range(1 << 16, 1 << 26)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SerialFill)->RangeMultiplier(4)->Range(1 << 16, 1 << 26)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_StdSort)->RangeMultiplier(4)->Range(1 << 16, 1 << 24)->UseRealTime()->Unit(benchmark::kMillisecond);

// Аргументы: {число элементов, число потоков}.
BENCHMARK(BM_ParallelRadix)
    ->ArgsProduct({{1 << 20, 1 << 24, 1 << 26}, {1, 2, 4, 8, 16}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ParallelFill)
    ->ArgsProduct({{1 << 20, 1 << 24, 1 << 26}, {1, 2, 4, 8, 16}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
