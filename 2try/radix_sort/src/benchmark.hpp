// benchmark.hpp — честный эксперимент «последовательная vs параллельная сортировка» и печать отчёта.
//
// Принципы честности измерений:
//   * обе реализации получают ОДИНАКОВЫЕ исходные данные (перед каждым запуском массив приводится к
//     исходному состоянию — см. ниже, как это делается без лишней памяти);
//   * восстановление данных, генерация, чтение/запись файлов и проверка корректности НЕ входят во
//     время сортировки: замеряется только вызов функции сортировки (std::chrono::steady_clock);
//   * все буферы и рабочая область выделяются заранее, один раз, до цикла замеров;
//   * перед замерами — один неучитываемый прогрев (запуск пула потоков OpenMP, подкачка страниц);
//   * serial и parallel чередуются в каждом повторе.
//
// Память (важно для N ~ 10^9). Эксперимент держит ровно столько массивов, сколько нужно:
//   * scatter: 2 массива (вход и выходной буфер). Сортировка подсчётом читает вход и пишет в буфер,
//     НЕ изменяя вход; после неё data и buffer меняются местами, так что исходные данные целиком
//     остаются в buffer. «Восстановление» перед следующим запуском — это O(1)-обмен указателями,
//     без копирования и без отдельной копии «оригинала»;
//   * fill: 1 массив, если сортировка выполняется один раз (ничего восстанавливать не нужно);
//     2 массива, если запусков несколько (вторая копия нужна, чтобы вернуть вход после сортировки на месте).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

#include "radix_sort.hpp"
#include "verify.hpp"

namespace radix {

// Начиная с этого числа элементов повторы и прогрев по умолчанию отключаются: одна сортировка такого
// массива — это уже секунды или десятки секунд, а 4–8 запусков подряд выглядят как «зависание».
inline constexpr std::size_t kLargeInputThreshold = 250'000'000;

struct RunSettings {
    int repeats = 3;
    bool warmup = true;
};

// Значения по умолчанию в зависимости от размера: N < 250 000 000 -> 3 повтора + прогрев; иначе 1 повтор без прогрева.
// Явные --repeats / --warmup / --no-warmup всегда имеют приоритет.
RunSettings default_run_settings(std::size_t n) noexcept;

struct RunConfig {
    Algorithm algorithm = Algorithm::Scatter;
    bool run_serial = true;
    bool run_parallel = true;
    int threads = 1;      // запрошенное число потоков параллельной версии
    int repeats = 3;      // число измеряемых повторов
    bool warmup = true;   // один неучитываемый прогрев каждой версии
    bool verify = false;  // проверить результат последнего повтора (вне измерения)

    // Необязательный приёмник статусных строк («warm-up…», «run 1/3 done in 5.2 s», …).
    // Вызывается из потока, запустившего эксперимент, только между замерами и никогда внутри них.
    std::function<void(const std::string&)> progress;

    // Диагностический хук (для тестов): вызывается ПЕРЕД каждой сортировкой, включая прогрев, с массивом,
    // который сейчас будет сортироваться, — вне измеряемого интервала. Позволяет убедиться, что каждый
    // запуск действительно начинается с исходных данных (иначе бенчмарк сортировал бы уже отсортированное).
    std::function<void(const std::vector<Key>&)> inspect_input;
};

// Сколько всего вызовов сортировки выполнит эксперимент (с прогревом).
int count_sorts(const RunConfig& config) noexcept;

// Пиковая потребность в памяти эксперимента, байт: массивы + таблицы + данные проверки.
// Используется для предварительной проверки «хватит ли ОЗУ» до выделения памяти.
std::uint64_t estimate_experiment_memory(std::size_t n, const RunConfig& config);

struct SeriesStats {
    std::vector<double> samples;  // секунды, по повторам
    double best = 0.0;
    double mean = 0.0;
    double stddev = 0.0;  // выборочное стандартное отклонение (0, если повтор один)
    bool empty() const noexcept { return samples.empty(); }
};

struct RunResult {
    RunConfig config;
    std::size_t n = 0;
    SeriesStats serial;
    SeriesStats parallel;
    PhaseTimes serial_phases;    // фазы самого быстрого повтора
    PhaseTimes parallel_phases;  // (threads_used — фактический размер команды OpenMP)
    Verification serial_check;
    Verification parallel_check;
    MemoryEstimate serial_memory;
    MemoryEstimate parallel_memory;
    std::uint64_t restore_bytes = 0;  // дополнительная копия данных только для восстановления входа (fill, несколько запусков)
};

// Выполняет эксперимент. Массив `input` ПОГЛОЩАЕТСЯ (перемещается внутрь): держать вторую копию
// исходных данных незачем. Если sorted_out != nullptr, туда перемещается отсортированный массив
// последнего запуска.
RunResult run_experiment(std::vector<Key>&& input, const RunConfig& config,
                         std::vector<Key>* sorted_out = nullptr);

struct ReportContext {
    std::string title = "Parallel Radix Sort";
    std::string input_description;   // откуда данные: «uniform, seed 12345» / «file 'x.txt' (text)»
    std::string load_label;          // «Generation» / «Reading» (пусто — не печатать)
    double load_seconds = -1.0;
    std::string output_description;  // куда записан результат (пусто — не записывался)
    double write_seconds = -1.0;
    std::string repeats_note;        // пояснение к числу повторов (например, «default for large N»)
    int hardware_threads = 1;
};

void print_report(std::ostream& os, const RunResult& result, const ReportContext& ctx);

std::string format_bytes(std::uint64_t bytes);  // «1.86 GiB»
std::string format_seconds(double seconds);     // «0.083 s»

}  // namespace radix
