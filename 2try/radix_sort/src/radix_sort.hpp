// radix_sort.hpp — публичный интерфейс цифровой сортировки массива uint16_t.
//
// Реализованы два семейства алгоритмов (подробное описание — в radix_sort.cpp):
//
//   * scatter (основной, по ТЗ): стабильная сортировка подсчётом с выходным буфером.
//       serial_radix_sort   — последовательный эталон;
//       parallel_radix_sort — OpenMP: локальные гистограммы + глобальные префиксные суммы
//                             + непересекающиеся диапазоны записи, без atomic/mutex.
//
//   * fill (расширение): «гистограмма + заливка», для ключей без полезной нагрузки.
//       serial_histogram_fill_sort / parallel_histogram_fill_sort — на месте, без буфера.
//
// Все функции сортируют по возрастанию; результат всегда оказывается в `data`.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace radix {

// ----------------------------------------------------------------------------
// Базовые типы и константы
// ----------------------------------------------------------------------------

// Ключ: 16-битное беззнаковое целое (на всех распространённых платформах = unsigned short).
using Key = std::uint16_t;

inline constexpr std::size_t kKeyBits = 16;
inline constexpr std::size_t kKeyRange = std::size_t{1} << kKeyBits;  // K = 65536 значений: 0..65535
inline constexpr std::size_t kCacheLine = 64;                         // байт
inline constexpr int kMaxThreads = 1024;                              // разумная верхняя граница T

// Тип счётчиков и позиций в выходном массиве.
// 32 бит хватает для N <= 4 294 967 295 (с запасом покрывает целевые 10^9 элементов):
// максимальное значение любого счётчика / префиксной суммы / позиции равно N.
// Для N > 2^32-1 соберите проект с -DRADIX_WIDE_COUNTERS=ON (счётчики станут 64-битными).
#ifdef RADIX_WIDE_COUNTERS
using Count = std::uint64_t;
#else
using Count = std::uint32_t;
#endif

// Максимальное число элементов, которое поддерживает текущая сборка.
constexpr std::size_t max_supported_elements() noexcept {
    constexpr std::uintmax_t by_counter = std::numeric_limits<Count>::max();
    constexpr std::uintmax_t by_size_t = std::numeric_limits<std::size_t>::max();
    return static_cast<std::size_t>(by_counter < by_size_t ? by_counter : by_size_t);
}

// Начало i-го блока при делении n элементов на `parts` непрерывных блоков.
// Размеры блоков отличаются не более чем на 1 (первые n % parts блоков на один элемент длиннее),
// блок i = [block_begin(i), block_begin(i + 1)), причём block_begin(parts) == n.
// Формула написана так, чтобы i * (n / parts) не переполнялось даже на 32-битных платформах
// (в отличие от наивного i * n / parts).
constexpr std::size_t block_begin(std::size_t i, std::size_t parts, std::size_t n) noexcept {
    const std::size_t base = n / parts;
    const std::size_t rem = n % parts;
    return i * base + (i < rem ? i : rem);
}

enum class Algorithm {
    Scatter,  // стабильная сортировка подсчётом, нужен выходной буфер (основной вариант по ТЗ)
    Fill      // гистограмма + заливка диапазонов; на месте, без буфера (расширение)
};

const char* to_string(Algorithm a) noexcept;

// ----------------------------------------------------------------------------
// Рабочая область параллельной сортировки
// ----------------------------------------------------------------------------

// Локальная гистограмма одного потока: K счётчиков. Выровнена по кэш-линии, поэтому
// гистограммы разных потоков никогда не делят одну линию (нет false sharing).
// Конструктор намеренно ПУСТОЙ: память не обнуляется при выделении. Обнуляет таблицу
// владеющий ею поток (first-touch => на NUMA страницы окажутся в «его» узле памяти).
struct alignas(kCacheLine) Histogram {
    Histogram() noexcept {}
    Count bins[kKeyRange];
};

// Таблица глобальных стартовых позиций: K + 1 элементов (последний — граница N).
struct alignas(kCacheLine) StartTable {
    StartTable() noexcept {}
    Count v[kKeyRange + kCacheLine / sizeof(Count)];
};

// Счётчик «сумма по диапазону значений потока», отдельная кэш-линия на поток.
struct alignas(kCacheLine) PaddedCount {
    Count value;
};

// Рабочая область: создаётся ОДИН раз и переиспользуется между вызовами сортировки,
// поэтому в измеряемом цикле бенчмарка не происходит ни одной аллокации.
// Размер: T * K * sizeof(Count) (например, 16 потоков * 65536 * 4 Б = 4 МиБ) + O(K).
// Объект не потокобезопасен: один workspace — один одновременный вызов сортировки.
class ParallelRadixWorkspace {
public:
    ParallelRadixWorkspace() = default;
    explicit ParallelRadixWorkspace(int max_threads) { reserve(max_threads); }

    // Гарантирует место под max_threads гистограмм. Только растёт, никогда не сжимается.
    // Содержимое при росте не сохраняется (между вызовами сортировки оно и не нужно).
    void reserve(int max_threads);

    int capacity() const noexcept { return capacity_; }

    // Сколько байт удерживает рабочая область.
    std::uint64_t bytes() const noexcept;

    // --- Доступ к внутренним таблицам (для алгоритма, тестов и диагностики) ---

    Histogram* histograms() noexcept { return hist_.get(); }
    const Histogram* histograms() const noexcept { return hist_.get(); }
    Count* starts() noexcept { return starts_ ? starts_->v : nullptr; }
    PaddedCount* totals() noexcept { return totals_.get(); }

    // Строка t матрицы локальных гистограмм (K элементов). После завершения
    // parallel_radix_sort(...) в ней лежит КОНЕЦ диапазона записи потока t для каждого значения
    // (= начало диапазона потока t + 1) — это используется в тесте инвариантов.
    const Count* row(int t) const noexcept { return hist_[static_cast<std::size_t>(t)].bins; }

private:
    std::unique_ptr<Histogram[]> hist_;
    std::unique_ptr<StartTable> starts_;
    std::unique_ptr<PaddedCount[]> totals_;
    int capacity_ = 0;
};

// ----------------------------------------------------------------------------
// Диагностика: времена фаз и оценка памяти
// ----------------------------------------------------------------------------

// Времена фаз (секунды, omp_get_wtime). Для scatter: подсчёт / слияние+префиксы / раскладка.
// Для fill: подсчёт / слияние+префиксы / заливка.
struct PhaseTimes {
    double count_s = 0.0;
    double prefix_s = 0.0;
    double move_s = 0.0;
    int threads_used = 1;  // фактический размер команды OpenMP (может быть меньше запрошенного)

    double total() const noexcept { return count_s + prefix_s + move_s; }
};

struct MemoryEstimate {
    std::uint64_t input_bytes = 0;      // N * sizeof(Key)
    std::uint64_t output_bytes = 0;     // N * sizeof(Key) (0 для fill — он работает на месте)
    std::uint64_t histogram_bytes = 0;  // T * K * sizeof(Count) (для serial — одна таблица)
    std::uint64_t aux_bytes = 0;        // таблица стартов, суммы по диапазонам
    std::uint64_t total() const noexcept {
        return input_bytes + output_bytes + histogram_bytes + aux_bytes;
    }
};

// Оценка расхода памяти алгоритма (без учёта памяти бенчмарка: копии исходных данных и т. п.).
MemoryEstimate estimate_memory(std::size_t n, int threads, Algorithm algo, bool parallel);

// ----------------------------------------------------------------------------
// Алгоритм scatter (основной): стабильная сортировка подсчётом
// ----------------------------------------------------------------------------

// Последовательная версия (эталон).
// data   — вход; по возвращении содержит отсортированный результат;
// buffer — выходной буфер; при необходимости изменяется до размера data.size().
//          Результат записывается в buffer, после чего data и buffer меняются местами
//          (O(1)), поэтому по возвращении buffer содержит старый, несортированный вход.
// Внимание: из-за swap указатели/итераторы на data.data() и buffer.data() инвалидируются
// в том смысле, что владельцы буферов меняются.
// ГАРАНТИЯ (на неё опирается бенчмарк и тест): исходные данные НЕ изменяются — по возвращении
// buffer содержит точную копию входа в прежнем порядке. Поэтому «вернуть вход» перед повторным
// запуском можно обменом data.swap(buffer), без копирования и без третьего массива.
// (Это не относится к fill: он сортирует на месте и разрушает вход.)
// Исключения: std::length_error (N не помещается в Count), std::invalid_argument
// (data и buffer — один и тот же объект).
void serial_radix_sort(std::vector<Key>& data, std::vector<Key>& buffer, PhaseTimes* times = nullptr);

// Параллельная версия (OpenMP). Контракт тот же, что у serial_radix_sort.
// num_threads — желаемое число потоков (1..kMaxThreads). Фактически используется
// min(num_threads, N) потоков (и не больше, чем выдаст OpenMP-рантайм); для N = 0 функция
// ничего не делает. num_threads = 1 корректно вырождается в один блок.
// Удобная перегрузка: создаёт временную рабочую область на один вызов.
void parallel_radix_sort(std::vector<Key>& data, std::vector<Key>& buffer, int num_threads);

// Основная перегрузка: рабочая область создаётся и переиспользуется вызывающим кодом.
void parallel_radix_sort(std::vector<Key>& data, std::vector<Key>& buffer, int num_threads,
                         ParallelRadixWorkspace& workspace, PhaseTimes* times = nullptr);

// ----------------------------------------------------------------------------
// Алгоритм fill (расширение): гистограмма + заливка, на месте
// ----------------------------------------------------------------------------

// Для «голых» ключей без полезной нагрузки результат полностью определяется гистограммой:
// отсортированный массив = count[0] нулей, затем count[1] единиц и т. д. Раскладывать
// исходные элементы по позициям (scatter) не нужно — достаточно залить диапазоны.
// Нет случайных записей и не нужен второй массив размера N.
void serial_histogram_fill_sort(std::vector<Key>& data, PhaseTimes* times = nullptr);
void parallel_histogram_fill_sort(std::vector<Key>& data, int num_threads,
                                  ParallelRadixWorkspace& workspace, PhaseTimes* times = nullptr);

}  // namespace radix
