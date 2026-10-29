// radix_sort.cpp — реализация цифровой сортировки uint16_t (последовательной и на OpenMP).
//
// ============================================================================
//  ОБЩАЯ ИДЕЯ
// ============================================================================
//  Ключ занимает ровно 16 бит, поэтому в нём одна «цифра» с основанием K = 2^16 = 65536.
//  Цифровая сортировка по такому ключу — это ОДИН проход, т. е. сортировка подсчётом
//  (counting sort). Время O(N + K), память: два массива по N элементов + O(K).
//
// ============================================================================
//  ПОСЛЕДОВАТЕЛЬНЫЙ АЛГОРИТМ (serial_radix_sort)
// ============================================================================
//    1. count[v] — сколько раз значение v встретилось во входе.
//    2. start[v] — exclusive prefix sum: число элементов, строго меньших v.
//    3. Стабильная раскладка во второй буфер проходом С КОНЦА входа.
//  Вместо двух таблиц (count и start) используется одна таблица bound[0..K]
//  (подробности — у самой функции): так в L2-кэше нужна лишь одна таблица на 256 КиБ.
//
// ============================================================================
//  ПАРАЛЛЕЛЬНЫЙ АЛГОРИТМ (parallel_radix_sort)
// ============================================================================
//  Вход режется на T непрерывных блоков с одинаковым (±1) ЧИСЛОМ ЭЛЕМЕНТОВ.
//  Поток t владеет блоком t входа и своей локальной гистограммой local[t][0..K).
//
//    Фаза 1  Каждый поток считает гистограмму СВОЕГО блока в СВОЕЙ таблице.
//            ------------------------------ барьер ------------------------------
//    Фаза 2  global[v] = Σ_t local[t][v];  start[v] = Σ_{u<v} global[u];
//            для каждой пары (t, v):
//              позиция_записи[t][v] = start[v] + Σ_{t'<t} local[t'][v].
//            Эта работа (она O(T·K)) тоже делится между потоками — по диапазонам ЗНАЧЕНИЙ.
//            ------------------------------ барьер ------------------------------
//    Фаза 3  Каждый поток проходит СВОЙ блок слева направо и пишет элемент x в
//            output[позиция_записи[t][x]++]. Диапазоны записи разных потоков не
//            пересекаются, поэтому атомики/мьютексы не нужны.
//
//  Почему локальные гистограммы, а не один общий счётчик.
//    Единая таблица count[65536] требовала бы атомарного инкремента на КАЖДЫЙ элемент
//    (или критической секции). Это сериализует потоки, а 16 соседних счётчиков живут в
//    одной кэш-линии, которая начинает «летать» между ядрами (true/false sharing).
//    Локальные гистограммы устраняют эту проблему: поток пишет только в свою таблицу,
//    и вся синхронизация сводится к трём барьерам на ВЕСЬ алгоритм, а не на элемент.
//    Цена — T·K счётчиков (T·256 КиБ) и O(T·K) работы по слиянию, не зависящие от N.
//
//  Почему потоки не пишут в одни и те же позиции.
//    Для фиксированного значения v диапазон [start[v], start[v+1]) нарезается на подряд
//    идущие куски длиной local[0][v], local[1][v], ..., local[T-1][v], в порядке номеров
//    потоков. Куски не перекрываются и покрывают диапазон без пропусков; их суммарная
//    длина global[v] = start[v+1] - start[v]. Поток t за фазу 3 увеличивает свой курсор
//    для v ровно local[t][v] раз (он видит те же элементы, что и при подсчёте), то есть
//    никогда не выходит за свой кусок. Значит, каждая позиция выхода записывается ровно
//    одним потоком ровно один раз.
//
//  Почему сохраняется стабильность.
//    Элементы с равным ключом x в исходном массиве идут так: сначала все из блока 0, затем
//    из блока 1 и т. д. (блоки — последовательные отрезки входа). В выходе куски для x тоже
//    идут в порядке номеров потоков, а внутри блока проход слева направо с растущим курсором
//    сохраняет взаимный порядок. Значит, порядок равных ключей сохраняется глобально.
//    (Для «голых» uint16 это ненаблюдаемо, но по конструкции алгоритм стабилен и останется
//    таковым, если к ключу добавить полезную нагрузку.)
//
//  Зачем повторно используется память локальных гистограмм.
//    После фазы 2 исходный счётчик local[t][v] больше не нужен: на его месте хранится
//    текущая позиция записи потока t для значения v. Отдельная матрица смещений
//    thread_offset[T][K] не требуется — экономим T·K·sizeof(Count) байт и один проход
//    по памяти. Фаза 3 только увеличивает эти курсоры.
//
//  Почему нагрузка делится по элементам, а не по значениям.
//    При делении по диапазонам значений (поток 0 — значения 0..16383 и т. д.) нагрузка
//    целиком зависит от распределения данных: если все элементы попадают в один диапазон,
//    работает один поток. Деление входа по числу элементов даёт каждому потоку N/T
//    элементов в фазах 1 и 3 при ЛЮБОМ распределении. В фазе 2 работа не зависит от
//    данных и делится поровну по K/T значений на поток.
//
//  Сложность.
//    Суммарная работа O(N + K + T·K): подсчёт и раскладка — O(N), обнуление/слияние
//    гистограмм и префиксные суммы — O(T·K). Глубина (время при T ядрах) —
//    O(N/T + K): в фазе 2 каждый поток суммирует T таблиц на своих K/T значениях.
//
//  Модель памяти OpenMP.
//    Данные между фазами передаются через общую память; согласованность обеспечивают
//    `#pragma omp barrier` (барьер неявно содержит flush). Внутри фазы любая ячейка памяти
//    пишется не более чем одним потоком и не читается другими.
//
//  Размер команды потоков.
//    Рантайм OpenMP вправе выдать меньше потоков, чем запрошено (ограничения среды,
//    вложенный параллелизм и т. п.). Поэтому ВСЕ разбиения считаются по фактическому
//    omp_get_num_threads() внутри параллельной области, а не по запрошенному числу —
//    иначе часть блоков осталась бы необработанной.
//
// ============================================================================
//  РАСШИРЕНИЕ: fill (serial_histogram_fill_sort / parallel_histogram_fill_sort)
// ============================================================================
//  Для ключей без полезной нагрузки отсортированный массив полностью задаётся гистограммой,
//  поэтому после подсчёта и префиксных сумм элементы не «раскладываются» — выходной массив
//  заливается диапазонами: [start[v], start[v+1]) := v. Фаза 3 превращается в последовательную
//  запись (скорость порядка пропускной способности памяти), второй массив не нужен.
//  Параллелизм сохраняется: фазы 1–2 те же, а в фазе 3 каждый поток заливает свой равный по
//  размеру отрезок выхода (балансировка по элементам; отрезки не пересекаются).
// ============================================================================

#include "radix_sort.hpp"

#include <omp.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace radix {

namespace {

// Сколько счётчиков помещается в одну кэш-линию. Границы диапазонов значений в фазе 2
// кратны этому числу, чтобы два потока никогда не писали в одну и ту же линию таблиц.
constexpr std::size_t kLineValues = kCacheLine / sizeof(Count);
constexpr std::size_t kValueLines = kKeyRange / kLineValues;

// Начало i-го диапазона значений из `parts` (в «номерах значений», кратно kLineValues).
constexpr std::size_t value_range_begin(std::size_t i, std::size_t parts) noexcept {
    return block_begin(i, parts, kValueLines) * kLineValues;
}

// ---------------------------------------------------------------------------
// Проверки аргументов (выполняются ДО входа в параллельную область: исключения
// не должны покидать область OpenMP)
// ---------------------------------------------------------------------------

[[noreturn]] void throw_too_large(std::size_t n) {
    throw std::length_error("radix sort: " + std::to_string(n) + " elements do not fit into " +
                            std::to_string(sizeof(Count) * 8) + "-bit counters (max " +
                            std::to_string(max_supported_elements()) +
                            "); rebuild with -DRADIX_WIDE_COUNTERS=ON");
}

void check_size(std::size_t n) {
    if (n > max_supported_elements()) throw_too_large(n);
}

void check_threads(int num_threads) {
    if (num_threads < 1 || num_threads > kMaxThreads) {
        throw std::invalid_argument("radix sort: number of threads must be in [1, " +
                                    std::to_string(kMaxThreads) + "], got " +
                                    std::to_string(num_threads));
    }
}

void check_distinct(const std::vector<Key>& data, const std::vector<Key>& buffer) {
    if (&data == &buffer) {
        throw std::invalid_argument("radix sort: data and buffer must be different vectors");
    }
}

// Сколько потоков запрашиваем у OpenMP: не больше, чем элементов (при N < T лишним потокам
// нечего делать). n >= 1.
int effective_threads(int num_threads, std::size_t n) noexcept {
    const std::size_t t = std::min<std::size_t>(static_cast<std::size_t>(num_threads), n);
    return static_cast<int>(t);
}

// ---------------------------------------------------------------------------
// Фазы параллельного алгоритма. Каждая функция выполняется ОДНИМ потоком над
// «своими» данными; барьеры расставлены в вызывающем коде.
// ---------------------------------------------------------------------------

// Фаза 1. Локальная гистограмма блока [first, last) в таблице потока.
// Обнуление делает сам владелец таблицы (first-touch для NUMA); счётчики берутся только из
// собственной таблицы, поэтому ни atomic, ни critical не нужны.
inline void phase_count(Histogram& h, const Key* first, const Key* last) noexcept {
    Count* const bins = h.bins;
    std::memset(bins, 0, sizeof h.bins);
    for (const Key* p = first; p != last; ++p) ++bins[*p];
}

// Фаза 2a. starts[v] = Σ_t local[t][v] для v ∈ [v_lo, v_hi); возвращает сумму по диапазону.
// Цикл «по потокам снаружи, по значениям внутри» читает каждую строку подряд (хорошая
// локальность и автовекторизация). Обход «по значениям снаружи» был бы T потоковых строк
// с шагом 256 КиБ — они конфликтуют в одних и тех же наборах кэша L1.
inline Count phase_sum_columns(const Histogram* hist, std::size_t nt, Count* starts,
                               std::size_t v_lo, std::size_t v_hi) noexcept {
    std::fill(starts + v_lo, starts + v_hi, Count{0});
    for (std::size_t t = 0; t < nt; ++t) {
        const Count* const h = hist[t].bins;
        for (std::size_t v = v_lo; v < v_hi; ++v) starts[v] += h[v];
    }
    Count total = 0;
    for (std::size_t v = v_lo; v < v_hi; ++v) total += starts[v];
    return total;
}

// Фаза 2b. Exclusive prefix sum по значениям внутри диапазона; base — сумма всех элементов
// в предыдущих диапазонах. После вызова starts[v] = число элементов со значениями < v.
// Глобальный prefix sum строится в два уровня: сумма по диапазону -> сумма по предыдущим
// диапазонам (T чисел, их читает каждый поток сам) -> локальный префикс внутри диапазона.
inline void phase_scan_range(Count* starts, std::size_t v_lo, std::size_t v_hi, Count base) noexcept {
    for (std::size_t v = v_lo; v < v_hi; ++v) {
        const Count c = starts[v];
        starts[v] = base;
        base += c;
    }
}

// Фаза 2c. Диапазоны потоков и повторное использование матрицы гистограмм.
//
// Для каждого значения v потоки получают подряд идущие куски выходного массива в порядке
// номеров t = 0, 1, ..., T-1. Здесь starts[v] играет роль «бегущей границы»: сначала это
// начало всех элементов v, а после обработки потока t — конец его куска (= начало куска
// потока t+1). Счётчик local[t][v] заменяется на начало куска потока t — то есть на
// стартовую позицию записи. По завершении starts[v] равен концу всех элементов v.
inline void phase_make_cursors(Histogram* hist, std::size_t nt, Count* starts, std::size_t v_lo,
                               std::size_t v_hi) noexcept {
    for (std::size_t t = 0; t < nt; ++t) {
        Count* const h = hist[t].bins;
        for (std::size_t v = v_lo; v < v_hi; ++v) {
            const Count c = h[v];  // сколько элементов v в блоке потока t
            h[v] = starts[v];      // откуда поток t начинает писать значение v
            starts[v] += c;        // следующий поток продолжит сразу после куска потока t
        }
    }
}

// Фаза 3 (scatter). Раскладка блока [first, last) слева направо в собственные диапазоны.
// Курсор каждого значения монотонно растёт, поэтому элементы блока с одинаковым ключом
// попадают в выход в том же порядке, в каком стояли во входе (стабильность внутри блока).
inline void phase_scatter(Histogram& h, const Key* first, const Key* last, Key* out) noexcept {
    Count* const cursor = h.bins;
    for (const Key* p = first; p != last; ++p) {
        const Key x = *p;
        out[cursor[x]++] = x;
    }
}

// Фаза 3 (fill). Залить выходные позиции [lo, hi) значениями из таблицы стартов
// (starts[K] = n — граница-часовой). Вызывается после барьера, когда все чтения входа
// завершены, поэтому писать можно прямо во вход (на месте).
inline void phase_fill(const Count* starts, std::size_t lo, std::size_t hi, Key* out) noexcept {
    if (lo >= hi) return;
    // Первый индекс u с starts[u] > lo; значению, которому принадлежит позиция lo, соответствует u-1.
    // Здесь lo < n = starts[K], поэтому 1 <= u <= K и v = u-1 лежит в [0, K-1].
    std::size_t v = static_cast<std::size_t>(
                        std::upper_bound(starts, starts + kKeyRange + 1, static_cast<Count>(lo)) - starts) -
                    1;
    std::size_t pos = lo;
    while (pos < hi) {
        const std::size_t run_end = starts[v + 1];  // конец диапазона значения v
        if (run_end > pos) {                        // пустые значения (run_end == start[v]) пропускаем
            const std::size_t stop = std::min(run_end, hi);
            std::fill(out + pos, out + stop, static_cast<Key>(v));
            pos = stop;
        }
        ++v;  // pos < hi <= n = starts[K], поэтому v не выйдет за K-1, пока цикл работает
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Algorithm / MemoryEstimate / Workspace
// ---------------------------------------------------------------------------

const char* to_string(Algorithm a) noexcept {
    switch (a) {
        case Algorithm::Scatter: return "scatter";
        case Algorithm::Fill: return "fill";
    }
    return "?";
}

void ParallelRadixWorkspace::reserve(int max_threads) {
    check_threads(max_threads);
    if (max_threads <= capacity_) return;
    const std::size_t t = static_cast<std::size_t>(max_threads);
    // Память выделяется БЕЗ обнуления: Histogram/StartTable/PaddedCount не инициализируются.
    // Если одно из выделений бросит bad_alloc, capacity_ остаётся прежним (не больше фактического).
    std::unique_ptr<Histogram[]> hist(new Histogram[t]);
    std::unique_ptr<StartTable> starts(new StartTable);
    std::unique_ptr<PaddedCount[]> totals(new PaddedCount[t]);
    hist_ = std::move(hist);
    starts_ = std::move(starts);
    totals_ = std::move(totals);
    capacity_ = max_threads;
}

std::uint64_t ParallelRadixWorkspace::bytes() const noexcept {
    if (capacity_ == 0) return 0;
    const std::uint64_t t = static_cast<std::uint64_t>(capacity_);
    return t * sizeof(Histogram) + sizeof(StartTable) + t * sizeof(PaddedCount);
}

MemoryEstimate estimate_memory(std::size_t n, int threads, Algorithm algo, bool parallel) {
    MemoryEstimate m;
    m.input_bytes = static_cast<std::uint64_t>(n) * sizeof(Key);
    m.output_bytes = (algo == Algorithm::Scatter) ? m.input_bytes : 0;
    if (parallel) {
        const std::uint64_t cap = n == 0 ? 1 : static_cast<std::uint64_t>(n);
        const std::uint64_t t = std::max<std::uint64_t>(1, std::min<std::uint64_t>(static_cast<std::uint64_t>(std::max(threads, 1)), cap));
        m.histogram_bytes = t * sizeof(Histogram);
        m.aux_bytes = sizeof(StartTable) + t * sizeof(PaddedCount);
    } else {
        m.histogram_bytes = (kKeyRange + 1) * sizeof(Count);  // единственная таблица bound[0..K]
        m.aux_bytes = 0;
    }
    return m;
}

// ---------------------------------------------------------------------------
// Последовательный scatter
// ---------------------------------------------------------------------------

void serial_radix_sort(std::vector<Key>& data, std::vector<Key>& buffer, PhaseTimes* times) {
    const std::size_t n = data.size();
    check_size(n);
    check_distinct(data, buffer);

    PhaseTimes pt;
    if (n == 0) {
        if (times) *times = pt;
        return;
    }
    if (buffer.size() != n) buffer.resize(n);

    const double t0 = omp_get_wtime();

    // Одна таблица bound[0..K] вместо пары count/start.
    //   Шаг 1: считаем в сдвинутую ячейку: bound[x + 1] = count[x].
    //   Шаг 2: обычная накопленная сумма bound[v] += bound[v-1] даёт exclusive prefix sum:
    //          bound[v] = start[v] = число элементов < v, bound[K] = n.
    //          Диапазон значения v — [bound[v], bound[v+1]).
    std::vector<Count> bound(kKeyRange + 1, Count{0});
    Count* const c = bound.data();
    const Key* const in = data.data();
    Key* const out = buffer.data();

    // Шаг 1. Подсчёт.
    for (std::size_t i = 0; i < n; ++i) ++c[static_cast<std::size_t>(in[i]) + 1];
    const double t1 = omp_get_wtime();

    // Шаг 2. Exclusive prefix sum.
    for (std::size_t v = 1; v <= kKeyRange; ++v) c[v] += c[v - 1];
    const double t2 = omp_get_wtime();

    // Шаг 3. Стабильная раскладка С КОНЦА входа (классическая counting sort).
    // range_end[x] = bound[x+1] — конец диапазона значения x. Предекремент даёт позиции
    // end-1, end-2, ...: ПОСЛЕДНИЙ в исходном массиве элемент x ложится на последнее место
    // своего диапазона, предпоследний — перед ним и т. д. — порядок равных ключей сохранён.
    Count* const range_end = c + 1;
    for (std::size_t i = n; i > 0; --i) {
        const Key x = in[i - 1];
        out[--range_end[x]] = x;
    }
    const double t3 = omp_get_wtime();

    data.swap(buffer);  // результат — в data; старый вход остаётся в buffer

    pt.count_s = t1 - t0;
    pt.prefix_s = t2 - t1;
    pt.move_s = t3 - t2;
    if (times) *times = pt;
}

// ---------------------------------------------------------------------------
// Параллельный scatter
// ---------------------------------------------------------------------------

void parallel_radix_sort(std::vector<Key>& data, std::vector<Key>& buffer, int num_threads) {
    ParallelRadixWorkspace workspace;  // временная область на один вызов
    parallel_radix_sort(data, buffer, num_threads, workspace, nullptr);
}

void parallel_radix_sort(std::vector<Key>& data, std::vector<Key>& buffer, int num_threads,
                         ParallelRadixWorkspace& workspace, PhaseTimes* times) {
    check_threads(num_threads);
    const std::size_t n = data.size();
    check_size(n);
    check_distinct(data, buffer);

    PhaseTimes pt;
    if (n == 0) {  // нечего сортировать; не плодим потоки и не выделяем память
        if (times) *times = pt;
        return;
    }
    if (buffer.size() != n) buffer.resize(n);

    const int requested = effective_threads(num_threads, n);
    workspace.reserve(requested);  // no-op, если область уже достаточно велика

    const Key* const in = data.data();
    Key* const out = buffer.data();
    Histogram* const hist = workspace.histograms();
    Count* const starts = workspace.starts();
    PaddedCount* const totals = workspace.totals();

    int threads_used = 1;
    double t_count = 0.0, t_merge = 0.0;  // пишет только поток 0 (master)
    const double t_begin = omp_get_wtime();

#pragma omp parallel num_threads(requested)
    {
        const std::size_t tid = static_cast<std::size_t>(omp_get_thread_num());
        const std::size_t nt = static_cast<std::size_t>(omp_get_num_threads());  // ФАКТИЧЕСКИЙ размер команды

        // Блок входа потока: деление по ЧИСЛУ ЭЛЕМЕНТОВ (см. комментарий в начале файла).
        const Key* const first = in + block_begin(tid, nt, n);
        const Key* const last = in + block_begin(tid + 1, nt, n);
        // Диапазон значений потока в фазе 2 (границы кратны размеру кэш-линии).
        const std::size_t v_lo = value_range_begin(tid, nt);
        const std::size_t v_hi = value_range_begin(tid + 1, nt);

        // Фаза 1: локальная гистограмма (без синхронизации между потоками).
        phase_count(hist[tid], first, last);
#pragma omp barrier  // все локальные гистограммы готовы
#pragma omp master
        {
            t_count = omp_get_wtime();
            threads_used = static_cast<int>(nt);
        }

        // Фаза 2a: глобальные счётчики по своему диапазону значений.
        totals[tid].value = phase_sum_columns(hist, nt, starts, v_lo, v_hi);
#pragma omp barrier  // суммы по диапазонам записаны всеми потоками

        // Фаза 2b: глобальный exclusive prefix sum (база = сумма по предыдущим диапазонам).
        Count base = 0;
        for (std::size_t t = 0; t < tid; ++t) base += totals[t].value;
        phase_scan_range(starts, v_lo, v_hi, base);

        // Фаза 2c: счётчики -> стартовые позиции записи каждой пары (поток, значение).
        phase_make_cursors(hist, nt, starts, v_lo, v_hi);
#pragma omp barrier  // курсоры всех потоков готовы (строку потока меняли другие потоки)
#pragma omp master
        {
            t_merge = omp_get_wtime();
        }

        // Фаза 3: стабильная раскладка блока в собственные диапазоны выходного массива.
        phase_scatter(hist[tid], first, last, out);
    }  // неявный барьер: все записи в out видны после выхода из области

    const double t_end = omp_get_wtime();
    data.swap(buffer);  // результат — в data

    pt.count_s = t_count - t_begin;
    pt.prefix_s = t_merge - t_count;
    pt.move_s = t_end - t_merge;
    pt.threads_used = threads_used;
    if (times) *times = pt;
}

// ---------------------------------------------------------------------------
// Расширение: fill
// ---------------------------------------------------------------------------

void serial_histogram_fill_sort(std::vector<Key>& data, PhaseTimes* times) {
    const std::size_t n = data.size();
    check_size(n);

    PhaseTimes pt;
    if (n == 0) {
        if (times) *times = pt;
        return;
    }

    const double t0 = omp_get_wtime();
    std::vector<Count> count(kKeyRange, Count{0});
    for (const Key x : data) ++count[x];
    const double t1 = omp_get_wtime();

    // Все чтения входа закончены — перезаписываем массив на месте.
    Key* p = data.data();
    for (std::size_t v = 0; v < kKeyRange; ++v) {
        const std::size_t c = count[v];
        std::fill(p, p + c, static_cast<Key>(v));
        p += c;
    }
    const double t2 = omp_get_wtime();

    pt.count_s = t1 - t0;
    pt.move_s = t2 - t1;
    if (times) *times = pt;
}

void parallel_histogram_fill_sort(std::vector<Key>& data, int num_threads, ParallelRadixWorkspace& workspace,
                                  PhaseTimes* times) {
    check_threads(num_threads);
    const std::size_t n = data.size();
    check_size(n);

    PhaseTimes pt;
    if (n == 0) {
        if (times) *times = pt;
        return;
    }

    const int requested = effective_threads(num_threads, n);
    workspace.reserve(requested);

    Key* const io = data.data();  // и вход, и выход: запись начинается после барьера
    Histogram* const hist = workspace.histograms();
    Count* const starts = workspace.starts();
    PaddedCount* const totals = workspace.totals();
    starts[kKeyRange] = static_cast<Count>(n);  // граница-часовой для бинарного поиска и заливки

    int threads_used = 1;
    double t_count = 0.0, t_merge = 0.0;
    const double t_begin = omp_get_wtime();

#pragma omp parallel num_threads(requested)
    {
        const std::size_t tid = static_cast<std::size_t>(omp_get_thread_num());
        const std::size_t nt = static_cast<std::size_t>(omp_get_num_threads());

        const std::size_t lo = block_begin(tid, nt, n);
        const std::size_t hi = block_begin(tid + 1, nt, n);
        const std::size_t v_lo = value_range_begin(tid, nt);
        const std::size_t v_hi = value_range_begin(tid + 1, nt);

        phase_count(hist[tid], io + lo, io + hi);
#pragma omp barrier
#pragma omp master
        {
            t_count = omp_get_wtime();
            threads_used = static_cast<int>(nt);
        }

        totals[tid].value = phase_sum_columns(hist, nt, starts, v_lo, v_hi);
#pragma omp barrier

        Count base = 0;
        for (std::size_t t = 0; t < tid; ++t) base += totals[t].value;
        phase_scan_range(starts, v_lo, v_hi, base);
#pragma omp barrier  // таблица стартов нужна целиком (бинарный поиск), и все чтения входа завершены
#pragma omp master
        {
            t_merge = omp_get_wtime();
        }

        // Выходной отрезок потока — тот же по размеру блок [lo, hi): N/T элементов на поток.
        phase_fill(starts, lo, hi, io);
    }

    const double t_end = omp_get_wtime();
    pt.count_s = t_count - t_begin;
    pt.prefix_s = t_merge - t_count;
    pt.move_s = t_end - t_merge;
    pt.threads_used = threads_used;
    if (times) *times = pt;
}

}  // namespace radix
