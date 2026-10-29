// data_gen.cpp — счётчиковый генератор на основе SplitMix64.
#include "data_gen.hpp"

#include <algorithm>
#include <cctype>

namespace radix {

namespace {

constexpr std::uint64_t kGolden = 0x9E3779B97F4A7C15ull;

// SplitMix64 в «безсостоянийной» форме: слово №index потока со стартовым состоянием seed.
// Это ровно j-й выход стандартного генератора SplitMix64 (state += golden; mix(state)),
// но вычисляемый по номеру, а не последовательно — отсюда независимость от числа потоков.
inline std::uint64_t random_word(std::uint64_t seed, std::uint64_t index) noexcept {
    std::uint64_t z = seed + (index + 1) * kGolden;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Заполнение out[0..count) значениями gen(i), i — номер элемента внутри порции.
// Индекс цикла знаковый: этого требует OpenMP 2.0 (MSVC).
template <class Gen>
void fill_keys(Key* out, std::size_t count, Gen gen) {
    const std::int64_t n = static_cast<std::int64_t>(count);
#pragma omp parallel for schedule(static) if (n >= (std::int64_t{1} << 16))
    for (std::int64_t i = 0; i < n; ++i) out[i] = gen(static_cast<std::uint64_t>(i));
}

// Одно из 16 значений для режима few-values: k * 4369, k = 0..15 (4369 * 15 = 65535).
constexpr std::uint32_t kFewStep = 4369;

}  // namespace

const char* to_string(Distribution d) noexcept {
    switch (d) {
        case Distribution::Uniform: return "uniform";
        case Distribution::Random: return "random";
        case Distribution::Sorted: return "sorted";
        case Distribution::Reverse: return "reverse";
        case Distribution::FewValues: return "few-values";
        case Distribution::Zeros: return "zeros";
        case Distribution::Max: return "max";
        case Distribution::TwoValues: return "two-values";
    }
    return "?";
}

const char* distribution_names() noexcept {
    return "uniform|random|sorted|reverse|few-values|zeros|max|two-values";
}

bool parse_distribution(const std::string& name, Distribution& out) {
    std::string s;
    s.reserve(name.size());
    for (const char ch : name) {
        const unsigned char c = static_cast<unsigned char>(ch);
        s.push_back(c == '_' ? '-' : static_cast<char>(std::tolower(c)));
    }
    if (s == "uniform") out = Distribution::Uniform;
    else if (s == "random") out = Distribution::Random;
    else if (s == "sorted") out = Distribution::Sorted;
    else if (s == "reverse") out = Distribution::Reverse;
    else if (s == "few-values" || s == "few") out = Distribution::FewValues;
    else if (s == "zeros") out = Distribution::Zeros;
    else if (s == "max") out = Distribution::Max;
    else if (s == "two-values") out = Distribution::TwoValues;
    else return false;
    return true;
}

void generate_keys(Key* out, std::size_t first_index, std::size_t count, std::size_t total,
                   Distribution dist, std::uint64_t seed) {
    if (count == 0) return;
    const std::uint64_t first = first_index;
    const std::uint64_t tot = total;  // total >= first_index + count >= 1

    switch (dist) {
        case Distribution::Uniform:
            fill_keys(out, count, [=](std::uint64_t i) {
                return static_cast<Key>(random_word(seed, first + i) >> 48);
            });
            break;
        case Distribution::Random:
            // Среднее четырёх 16-битных срезов одного слова: сумма <= 4*65535, после >> 2 — в диапазоне ключа.
            fill_keys(out, count, [=](std::uint64_t i) {
                const std::uint64_t w = random_word(seed, first + i);
                const std::uint64_t s = (w & 0xFFFFu) + ((w >> 16) & 0xFFFFu) + ((w >> 32) & 0xFFFFu) + (w >> 48);
                return static_cast<Key>(s >> 2);
            });
            break;
        case Distribution::Sorted:
            // floor(j * 65536 / N) < 65536 при j < N; j * 65536 не переполняет uint64 для любых реальных N.
            fill_keys(out, count, [=](std::uint64_t i) {
                return static_cast<Key>(((first + i) << kKeyBits) / tot);
            });
            break;
        case Distribution::Reverse:
            fill_keys(out, count, [=](std::uint64_t i) {
                return static_cast<Key>(kKeyRange - 1 - (((first + i) << kKeyBits) / tot));
            });
            break;
        case Distribution::FewValues:
            fill_keys(out, count, [=](std::uint64_t i) {
                const std::uint32_t k = static_cast<std::uint32_t>(random_word(seed, first + i) >> 60);  // 0..15
                return static_cast<Key>(k * kFewStep);
            });
            break;
        case Distribution::Zeros:
            fill_keys(out, count, [](std::uint64_t) { return Key{0}; });
            break;
        case Distribution::Max:
            fill_keys(out, count, [](std::uint64_t) { return static_cast<Key>(kKeyRange - 1); });
            break;
        case Distribution::TwoValues:
            fill_keys(out, count, [=](std::uint64_t i) {
                return (random_word(seed, first + i) >> 63) ? static_cast<Key>(kKeyRange - 1) : Key{0};
            });
            break;
    }
}

void generate_keys(std::vector<Key>& data, std::size_t n, Distribution dist, std::uint64_t seed) {
    data.resize(n);
    generate_keys(data.data(), 0, n, n, dist, seed);
}

}  // namespace radix
