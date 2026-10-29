// verify.hpp — проверка корректности результата сортировки.
//
// Проверяется:
//   1. массив отсортирован (неубывающий);
//   2. число элементов не изменилось;
//   3. контрольные суммы (Σx и Σx², по модулю 2^64) до и после совпадают;
//   4. точное совпадение мультимножеств: 65536 счётчиков значений «до» и «после» равны.
//      Вместе с п. 1 это ДОКАЗЫВАЕТ корректность: отсортированная перестановка заданного
//      мультимножества единственна, поэтому результат обязан совпасть с эталоном;
//   5. (для небольших массивов) прямое сравнение с результатом std::sort.
// Проверка не зависит от кода сортировки (свои счётчики типа uint64_t) и не входит во время сортировки.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "radix_sort.hpp"

namespace radix {

// Для массивов не больше этого размера дополнительно выполняется сравнение с std::sort.
inline constexpr std::size_t kReferenceSortLimit = 10'000'000;

// «Отпечаток» массива, не зависящий от порядка элементов.
struct Fingerprint {
    std::uint64_t count = 0;
    std::uint64_t sum = 0;     // Σ x   (mod 2^64)
    std::uint64_t sum_sq = 0;  // Σ x²  (mod 2^64)
    std::vector<std::uint64_t> histogram;  // histogram[v] — сколько раз встретилось значение v
};

Fingerprint make_fingerprint(const Key* data, std::size_t n);
inline Fingerprint make_fingerprint(const std::vector<Key>& v) { return make_fingerprint(v.data(), v.size()); }

struct Verification {
    bool sorted = false;
    bool count_ok = false;
    bool checksum_ok = false;
    bool histogram_ok = false;
    bool reference_checked = false;  // сравнение с std::sort выполнялось
    bool reference_ok = false;

    bool passed() const noexcept {
        return sorted && count_ok && checksum_ok && histogram_ok && (!reference_checked || reference_ok);
    }
};

// Копия массива, отсортированная std::sort (эталон для небольших массивов).
std::vector<Key> make_reference_sorted(const std::vector<Key>& original);

// result — проверяемый массив; expected — отпечаток ИСХОДНОГО массива (снятый до сортировки);
// reference_sorted — nullptr либо эталон из make_reference_sorted.
Verification verify_result(const std::vector<Key>& result, const Fingerprint& expected,
                           const std::vector<Key>* reference_sorted);

}  // namespace radix
