// verify.cpp — независимая от сортировки проверка результата.
#include "verify.hpp"

#include <algorithm>

namespace radix {

Fingerprint make_fingerprint(const Key* data, std::size_t n) {
    Fingerprint fp;
    fp.count = n;
    fp.histogram.assign(kKeyRange, 0);
    std::uint64_t* const h = fp.histogram.data();
    std::uint64_t sum = 0, sum_sq = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t x = data[i];
        ++h[x];
        sum += x;
        sum_sq += x * x;  // арифметика по модулю 2^64 — это нормально для контрольной суммы
    }
    fp.sum = sum;
    fp.sum_sq = sum_sq;
    return fp;
}

std::vector<Key> make_reference_sorted(const std::vector<Key>& original) {
    std::vector<Key> ref = original;
    std::sort(ref.begin(), ref.end());
    return ref;
}

Verification verify_result(const std::vector<Key>& result, const Fingerprint& expected,
                           const std::vector<Key>* reference_sorted) {
    Verification v;
    v.sorted = std::is_sorted(result.begin(), result.end());

    const Fingerprint actual = make_fingerprint(result);
    v.count_ok = (actual.count == expected.count);
    v.checksum_ok = (actual.sum == expected.sum) && (actual.sum_sq == expected.sum_sq);
    v.histogram_ok = (actual.histogram == expected.histogram);

    if (reference_sorted != nullptr) {
        v.reference_checked = true;
        v.reference_ok = (result == *reference_sorted);
    }
    return v;
}

}  // namespace radix
