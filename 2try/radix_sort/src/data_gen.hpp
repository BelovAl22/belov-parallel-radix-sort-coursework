// data_gen.hpp — детерминированная генерация тестовых массивов uint16_t.
//
// Генератор «счётчиковый»: значение элемента с номером j зависит ТОЛЬКО от (seed, j, N,
// распределение) и не зависит ни от числа потоков, ни от порядка генерации. Поэтому:
//   * результат воспроизводим для любого --threads и на любом компиляторе/платформе
//     (стандартные std::*_distribution не годятся: их результат зависит от реализации);
//   * генерацию можно вести порциями (чанками) без временных структур — так создаётся
//     файл на миллиард элементов, не держа весь массив в памяти;
//   * генерация легко распараллеливается (OpenMP) без синхронизации.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "radix_sort.hpp"

namespace radix {

enum class Distribution {
    Uniform,    // независимые, равномерно распределённые по 0..65535
    Random,     // независимые, «колоколообразные» (среднее четырёх равномерных): неравномерная плотность
    Sorted,     // неубывающая «лесенка» 0..65535 (уже отсортирован)
    Reverse,    // невозрастающая «лесенка» (обратный порядок)
    FewValues,  // всего 16 различных значений (в том числе 0 и 65535)
    Zeros,      // все элементы равны 0
    Max,        // все элементы равны 65535
    TwoValues   // только 0 и 65535
};

const char* to_string(Distribution d) noexcept;

// Краткое описание для --help: "uniform|random|sorted|..."
const char* distribution_names() noexcept;

// Разбор имени (без учёта регистра; '_' и '-' равнозначны; допускается "few" вместо "few-values").
bool parse_distribution(const std::string& name, Distribution& out);

// Записывает в out[0..count) элементы с номерами first_index .. first_index+count-1
// виртуального массива из `total` элементов. Параллельна (OpenMP), результат от числа потоков не зависит.
void generate_keys(Key* out, std::size_t first_index, std::size_t count, std::size_t total,
                   Distribution dist, std::uint64_t seed);

// Удобная обёртка: data.resize(n) и заполнение целиком.
void generate_keys(std::vector<Key>& data, std::size_t n, Distribution dist, std::uint64_t seed);

}  // namespace radix
