// sysinfo.hpp — сведения об оперативной памяти для предварительной проверки перед запуском.
//
// Зачем это нужно. Параллельная и последовательная сортировки подсчётом раскладывают элементы
// случайным образом по ВСЕМУ выходному массиву (65536 «потоков записи»). Если массивы не
// помещаются в ОЗУ и ОС начинает использовать swap (на macOS — сжатие памяти и swap), каждая такая
// запись превращается в обращение к диску, и сортировка 10^9 элементов занимает не секунды, а многие
// часы. Поэтому программа заранее сравнивает потребность в памяти с доступной и отказывается
// стартовать (или предупреждает).
#pragma once

#include <cstdint>
#include <string>

namespace radix {

struct MemoryInfo {
    std::uint64_t total = 0;      // физическая память, видимая процессу, байт (0 — неизвестно)
    std::uint64_t available = 0;  // сколько можно занять без вытеснения в swap, байт (0 — неизвестно)
    bool exact_available = false; // false: available — лишь приближение (ОС не сообщила свободную память)

    bool known() const noexcept { return total != 0 && available != 0; }
};

// Best-effort: Linux (/proc/meminfo: MemAvailable), Windows (GlobalMemoryStatusEx),
// macOS (sysctl hw.memsize + вывод `vm_stat`: free + inactive + speculative + purgeable страницы).
// На прочих системах — «неизвестно».
MemoryInfo query_memory();

namespace detail {

// Разбор вывода команды `vm_stat` (macOS). Возвращает число байт, которые можно занять без
// вытеснения в swap: (free + inactive + speculative + purgeable страниц) * размер страницы.
// 0 — если разобрать не удалось (нет размера страницы или обязательных строк).
// Вынесено отдельно, чтобы быть проверяемым тестами на любой ОС.
std::uint64_t parse_vm_stat_available(const std::string& text);

}  // namespace detail

}  // namespace radix
