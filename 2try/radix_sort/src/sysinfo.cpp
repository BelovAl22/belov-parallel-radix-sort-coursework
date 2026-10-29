// sysinfo.cpp — реализация query_memory() для Linux / Windows / macOS.
#include "sysinfo.hpp"

#include <cstddef>  // std::size_t (в libc++ на macOS его нет «по цепочке» от <cstdint>)
#include <cstdio>
#include <sstream>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX  // windows.h не должен определять макросы min/max
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <stdio.h>  // popen / pclose (POSIX)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace radix {

namespace detail {

std::uint64_t parse_vm_stat_available(const std::string& text) {
    // Формат вывода `vm_stat`:
    //   Mach Virtual Memory Statistics: (page size of 16384 bytes)
    //   Pages free:                               5324.
    //   Pages inactive:                         198765.
    //   Pages speculative:                        1234.
    //   Pages purgeable:                          2000.
    std::istringstream in(text);
    std::string line;
    unsigned long long page_size = 0;
    unsigned long long pages = 0;
    bool have_free = false;
    bool have_inactive = false;
    while (std::getline(in, line)) {
        unsigned long long v = 0;
        if (std::sscanf(line.c_str(), "Mach Virtual Memory Statistics: (page size of %llu bytes)", &v) == 1) {
            page_size = v;
        } else if (std::sscanf(line.c_str(), "Pages free: %llu", &v) == 1) {
            pages += v;
            have_free = true;
        } else if (std::sscanf(line.c_str(), "Pages inactive: %llu", &v) == 1) {
            pages += v;
            have_inactive = true;
        } else if (std::sscanf(line.c_str(), "Pages speculative: %llu", &v) == 1) {
            pages += v;
        } else if (std::sscanf(line.c_str(), "Pages purgeable: %llu", &v) == 1) {
            pages += v;
        }
    }
    if (page_size == 0 || !have_free || !have_inactive) return 0;
    return static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page_size);
}

}  // namespace detail

MemoryInfo query_memory() {
    MemoryInfo m;

#if defined(_WIN32)
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) {
        m.total = static_cast<std::uint64_t>(st.ullTotalPhys);
        m.available = static_cast<std::uint64_t>(st.ullAvailPhys);  // свободная + «резервная» (кэш), освобождаемая по требованию
        m.exact_available = true;
    }
#elif defined(__APPLE__)
    std::uint64_t mem = 0;
    std::size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) {
        m.total = mem;
        m.available = mem;  // запасной вариант: если свободную память узнать не удалось, берём всю как верхнюю границу
        m.exact_available = false;
        // macOS держит почти всю ОЗУ занятой кэшами и «неактивными» страницами, которые освобождаются по
        // требованию, поэтому свободной считаем сумму free + inactive + speculative + purgeable (из `vm_stat`).
        std::string text;
        if (std::FILE* pipe = ::popen("/usr/bin/vm_stat", "r")) {
            char buf[512];
            std::size_t got = 0;
            while ((got = std::fread(buf, 1, sizeof buf, pipe)) > 0) text.append(buf, got);
            ::pclose(pipe);
        }
        const std::uint64_t avail = detail::parse_vm_stat_available(text);
        if (avail != 0) {
            m.available = (avail < mem) ? avail : mem;
            m.exact_available = true;
        }
    }
#elif defined(__linux__)
    // MemAvailable — оценка ядра: сколько можно выделить, не вытесняя в swap (учитывает освобождаемый кэш страниц).
    // Внутри WSL2 /proc/meminfo отражает лимит виртуальной машины, а не всего компьютера.
    if (std::FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        unsigned long long total_kb = 0, avail_kb = 0, free_kb = 0, cached_kb = 0, buffers_kb = 0;
        bool have_avail = false;
        while (std::fgets(line, sizeof line, f) != nullptr) {
            unsigned long long v = 0;
            if (std::sscanf(line, "MemTotal: %llu kB", &v) == 1) total_kb = v;
            else if (std::sscanf(line, "MemAvailable: %llu kB", &v) == 1) { avail_kb = v; have_avail = true; }
            else if (std::sscanf(line, "MemFree: %llu kB", &v) == 1) free_kb = v;
            else if (std::sscanf(line, "Cached: %llu kB", &v) == 1) cached_kb = v;
            else if (std::sscanf(line, "Buffers: %llu kB", &v) == 1) buffers_kb = v;
        }
        std::fclose(f);
        if (!have_avail) avail_kb = free_kb + cached_kb + buffers_kb;  // очень старые ядра без MemAvailable
        m.total = static_cast<std::uint64_t>(total_kb) * 1024u;
        m.available = static_cast<std::uint64_t>(avail_kb) * 1024u;
        m.exact_available = (m.total != 0 && m.available != 0);
    }
#endif
    return m;
}

}  // namespace radix
