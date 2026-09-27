// radix_sort.cpp
//
// Последовательная цифровая сортировка (radix sort) для unsigned short.
// Так как диапазон значений k = 2^16 = 65536 совпадает с диапазоном самого
// типа, весь массив сортируется ЗА ОДИН ПРОХОД сортировкой подсчётом:
//   1) count[v]      — сколько раз встретилось значение v
//   2) count[v] (pfx)— префиксная сумма -> позиция конца корзины v
//   3) обратный проход по входу, запись в output с конца корзины (для
//      устойчивости; на данном этапе, с одним разрядом, устойчивость сама
//      по себе значения не имеет, но так реализация сразу совместима со
//      схемой, которая в дальнейшем ляжет в основу параллельной версии)
//
// Отдельно есть быстрый генератор тестовых данных и бинарный I/O — при
// N ~ 10^9 текстовый файл (один int в строке) читается/пишется на порядок
// медленнее и занимает на диске в разы больше места, чем бинарный дамп
// uint16_t, поэтому бинарный формат используется по умолчанию.
//
// Запуск без аргументов (например, кнопкой Run/▷ в VS Code) открывает
// простое текстовое меню — ничего печатать в терминале руками не нужно,
// только отвечать на вопросы. Использование через командную строку с
// аргументами (generate/sort/show/demo/bench) тоже работает как раньше —
// это нужно для больших замеров (например, generate 1000000000 ...).
//
// Сборка вручную (если понадобится):
//   g++ -O3 -march=native -std=c++17 -o radix_sort radix_sort.cpp

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

using u16 = uint16_t;
using u32 = uint32_t;

constexpr size_t RADIX    = 1u << 16;  // 65536 возможных значений unsigned short
constexpr size_t IO_CHUNK = 1u << 20;  // элементов на один fread/fwrite вызов

// ---------------------------------------------------------------- таймер --
struct Timer {
    using clock = std::chrono::steady_clock;
    clock::time_point t0 = clock::now();
    double elapsed_s() const {
        return std::chrono::duration<double>(clock::now() - t0).count();
    }
};

// ------------------------------------------------------ генерация данных --
// Заполняет data случайными unsigned short на [0, 65535].
// Берём mt19937 и режем каждое 32-битное слово на два 16-битных значения —
// это вдвое быстрее, чем uniform_int_distribution<u16> на каждый элемент,
// и не даёт смещения, так как диапазон значений и так равен полной ширине
// типа.
void fill_random(std::vector<u16>& data, uint32_t seed) {
    std::mt19937 gen(seed);
    size_t n = data.size();
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        u32 w = gen();
        data[i]     = static_cast<u16>(w & 0xFFFFu);
        data[i + 1] = static_cast<u16>(w >> 16);
    }
    if (i < n) {
        data[i] = static_cast<u16>(gen() & 0xFFFFu);
    }
}

// ---------------------------------------------------- цифровая сортировка --
// Сортирует in -> out (out будет пересоздан нужного размера).
void radix_sort_u16(const std::vector<u16>& in, std::vector<u16>& out) {
    size_t n = in.size();
    out.assign(n, 0);

    std::vector<u32> count(RADIX, 0);
    for (size_t i = 0; i < n; ++i) {
        ++count[in[i]];
    }

    // Префиксные суммы: count[v] становится индексом ПОСЛЕ последней ячейки
    // корзины v (правая граница полуинтервала).
    for (size_t v = 1; v < RADIX; ++v) {
        count[v] += count[v - 1];
    }

    // Идём по входу с конца и пишем в конец соответствующей корзины —
    // так одинаковые значения сохраняют исходный относительный порядок.
    for (size_t i = n; i-- > 0;) {
        u16 v = in[i];
        out[--count[v]] = v;
    }
}

// ------------------------------------------------------------- проверка --
bool is_sorted(const std::vector<u16>& data) {
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i - 1] > data[i]) return false;
    }
    return true;
}

// -------------------------------------------------------- бинарный I/O --
// Формат: 4 байта магической метки "RS16" + 8 байт количества элементов
// (size_t) + сами uint16_t. Магическая метка нужна не для алгоритма, а
// чтобы сразу дать понятную ошибку, если файл перепутан с текстовым
// (например, забыли --text) — вместо того чтобы молча прочитать мусор.
constexpr char BIN_MAGIC[4] = {'R', 'S', '1', '6'};

void write_binary(const std::string& path, const std::vector<u16>& data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("не удалось открыть файл на запись: " + path);
    std::fwrite(BIN_MAGIC, 1, sizeof(BIN_MAGIC), f);
    size_t n = data.size();
    std::fwrite(&n, sizeof(n), 1, f);
    const u16* p = data.data();
    size_t remaining = n;
    while (remaining > 0) {
        size_t chunk = std::min(remaining, IO_CHUNK);
        std::fwrite(p, sizeof(u16), chunk, f);
        p += chunk;
        remaining -= chunk;
    }
    std::fclose(f);
}

std::vector<u16> read_binary(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("не удалось открыть файл на чтение: " + path);
    char magic[4];
    if (std::fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        std::memcmp(magic, BIN_MAGIC, sizeof(magic)) != 0) {
        std::fclose(f);
        throw std::runtime_error(
            "'" + path + "' не похож на бинарный файл этой программы. "
            "Похоже, файл был создан или ожидается с флагом --text — "
            "добавьте --text и к generate/sort, и при повторном чтении этого файла.");
    }
    size_t n = 0;
    if (std::fread(&n, sizeof(n), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("повреждённый заголовок файла: " + path);
    }
    std::vector<u16> data(n);
    u16* p = data.data();
    size_t remaining = n;
    while (remaining > 0) {
        size_t chunk = std::min(remaining, IO_CHUNK);
        size_t got = std::fread(p, sizeof(u16), chunk, f);
        if (got != chunk) {
            std::fclose(f);
            throw std::runtime_error("файл обрезан: " + path);
        }
        p += chunk;
        remaining -= chunk;
    }
    std::fclose(f);
    return data;
}

// ------------------------------------------------- текстовый I/O (мал. N) --
// Один int в строке. Только для небольших N — удобно для ручной проверки,
// глазами посмотреть на файл; для сотен миллионов и выше используйте
// бинарный формат (см. выше).
void write_text(const std::string& path, const std::vector<u16>& data) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("не удалось открыть файл на запись: " + path);
    std::string buf;
    buf.reserve(1u << 20);
    char tmp[8];
    for (u16 v : data) {
        int len = std::snprintf(tmp, sizeof(tmp), "%u\n", v);
        buf.append(tmp, static_cast<size_t>(len));
        if (buf.size() > (1u << 20)) {
            std::fwrite(buf.data(), 1, buf.size(), f);
            buf.clear();
        }
    }
    if (!buf.empty()) std::fwrite(buf.data(), 1, buf.size(), f);
    std::fclose(f);
}

std::vector<u16> read_text(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) throw std::runtime_error("не удалось открыть файл на чтение: " + path);
    std::vector<u16> data;
    data.reserve(1u << 16);
    int v;
    while (std::fscanf(f, "%d", &v) == 1) {
        data.push_back(static_cast<u16>(v));
    }

    // Файл не пустой, а чисел не нашли -> почти наверняка это бинарный
    // файл, который читают без --text (частая путаница с флагами).
    if (data.empty()) {
        std::fseek(f, 0, SEEK_END);
        long size = std::ftell(f);
        if (size > 0) {
            std::fclose(f);
            throw std::runtime_error(
                "'" + path + "' не удалось прочитать как текст (ни одного числа не найдено), "
                "хотя файл не пустой. Похоже, это бинарный файл — уберите --text для этого файла.");
        }
    }

    std::fclose(f);
    return data;
}

// ------------------------------------------------ человеко-читаемый вывод --
// Печатает значения в одну строку через пробел, без обрезки — используется
// только для demo с небольшим N, где массив целиком помещается на экран.
void print_values(const std::vector<u16>& data) {
    for (size_t i = 0; i < data.size(); ++i) {
        std::printf("%u%s", data[i], (i + 1 < data.size()) ? " " : "\n");
    }
}

// Печатает первые head и последние tail значений — для больших файлов,
// которые целиком на экран не влезут и не нужны.
void print_preview(const std::vector<u16>& data, size_t head, size_t tail) {
    size_t n = data.size();
    if (n <= head + tail) {
        print_values(data);
        return;
    }
    for (size_t i = 0; i < head; ++i) std::printf("%u ", data[i]);
    std::printf("... (%zu значений пропущено) ... ", n - head - tail);
    for (size_t i = n - tail; i < n; ++i) std::printf("%u%s", data[i], (i + 1 < n) ? " " : "");
    std::printf("\n");
}

// -------------------------------------------------- интерактивное меню ----
// Режим для запуска прямо из VS Code кнопкой "Run" — без единой команды в
// терминале. Просто отвечаете на вопросы (Enter = взять значение по
// умолчанию в квадратных скобках).

size_t ask_size_t(const std::string& label, size_t def) {
    std::printf("%s [%zu]: ", label.c_str(), def);
    std::string line;
    std::getline(std::cin, line);
    if (line.empty()) return def;
    try {
        return std::stoull(line);
    } catch (...) {
        std::printf("Не похоже на число, беру значение по умолчанию: %zu\n", def);
        return def;
    }
}

std::string ask_string(const std::string& label, const std::string& def) {
    std::printf("%s [%s]: ", label.c_str(), def.c_str());
    std::string line;
    std::getline(std::cin, line);
    return line.empty() ? def : line;
}

bool ask_yes_no(const std::string& label, bool def) {
    std::printf("%s [%s]: ", label.c_str(), def ? "Y/n" : "y/N");
    std::string line;
    std::getline(std::cin, line);
    if (line.empty()) return def;
    char c = static_cast<char>(std::tolower(static_cast<unsigned char>(line[0])));
    if (c == 'y') return true;
    if (c == 'n') return false;
    // "да"/"нет" на русском тоже принимаем (сравниваем целиком, а не по
    // байту — русские буквы в UTF-8 многобайтовые).
    if (line == "да" || line == "Да" || line == "ДА") return true;
    if (line == "нет" || line == "Нет" || line == "НЕТ") return false;
    return def;
}

void wait_enter() {
    std::printf("\nНажмите Enter, чтобы вернуться в меню...");
    std::string dummy;
    std::getline(std::cin, dummy);
}

void interactive_menu() {
    std::printf("=== Radix Sort (unsigned short, один проход) ===\n");
    while (true) {
        std::printf(
            "\nЧто сделать?\n"
            "  1 - Демо: \"было / стало\" на маленьком массиве\n"
            "  2 - Сгенерировать файл со случайными числами\n"
            "  3 - Отсортировать файл\n"
            "  4 - Посмотреть содержимое файла\n"
            "  5 - Замер скорости (без файлов)\n"
            "  0 - Выход\n"
            "Выберите пункт: ");
        std::string choice;
        std::getline(std::cin, choice);

        try {
            if (choice == "1") {
                size_t n = ask_size_t("Сколько чисел сгенерировать для демо", 20);
                std::vector<u16> data(n);
                fill_random(data, 12345u);
                std::printf("\nБыло (%zu чисел, случайный порядок):\n", n);
                print_values(data);

                std::vector<u16> sorted;
                radix_sort_u16(data, sorted);
                std::printf("\nСтало (%zu чисел, по возрастанию):\n", n);
                print_values(sorted);

                std::printf("\nКорректность: %s\n",
                             is_sorted(sorted) ? "OK (массив действительно отсортирован)"
                                                : "ОШИБКА (массив НЕ отсортирован)");

            } else if (choice == "2") {
                size_t n = ask_size_t("Сколько чисел сгенерировать", 1000000);
                std::string path = ask_string("Имя файла", "data.bin");
                bool text = ask_yes_no("Текстовый формат (человекочитаемый, но медленнее/больше на диске)", false);

                std::vector<u16> data(n);
                Timer t;
                fill_random(data, 12345u);
                std::printf("Сгенерировано %zu значений за %.3f с\n", n, t.elapsed_s());

                Timer t2;
                if (text) write_text(path, data); else write_binary(path, data);
                std::printf("Записано в %s (%s) за %.3f с\n",
                             path.c_str(), text ? "текст" : "бинарный", t2.elapsed_s());

            } else if (choice == "3") {
                std::string inpath = ask_string("Входной файл", "data.bin");
                std::string outpath = ask_string("Выходной файл", "sorted.bin");
                bool text = ask_yes_no("Текстовый формат (должен совпадать с тем, каким создавался входной файл!)", false);

                Timer t;
                std::vector<u16> data = text ? read_text(inpath) : read_binary(inpath);
                std::printf("Прочитано %zu значений за %.3f с\n", data.size(), t.elapsed_s());

                std::vector<u16> sorted;
                Timer t2;
                radix_sort_u16(data, sorted);
                double sort_s = t2.elapsed_s();
                std::printf("Отсортировано за %.3f с (%.1f млн знач/с)\n",
                             sort_s, sort_s > 0 ? data.size() / sort_s / 1e6 : 0.0);
                std::printf("Корректность: %s\n", is_sorted(sorted) ? "OK" : "ОШИБКА");

                Timer t3;
                if (text) write_text(outpath, sorted); else write_binary(outpath, sorted);
                std::printf("Записано в %s за %.3f с\n", outpath.c_str(), t3.elapsed_s());

            } else if (choice == "4") {
                std::string path = ask_string("Какой файл посмотреть", "data.bin");
                bool text = ask_yes_no("Текстовый формат (должен совпадать с тем, каким создавался файл!)", false);
                size_t count = ask_size_t("Сколько значений показать с начала и с конца", 10);

                std::vector<u16> data = text ? read_text(path) : read_binary(path);
                std::printf("Файл: %s (%s), всего значений: %zu\n",
                             path.c_str(), text ? "текст" : "бинарный", data.size());
                print_preview(data, count, count);
                std::printf("Отсортирован: %s\n", is_sorted(data) ? "да" : "нет");

            } else if (choice == "5") {
                size_t n = ask_size_t("Сколько чисел для замера скорости", 10000000);

                std::vector<u16> data(n);
                Timer tgen;
                fill_random(data, 12345u);
                double gen_s = tgen.elapsed_s();
                std::printf("Генерация: %zu значений за %.3f с (%.1f млн знач/с)\n",
                             n, gen_s, gen_s > 0 ? n / gen_s / 1e6 : 0.0);

                std::vector<u16> sorted;
                Timer tsort;
                radix_sort_u16(data, sorted);
                double sort_s = tsort.elapsed_s();
                std::printf("Сортировка: %zu значений за %.3f с (%.1f млн знач/с)\n",
                             n, sort_s, sort_s > 0 ? n / sort_s / 1e6 : 0.0);
                std::printf("Корректность: %s\n", is_sorted(sorted) ? "OK" : "ОШИБКА");

            } else if (choice == "0") {
                std::printf("Пока!\n");
                return;

            } else {
                std::printf("Не понял выбор, попробуйте ещё раз.\n");
                continue;
            }
        } catch (const std::exception& e) {
            std::printf("Ошибка: %s\n", e.what());
        }

        wait_enter();
    }
}

// --------------------------------------------------------------- CLI ------
void usage(const char* argv0) {
    std::fprintf(stderr,
        "Использование:\n"
        "  %s generate <N> <file> [--text]         сгенерировать файл со случайными числами\n"
        "  %s sort <in> <out> [--text]              отсортировать файл\n"
        "  %s show <file> [--text] [--count K]      посмотреть первые/последние K значений файла\n"
        "  %s demo [N]                              наглядно: было/стало на N числах (по умолч. 20)\n"
        "  %s bench <N>                              замер скорости на N числах, без файлов\n",
        argv0, argv0, argv0, argv0, argv0);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    // На Windows консоль (cmd.exe/PowerShell) по умолчанию открывается не в
    // UTF-8, а исходники и все строки в программе — в UTF-8, отсюда
    // "кракозябры" вместо кириллицы. Переключаем кодовую страницу консоли
    // на UTF-8 для ввода и вывода.
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    if (argc < 2) {
        // Запуск без аргументов (в т.ч. кнопкой Run в VS Code) -> меню.
        interactive_menu();
        return 0;
    }
    std::string cmd = argv[1];

    try {
        if (cmd == "generate") {
            if (argc < 4) { usage(argv[0]); return 1; }
            size_t n = std::stoull(argv[2]);
            std::string path = argv[3];
            bool text = (argc > 4 && std::string(argv[4]) == "--text");

            std::vector<u16> data(n);
            Timer t;
            fill_random(data, 12345u);
            std::fprintf(stderr, "сгенерировано %zu значений за %.3f с\n", n, t.elapsed_s());

            Timer t2;
            if (text) write_text(path, data); else write_binary(path, data);
            std::fprintf(stderr, "записано %s (%s) за %.3f с\n",
                         path.c_str(), text ? "текст" : "бинарный", t2.elapsed_s());

        } else if (cmd == "sort") {
            if (argc < 4) { usage(argv[0]); return 1; }
            std::string inpath = argv[2];
            std::string outpath = argv[3];
            bool text = (argc > 4 && std::string(argv[4]) == "--text");

            Timer t;
            std::vector<u16> data = text ? read_text(inpath) : read_binary(inpath);
            std::fprintf(stderr, "прочитано %zu значений за %.3f с\n", data.size(), t.elapsed_s());

            std::vector<u16> sorted;
            Timer t2;
            radix_sort_u16(data, sorted);
            double sort_s = t2.elapsed_s();
            std::fprintf(stderr, "отсортировано за %.3f с (%.1f млн знач/с)\n",
                         sort_s, sort_s > 0 ? data.size() / sort_s / 1e6 : 0.0);
            std::fprintf(stderr, "корректность: %s\n", is_sorted(sorted) ? "OK" : "ОШИБКА");

            Timer t3;
            if (text) write_text(outpath, sorted); else write_binary(outpath, sorted);
            std::fprintf(stderr, "записано %s за %.3f с\n", outpath.c_str(), t3.elapsed_s());

        } else if (cmd == "demo") {
            size_t n = 20;
            if (argc > 2) n = std::stoull(argv[2]);

            std::vector<u16> data(n);
            fill_random(data, 12345u);

            std::printf("Было (%zu чисел, случайный порядок):\n", n);
            print_values(data);

            std::vector<u16> sorted;
            radix_sort_u16(data, sorted);

            std::printf("\nСтало (%zu чисел, по возрастанию):\n", n);
            print_values(sorted);

            std::printf("\nКорректность: %s\n", is_sorted(sorted) ? "OK (массив действительно отсортирован)"
                                                                    : "ОШИБКА (массив НЕ отсортирован)");

        } else if (cmd == "show") {
            if (argc < 3) { usage(argv[0]); return 1; }
            std::string path = argv[2];
            bool text = false;
            size_t count = 10;
            for (int i = 3; i < argc; ++i) {
                std::string a = argv[i];
                if (a == "--text") text = true;
                else if (a == "--count" && i + 1 < argc) count = std::stoull(argv[++i]);
            }

            std::vector<u16> data = text ? read_text(path) : read_binary(path);
            std::printf("Файл: %s (%s), всего значений: %zu\n",
                        path.c_str(), text ? "текст" : "бинарный", data.size());
            print_preview(data, count, count);
            std::printf("Отсортирован: %s\n", is_sorted(data) ? "да" : "нет");

        } else if (cmd == "bench") {
            if (argc < 3) { usage(argv[0]); return 1; }
            size_t n = std::stoull(argv[2]);

            std::vector<u16> data(n);
            Timer tgen;
            fill_random(data, 12345u);
            double gen_s = tgen.elapsed_s();
            std::fprintf(stderr, "генерация: %zu значений за %.3f с (%.1f млн знач/с)\n",
                         n, gen_s, gen_s > 0 ? n / gen_s / 1e6 : 0.0);

            std::vector<u16> sorted;
            Timer tsort;
            radix_sort_u16(data, sorted);
            double sort_s = tsort.elapsed_s();
            std::fprintf(stderr, "сортировка: %zu значений за %.3f с (%.1f млн знач/с)\n",
                         n, sort_s, sort_s > 0 ? n / sort_s / 1e6 : 0.0);

            std::fprintf(stderr, "корректность: %s\n", is_sorted(sorted) ? "OK" : "ОШИБКА");

        } else {
            usage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ошибка: %s\n", e.what());
        return 1;
    }

    return 0;
}
