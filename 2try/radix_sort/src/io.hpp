// io.hpp — чтение и запись массивов uint16_t.
//
// Форматы:
//   text   — десятичные целые 0..65535, по одному на строку (основной формат). Читатель терпим к
//            любым пробельным разделителям, пустым строкам, CRLF, UTF-8 BOM и отсутствию
//            перевода строки в конце; писатель всегда пишет '\n' (без CRLF на любой ОС).
//   binary — «сырые» uint16_t в порядке байт платформы (little-endian на x86/ARM), 2 байта на элемент.
//            Файл на 10^9 элементов занимает 2 ГБ и читается на скорости диска.
//
// Время ввода-вывода измеряется вызывающим кодом отдельно от времени сортировки.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "radix_sort.hpp"

namespace radix {

class IoError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class FileFormat { Auto, Text, Binary };

const char* to_string(FileFormat f) noexcept;
bool parse_file_format(const std::string& name, FileFormat& out);

// Auto: расширение ".bin" -> binary, иначе text.
FileFormat resolve_format(FileFormat requested, const std::string& path);

struct ReadInfo {
    std::uint64_t file_bytes = 0;
    FileFormat format = FileFormat::Text;
};

// Обратный вызов прогресса: (прочитано байт, всего байт). Вызывается не чаще раза в ~2 секунды,
// поэтому для небольших файлов не вызывается вообще.
using ProgressFn = std::function<void(std::uint64_t done, std::uint64_t total)>;

// Читает массив из файла. Бросает IoError с понятным сообщением (в т. ч. номер строки для text):
// файл не существует / не открывается, посторонний символ, значение вне 0..65535, отрицательное
// число, размер binary-файла не кратен 2, ошибка чтения.
// Пустой файл допустим (результат — пустой массив).
ReadInfo read_keys(const std::string& path, FileFormat format, std::vector<Key>& out,
                   const ProgressFn& progress = ProgressFn());

// Оценка числа ключей в файле БЕЗ его полного чтения: для binary — точно (размер / 2), для text —
// по средней длине записи в трёх фрагментах файла (начало, середина, конец) с запасом ~3 %.
// Нужна, чтобы заранее выделить память без переаллокаций и проверить, хватит ли ОЗУ.
// Для небольших файлов возвращает безопасную верхнюю границу (размер / 2 + 1). 0 — размер неизвестен.
std::uint64_t estimate_key_count(const std::string& path, FileFormat format);

// Потоковый писатель: позволяет записывать огромные массивы порциями, не держа их в памяти целиком.
class KeyWriter {
public:
    KeyWriter(const std::string& path, FileFormat format);  // бросает IoError, если файл не открыть
    ~KeyWriter();
    KeyWriter(const KeyWriter&) = delete;
    KeyWriter& operator=(const KeyWriter&) = delete;

    void write(const Key* data, std::size_t n);  // бросает IoError при ошибке записи
    void close();                                // сбрасывает буфер и закрывает файл; бросает IoError
    std::uint64_t bytes_written() const noexcept { return bytes_; }

private:
    void flush_buffer();

    std::FILE* file_ = nullptr;
    FileFormat format_;
    std::string path_;
    std::vector<char> buffer_;  // буфер форматирования (только для text)
    std::size_t used_ = 0;
    std::uint64_t bytes_ = 0;
};

// Запись массива целиком; возвращает число записанных байт.
std::uint64_t write_keys(const std::string& path, FileFormat format, const std::vector<Key>& data);

}  // namespace radix
