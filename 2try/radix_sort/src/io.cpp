// io.cpp — быстрый буферизованный ввод-вывод массивов uint16_t.
//
// Чтение текста устроено как потоковый конечный автомат по блокам в 4 МиБ: никаких строк,
// std::getline, istringstream и operator>>; на входе нет «объектов на строку», расход памяти —
// один буфер чтения + сам массив ключей. Состояние автомата (текущее число, номер строки)
// переносится между блоками, поэтому число, разрезанное границей блока, разбирается верно.
#include "io.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <system_error>

namespace radix {

namespace {

constexpr std::size_t kReadChunk = std::size_t{1} << 22;     // 4 МиБ
constexpr std::size_t kWriteBuffer = std::size_t{1} << 20;   // 1 МиБ
constexpr unsigned kZero = static_cast<unsigned>('0');
constexpr std::uint32_t kMaxKey = static_cast<std::uint32_t>(kKeyRange - 1);  // 65535

struct FileCloser {
    void operator()(std::FILE* f) const noexcept {
        if (f) std::fclose(f);
    }
};
using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

std::string os_error() { return std::strerror(errno); }

// --- Разбор текста --------------------------------------------------------

struct ParseState {
    std::uint32_t value = 0;     // накопленное значение текущего числа
    bool in_number = false;      // мы внутри числа (видели хотя бы одну цифру)
    std::uint64_t line = 1;      // номер текущей строки (для сообщений об ошибках)
};

[[noreturn]] void parse_error(const std::string& path, std::uint64_t line, const std::string& what) {
    throw IoError(path + ": line " + std::to_string(line) + ": " + what);
}

[[noreturn]] void bad_char(const std::string& path, std::uint64_t line, unsigned char c) {
    if (c == '-') {
        parse_error(path, line, "negative values are not allowed (expected an integer in 0..65535)");
    }
    char hex[8];
    std::snprintf(hex, sizeof hex, "0x%02X", static_cast<unsigned>(c));
    if (std::isprint(c)) {
        parse_error(path, line, std::string("unexpected character '") + static_cast<char>(c) + "' (" + hex +
                                    "); expected an integer in 0..65535");
    }
    parse_error(path, line, std::string("unexpected byte ") + hex + "; expected an integer in 0..65535");
}

// Разбор одного блока. Допустимые символы: цифры и пробельные разделители.
// Значение проверяется на каждой цифре: value <= 65535 до умножения, значит value*10+9 не переполняет uint32.
void parse_chunk(const unsigned char* p, const unsigned char* e, ParseState& st, std::vector<Key>& out,
                 const std::string& path) {
    std::uint32_t value = st.value;
    bool in_number = st.in_number;
    std::uint64_t line = st.line;
    for (; p != e; ++p) {
        const unsigned char c = *p;
        const unsigned digit = static_cast<unsigned>(c) - kZero;  // для не-цифр получится большое число
        if (digit <= 9u) {
            value = value * 10u + digit;
            if (value > kMaxKey) parse_error(path, line, "value out of range (allowed range is 0..65535)");
            in_number = true;
        } else if (c == '\n' || c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f') {
            if (in_number) {
                out.push_back(static_cast<Key>(value));
                value = 0;
                in_number = false;
            }
            if (c == '\n') ++line;
        } else {
            bad_char(path, line, c);
        }
    }
    st.value = value;
    st.in_number = in_number;
    st.line = line;
}

// Вызывает обратный вызов прогресса не чаще раза в 2 секунды (проверка стоит одного чтения часов на блок).
class ProgressThrottle {
public:
    explicit ProgressThrottle(const ProgressFn& fn) : fn_(fn), last_(std::chrono::steady_clock::now()) {}
    void tick(std::uint64_t done, std::uint64_t total) {
        if (!fn_) return;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_ >= std::chrono::seconds(2)) {
            last_ = now;
            fn_(done, total);
        }
    }

private:
    const ProgressFn& fn_;
    std::chrono::steady_clock::time_point last_;
};

void read_text(std::FILE* f, const std::string& path, std::uint64_t file_bytes, std::vector<Key>& out,
               const ProgressFn& progress) {
    std::unique_ptr<unsigned char[]> chunk(new unsigned char[kReadChunk]);
    ParseState st;
    ProgressThrottle throttle(progress);
    std::uint64_t bytes_done = 0;
    bool first = true;
    for (;;) {
        const std::size_t got = std::fread(chunk.get(), 1, kReadChunk, f);
        if (got == 0) break;
        const unsigned char* p = chunk.get();
        const unsigned char* const e = p + got;
        if (first) {
            if (got >= 3 && p[0] == 0xEF && p[1] == 0xBB && p[2] == 0xBF) p += 3;  // UTF-8 BOM
            first = false;
        }
        parse_chunk(p, e, st, out, path);
        bytes_done += got;
        throttle.tick(bytes_done, file_bytes);
    }
    if (std::ferror(f)) throw IoError("read error on '" + path + "': " + os_error());
    if (st.in_number) out.push_back(static_cast<Key>(st.value));  // последняя строка без '\n'
}

void read_binary(std::FILE* f, const std::string& path, std::uint64_t file_bytes, std::vector<Key>& out,
                 const ProgressFn& progress) {
    if (file_bytes % sizeof(Key) != 0) {
        throw IoError(path + ": binary file size (" + std::to_string(file_bytes) +
                      " bytes) is not a multiple of 2 (sizeof(uint16_t))");
    }
    const std::uint64_t n = file_bytes / sizeof(Key);
    if (n > out.max_size()) throw IoError(path + ": file is too large for this platform");
    out.resize(static_cast<std::size_t>(n));
    // Читаем порциями по 64 МиБ: так можно показывать прогресс на файлах в гигабайты.
    constexpr std::size_t kPiece = std::size_t{1} << 25;  // 32M элементов = 64 МиБ
    ProgressThrottle throttle(progress);
    std::size_t done = 0;
    while (done < out.size()) {
        const std::size_t want = std::min(kPiece, out.size() - done);
        const std::size_t got = std::fread(out.data() + done, sizeof(Key), want, f);
        done += got;
        if (got != want) {
            throw IoError("read error on '" + path + "': expected " + std::to_string(out.size()) +
                          " elements, got " + std::to_string(done));
        }
        throttle.tick(static_cast<std::uint64_t>(done) * sizeof(Key), file_bytes);
    }
}

}  // namespace

const char* to_string(FileFormat f) noexcept {
    switch (f) {
        case FileFormat::Auto: return "auto";
        case FileFormat::Text: return "text";
        case FileFormat::Binary: return "binary";
    }
    return "?";
}

bool parse_file_format(const std::string& name, FileFormat& out) {
    std::string s;
    for (const char ch : name) s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    if (s == "auto") out = FileFormat::Auto;
    else if (s == "text" || s == "txt") out = FileFormat::Text;
    else if (s == "binary" || s == "bin") out = FileFormat::Binary;
    else return false;
    return true;
}

FileFormat resolve_format(FileFormat requested, const std::string& path) {
    if (requested != FileFormat::Auto) return requested;
    const std::size_t dot = path.rfind('.');
    if (dot != std::string::npos) {
        std::string ext;
        for (std::size_t i = dot; i < path.size(); ++i) {
            ext.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(path[i]))));
        }
        if (ext == ".bin") return FileFormat::Binary;
    }
    return FileFormat::Text;
}

ReadInfo read_keys(const std::string& path, FileFormat format, std::vector<Key>& out, const ProgressFn& progress) {
    ReadInfo info;
    info.format = resolve_format(format, path);

    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) throw IoError("input file '" + path + "' does not exist");
    if (std::filesystem::is_directory(path, ec)) throw IoError("'" + path + "' is a directory, not a file");

    FilePtr file(std::fopen(path.c_str(), "rb"));  // "rb": никакой трансляции \r\n — разбор делаем сами
    if (!file) throw IoError("cannot open '" + path + "' for reading: " + os_error());

    std::error_code size_ec;
    const std::uintmax_t size = std::filesystem::file_size(path, size_ec);
    info.file_bytes = size_ec ? 0 : static_cast<std::uint64_t>(size);

    out.clear();
    if (info.format == FileFormat::Binary) {
        if (size_ec) throw IoError("cannot determine the size of binary file '" + path + "': " + size_ec.message());
        read_binary(file.get(), path, info.file_bytes, out, progress);
    } else {
        // Память под результат выделяется ОДИН раз по оценке числа ключей: вектор не растёт «удвоениями»
        // (иначе при 10^9 элементов он копировал бы гигабайты и на миг занимал бы в полтора раза больше).
        // Если оценка окажется занижена, вектор просто вырастет сам — это влияет только на скорость.
        const std::uint64_t estimate = estimate_key_count(path, info.format);
        if (estimate > out.capacity()) {
            out.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(estimate, out.max_size())));
        }
        read_text(file.get(), path, info.file_bytes, out, progress);
    }
    return info;
}

std::uint64_t estimate_key_count(const std::string& path, FileFormat format) {
    const FileFormat fmt = resolve_format(format, path);
    std::error_code ec;
    const std::uintmax_t size_raw = std::filesystem::file_size(path, ec);
    if (ec) return 0;
    const std::uint64_t size = static_cast<std::uint64_t>(size_raw);
    if (fmt == FileFormat::Binary) return size / sizeof(Key);

    constexpr std::uint64_t kSample = std::uint64_t{1} << 18;  // 256 КиБ на фрагмент
    const std::uint64_t upper_bound = size / 2 + 1;            // минимум 2 байта на ключ: цифра + разделитель
    if (size < 4 * kSample) return upper_bound;                // небольшой файл: безопасная верхняя граница

    std::ifstream in(path, std::ios::binary);
    if (!in) return upper_bound;
    std::vector<char> buf(static_cast<std::size_t>(kSample));
    std::uint64_t sampled_bytes = 0, tokens = 0;
    // Три фрагмента (начало, середина, конец) нужны, чтобы тренды не искажали оценку: например,
    // у отсортированного массива в начале файла короткие числа, а в конце — пятизначные.
    const std::uint64_t offsets[3] = {0, (size - kSample) / 2, size - kSample};
    for (const std::uint64_t off : offsets) {
        in.clear();
        in.seekg(static_cast<std::streamoff>(off));
        in.read(buf.data(), static_cast<std::streamsize>(kSample));
        const std::size_t got = static_cast<std::size_t>(in.gcount());
        bool in_number = false;
        for (std::size_t i = 0; i < got; ++i) {
            const bool digit = (buf[i] >= '0' && buf[i] <= '9');
            if (in_number && !digit) ++tokens;  // число закончилось (разделитель любого вида)
            in_number = digit;
        }
        if (in_number) ++tokens;
        sampled_bytes += got;
    }
    if (tokens == 0 || sampled_bytes == 0) return upper_bound;
    const double bytes_per_key = static_cast<double>(sampled_bytes) / static_cast<double>(tokens);
    const double estimate = static_cast<double>(size) / bytes_per_key * 1.03 + 1024.0;
    return std::min<std::uint64_t>(static_cast<std::uint64_t>(estimate), upper_bound);
}

// --- Запись -----------------------------------------------------------------

KeyWriter::KeyWriter(const std::string& path, FileFormat format)
    : format_(resolve_format(format, path)), path_(path) {
    file_ = std::fopen(path.c_str(), "wb");  // "wb": всегда '\n', без CRLF
    if (!file_) throw IoError("cannot open '" + path + "' for writing: " + os_error());
    if (format_ == FileFormat::Text) buffer_.resize(kWriteBuffer);
}

KeyWriter::~KeyWriter() {
    if (file_) std::fclose(file_);  // ошибки при закрытии в деструкторе игнорируем; используйте close()
}

void KeyWriter::flush_buffer() {
    if (used_ == 0) return;
    if (std::fwrite(buffer_.data(), 1, used_, file_) != used_) {
        throw IoError("write error on '" + path_ + "': " + os_error());
    }
    bytes_ += used_;
    used_ = 0;
}

void KeyWriter::write(const Key* data, std::size_t n) {
    if (!file_) throw IoError("write to a closed file '" + path_ + "'");
    if (format_ == FileFormat::Binary) {
        if (n != 0 && std::fwrite(data, sizeof(Key), n, file_) != n) {
            throw IoError("write error on '" + path_ + "': " + os_error());
        }
        bytes_ += static_cast<std::uint64_t>(n) * sizeof(Key);
        return;
    }
    char* const buf = buffer_.data();
    const std::size_t cap = buffer_.size();
    std::size_t used = used_;
    for (std::size_t i = 0; i < n; ++i) {
        if (cap - used < 8) {  // запись занимает не более 6 байт (5 цифр + '\n')
            used_ = used;
            flush_buffer();
            used = 0;
        }
        unsigned v = data[i];
        char digits[5];
        int len = 0;
        do {
            digits[len++] = static_cast<char>(kZero + v % 10u);
            v /= 10u;
        } while (v != 0);
        while (len > 0) buf[used++] = digits[--len];
        buf[used++] = '\n';
    }
    used_ = used;
}

void KeyWriter::close() {
    if (!file_) return;
    if (format_ == FileFormat::Text) flush_buffer();
    std::FILE* f = file_;
    file_ = nullptr;
    if (std::fclose(f) != 0) throw IoError("error while closing '" + path_ + "': " + os_error());
}

std::uint64_t write_keys(const std::string& path, FileFormat format, const std::vector<Key>& data) {
    KeyWriter w(path, format);
    w.write(data.data(), data.size());
    w.close();
    return w.bytes_written();
}

}  // namespace radix
