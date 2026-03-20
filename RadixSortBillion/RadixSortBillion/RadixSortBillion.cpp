#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdint>

using namespace std;

// Проверка отсортированности
bool is_sorted(uint16_t* arr, size_t n) {
    for (size_t i = 1; i < n; i++) {
        if (arr[i - 1] > arr[i]) return false;
    }
    return true;
}

// Последовательный radix sort (2 прохода по 8 бит)
void radix_sort(uint16_t* arr, uint16_t* output, size_t n) {
    const int K = 256; // 2^8
    uint32_t count[K];

    // --- PASS 1: младший байт ---
    for (int i = 0; i < K; i++) count[i] = 0;

    for (size_t i = 0; i < n; i++) {
        uint8_t byte = arr[i] & 0xFF;
        count[byte]++;
    }

    for (int i = 1; i < K; i++) {
        count[i] += count[i - 1];
    }

    for (int i = (int)n - 1; i >= 0; i--) {
        uint8_t byte = arr[i] & 0xFF;
        output[--count[byte]] = arr[i];
    }

    // --- PASS 2: старший байт ---
    for (int i = 0; i < K; i++) count[i] = 0;

    for (size_t i = 0; i < n; i++) {
        uint8_t byte = (output[i] >> 8) & 0xFF;
        count[byte]++;
    }

    for (int i = 1; i < K; i++) {
        count[i] += count[i - 1];
    }

    for (int i = (int)n - 1; i >= 0; i--) {
        uint8_t byte = (output[i] >> 8) & 0xFF;
        arr[--count[byte]] = output[i];
    }
}

int main() {
    const char* filename = "input.txt";
    const char* output_file = "output.txt";

    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error while opening file\n";
        return 1;
    }

    // --- Считаем количество элементов ---
    size_t n = 0;
    uint16_t temp;
    while (file >> temp) n++;

    file.clear();
    file.seekg(0);

    // --- Выделение памяти ---
    uint16_t* arr = new uint16_t[n];
    uint16_t* output = new uint16_t[n];

    // --- Чтение данных ---
    for (size_t i = 0; i < n; i++) {
        file >> arr[i];
    }

    file.close();

    cout << "Element number: " << n << endl;

    // --- Таймер ---
    auto start = chrono::high_resolution_clock::now();

    radix_sort(arr, output, n);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "Sorting time: " << duration.count() << " seconds\n";

    // --- Проверка ---
    if (is_sorted(arr, n)) {
        cout << "Array is sorted\n";
    }
    else {
        cout << "Sorting error!\n";
    }

    ofstream out(output_file);
    if (!out.is_open()) {
        cout << "Output file creation error\n";
        return 1;
    }

    for (size_t i = 0; i < n; i++) {
        out << arr[i];
        if (i != n - 1) out << " ";
    }

    out.close();
    cout << "Output file created: " << output_file << endl;

    // --- Освобождение памяти ---
    delete[] arr;
    delete[] output;

    return 0;
}