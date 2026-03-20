#include <iostream>
#include <fstream>
#include <random>

using namespace std;

int main() {
    const size_t N = 1000000000; // Количество элементов для генерации
    const char* filename = "input.txt";

    ofstream file(filename);
    if (!file.is_open()) {
        cout << "Open file error\n";
        return 1;
    }

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<uint16_t> dist(0, 65535);

    for (size_t i = 0; i < N; i++) {
        file << dist(gen);
        if (i != N - 1) file << " ";
    }

    file.close();

    cout << "File generated: " << filename << endl;
    return 0;
}