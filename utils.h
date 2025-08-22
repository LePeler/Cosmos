#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>


template<typename T>
void save_vector_as_txt(const std::vector<T> &vec, std::filesystem::path path) {
    // save a std::vector<T> as a .txt file

    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error(("Could not open file: " + path.string()).c_str());
    }

    for (const T &elem : vec) {
        file << elem << "\n";
    }

    file.close();
}


template<typename T, size_t N>
void save_arrayvector_as_txt(const std::vector<std::array<T, N>> &vec, std::filesystem::path path) {
    // save a std::vector<std::array<T, N>> as a .txt file

    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error(("Could not open file: " + path.string()).c_str());
    }

    for (const std::array<T, N> &arr : vec) {
        for (size_t n = 0; n < N; n++) {
            file << arr[n] << ", ";
        }
        file << "\n";
    }

    file.close();
}


