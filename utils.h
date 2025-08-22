#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>


template<typename T>
void save_vector_as_txt(const std::vector<T> &vec, std::filesystem::path path, unsigned int skip) {
    // save a std::vector<T> as a .txt file, save only every skip values

    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error(("Could not open file: " + path.string()).c_str());
    }

    for (size_t i = 0; i < vec.size()/skip; i++) {
        file << vec[i*skip] << "\n";
    }

    file.close();
}


template<typename T, size_t N>
void save_arrayvector_as_txt(const std::vector<std::array<T, N>> &vec, std::filesystem::path path, unsigned int skip) {
    // save a std::vector<std::array<T, N>> as a .txt file, save only every skip values

    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error(("Could not open file: " + path.string()).c_str());
    }

    for (size_t i = 0; i < vec.size()/skip; i++) {
        for (size_t n = 0; n < N-1; n++) {
            file << vec[i*skip][n] << ", ";
        }
        file << vec[i*skip][N-1] << "\n";
    }

    file.close();
}


