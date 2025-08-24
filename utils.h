#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>

#include <Eigen/Dense>


#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED


template<int N>
using Vector = Eigen::Matrix<double, N, 1>;
template<int N>
using Matrix = Eigen::Matrix<double, N, N>;


void save_vector_as_txt(const std::vector<double> &vec, std::filesystem::path path, unsigned int save_every = 1) {
    // save a std::vector<T> as a .txt file, save only every skip values

    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error(("Could not open file: " + path.string()).c_str());
    }

    for (size_t i = 0; i < vec.size()/save_every; i++) {
        file << vec[i*save_every] << "\n";
    }

    file.close();
}


template<int N>
void save_vector_as_txt(const std::vector<Vector<N>> &vec, std::filesystem::path path, unsigned int save_every = 1) {
    // save a std::vector<Vector<N>> as a .txt file, save only every skip values

    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error(("Could not open file: " + path.string()).c_str());
    }

    for (size_t i = 0; i < vec.size()/save_every; i++) {
        for (int n = 0; n < N-1; n++) {
            file << vec[i*save_every][n] << ", ";
        }
        file << vec[i*save_every][N-1] << "\n";
    }

    file.close();
}



#endif //UTILS_H_INCLUDED

