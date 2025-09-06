#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <cmath>

#include <Eigen/Dense>


#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED


namespace fs = std::filesystem;

template<int N>
using Vector = Eigen::Matrix<double, N, 1>;
template<int N>
using Matrix = Eigen::Matrix<double, N, N>;
template<int N, int K>
using NonSquareMatrix = Eigen::Matrix<double, N, K>;


// speed of light in km/s
const double PHYS_C = 299792.458;


// get the mean of the vectors inside the vector
template<int N>
Vector<N> get_mean(const std::vector<Vector<N>> &vecs) {
    Vector<N> result = Vector<N>::Zero();
    for (size_t j = 0; j < vecs.size(); j++) {
    result += vecs[j];
    }

    return result /vecs.size();
}


// get the mean of the vectors inside the array
template<int N, size_t W>
Vector<N> get_mean(const std::array<Vector<N>, W> &vecs) {
    Vector<N> result = Vector<N>::Zero();
    for (size_t w = 0; w < W; w++) {
    result += vecs[w];
    }

    return result /W;
}


// get the scalar lag-k covariance of the vectors in the vector
template<int N>
double get_lag_k_covariance(const std::vector<Vector<N>> &vecs, size_t k, const Vector<N> &mean) {
    double result = 0.0;
    for (size_t j = 0; j < vecs.size()-k; j++) {
        result += (vecs[j]-mean).dot(vecs[j+k]-mean);
    }
    result /= (vecs.size()-1);

    return result;
}


// get the scalar variance of the vectors in the vector (lag-0 scalar covariance)
template<int N>
double get_variance(const std::vector<Vector<N>> &vecs, const Vector<N> &mean) {
    return get_lag_k_covariance(vecs, 0, mean);
}


template<typename T>
size_t get_index(T val, const std::vector<T> &vec) {
    // get the index of val in vec, if elem is not in vec vec.size() is returned
    return std::distance(vec.begin(), std::find(vec.begin(), vec.end(), val));
}


// Inverse CDF (Quantile) of the standard normal distribution
// Based on Peter J. Acklam's algorithm (2003)
double probit(double p) {
    // Coefficients in rational approximations
    static const double a[] = {
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00
    };

    static const double b[] = {
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01
    };

    static const double c[] = {
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00
    };

    static const double d[] = {
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00
    };

    // Define break-points
    const double plow  = 0.02425;
    const double phigh = 1 - plow;

    double q, r;

    if (p < plow) {
        // Rational approximation for lower region
        q = std::sqrt(-2 * std::log(p));
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1);
    }
    if (p > phigh) {
        // Rational approximation for upper region
        q = std::sqrt(-2 * std::log(1 - p));
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1);
    }

    // Rational approximation for central region
    q = p - 0.5;
    r = q * q;
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q /
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1);
}


void save_vector_as_txt(const std::vector<double> &vec, fs::path path, unsigned int save_every = 1) {
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
void save_vector_as_txt(const std::vector<Vector<N>> &vec, fs::path path, unsigned int save_every = 1) {
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


// read a txt file into a std::vector<std::vector<double>>
std::vector<std::vector<double>> read_txt(fs::path path, char delimiter) {
    // open file
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error(("Could not open file: " + path.string()).c_str());
    }

    std::vector<std::vector<double>> ret_vec;
    std::string line;
    std::vector<double> row;
    std::string value;

    // read all lines
    while (std::getline(file, line)) {
        if (line[0] == '#') {
            continue;
        }

        std::stringstream ss(line);

        // read all values
        while (std::getline(ss, value, delimiter)) {
            row.push_back(std::stod(value));
        }

        ret_vec.push_back(row);
        row.clear();
    }


    // close the file
    file.close();

    return ret_vec;
}


// wrapper around std::vector<>::insert
template<typename T>
void insert(std::vector<T> &insert_into, const std::vector<T> &to_insert) {
    insert_into.insert(insert_into.end(), to_insert.begin(), to_insert.end());
}


// wrapper around std::sort
template<typename T>
void sort(std::vector<T> &to_sort) {
    std::sort(to_sort.begin(), to_sort.end());
}


// interpolate linearly
double interpolate(const std::vector<double> &X, const std::vector<double> &Y, double x) {
    if (X.size() != Y.size()) {
        throw std::runtime_error(("X and Y must be of same size, but have " + std::to_string(X.size()) + " and " + std::to_string(Y.size()) + ".").c_str());
    }

    // if x is in X return the corresponding element from Y
    for (size_t j = 0; j < X.size(); j++) {
        if (X[j] == x) {
            return Y[j];
        }
    }

    // else do the usual interpolation
    for (size_t j = 0; j < X.size()-1; j++) {
        if (X[j] <= x && x < X[j+1]) {
            return Y[j] + (Y[j+1]-Y[j]) /(X[j+1]-X[j]) *(x-X[j]);
        }
    }

    // x is not between min(X) and max(X) so cannot interpolate
    throw std::invalid_argument("x is outside of the interpolatable range.");
}


// interpolate semi-linearly in 2 dimensions
double interpolate2d(const std::vector<double> &X, const std::vector<double> &Y, const std::vector<std::vector<double>> &grid, double x, double y) {
    if (X.size() != grid.size()) {
        throw std::runtime_error(("X and grid must be of same size, but have " + std::to_string(X.size()) + " and " + std::to_string(grid.size()) + ".").c_str());
    }

    // do the usual interpolation (skip checking whether x,y are in X,Y as it is unlikely)
    for (size_t j = 0; j < X.size()-1; j++) {
        if (X[j] <= x && x < X[j+1]) {
            for (size_t k = 0; k < Y.size()-1; k++) {
                if (Y[k] <= y && y < Y[k+1]) {
                    double dx = X[j+1] - X[j];
                    double dy = Y[k+1] - Y[k];
                    double z00 = grid[j].at(k);
                    return z00 + (grid[j+1].at(k)-z00) /dx *(x-X[j]) + (grid[j].at(k+1)-z00) /dy *(y-Y[k])
                               + (grid[j+1].at(k+1)-grid[j+1].at(k)-grid[j].at(k+1)+z00) /dx /dy *(x-X[j]) *(y-Y[k]);
                }
            }
        }
    }

    // (x,y) is not in the range spanned by X and Y so cannot interpolate
    throw std::invalid_argument("(x,y) is outside of the interpolatable range.");
}


// numerical integration of the multiplicative inverse
double integrate_inverse(const std::vector<double> &X, const std::vector<double> &Y, double x0, double x1) {
    if (X.size() != Y.size()) {
        throw std::runtime_error(("X and Y must be of same length, but have " + std::to_string(X.size()) + " and " + std::to_string(Y.size()) + ".").c_str());
    }

    double result = 0.0;
    for (size_t j = 0; j < X.size()-1; j++) {
        if (X[j+1] <= x0) {
            continue;
        }
        else if (X[j] >= x1) {
            break;
        }

        else if (x0 <= X[j] && X[j+1] <= x1) {
            result += (1/Y[j]+1/Y[j+1]) /2 *(X[j+1]-X[j]);
        }
        else if (X[j] < x0 && x0 < X[j+1]) {
            double y0 = Y[j] + (Y[j+1]-Y[j]) /(X[j+1]-X[j]) *(x0-X[j]);
            result += (1/y0+1/Y[j+1]) /2 *(X[j+1]-x0);
        }
        else if (X[j] < x1 && x1 < X[j+1]) {
            double y1 = Y[j] + (Y[j+1]-Y[j]) /(X[j+1]-X[j]) *(x1-X[j]);
            result += (1/Y[j]+1/y1) /2 *(x1-X[j]);
        }
    }

    return result;
}


// convert a std::vector<double> to a Vector<-1>
Vector<-1> convert_vector(const std::vector<double> &vec) {
    Vector<-1> ret_vec;
    ret_vec.resize(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        ret_vec(i) = vec[i];
    }

    return ret_vec;
}


// convert a std::vector<std::vector<double>> to a Matrix<-1>
Matrix<-1> convert_matrix(const std::vector<std::vector<double>> &mat) {
    Matrix<-1> ret_mat;
    ret_mat.resize(mat.size(), mat.size());
    for (size_t i = 0; i < mat.size(); i++) {
        for (size_t j = 0; j < mat.size(); j++) {
            ret_mat(i,j) = mat[i].at(j);
        }
    }

    return ret_mat;
}


// convert a std::vector<std::vector<double>> to a NonSquareMatrix<-1,-1>
NonSquareMatrix<-1,-1> convert_nonsquarematrix(const std::vector<std::vector<double>> &mat) {
    NonSquareMatrix<-1,-1> ret_mat;
    if (mat.size() == 0) {
        ret_mat.resize(0, 0);
        return ret_mat;
    }
    ret_mat.resize(mat.size(), mat[0].size());
    for (size_t i = 0; i < mat.size(); i++) {
        for (size_t j = 0; j < mat[0].size(); j++) {
            ret_mat(i,j) = mat[i].at(j);
        }
    }

    return ret_mat;
}


#endif //UTILS_H_INCLUDED

