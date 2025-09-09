#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <cmath>

#include <Eigen/Dense>
#include <cblas.h>


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



// multithreaded vector scalar multiplication using BLAS
template<int N>
double blas_dot(const Vector<N> &v1, const Vector<N> &v2) {
    if (v1.rows() != v2.rows()) {
        throw std::runtime_error("The vectors have different dimensions.");
    }

    // call BLAS ddot
    return cblas_ddot(
        v1.rows(),      // length of vectors
        v1.data(), 1,   // x pointer and stride (1 = contiguous)
        v2.data(), 1    // y pointer and stride
    );
}



// multithreaded vector squared norm using BLAS
template<int N>
double blas_squared_norm(const Vector<N> &v) {
    return blas_dot(v, v);
}


// multithreaded vector-matrix multiplication using BLAS
template<int N, int K>
Vector<N> blas_gemv(const NonSquareMatrix<N, K> &M, const Vector<K> &v) {
    if (M.cols() != v.rows()) {
        throw std::runtime_error("The matrix and vector dimensions don't match.");
    }

    Vector<N> result(M.rows());

    // call BLAS dgemv
    cblas_dgemv(
        CblasColMajor,      // Eigen uses column-major storage by default
        CblasNoTrans,       // M * v, not M^T * v
        M.rows(),           // number of rows of M
        M.cols(),           // number of cols of M
        1.0,                // alpha
        M.data(),           // pointer to M
        M.outerStride(),    // leading dimension (stride between columns)
        v.data(),           // pointer to v
        1,                  // increment for v (contiguous)
        0.0,                // beta (result = alpha*M*v + beta*result)
        result.data(),      // pointer to result
        1                   // increment for result
    );

    return result;
}


// multithreaded vector-upper-triangular-matrix multiplication using BLAS
template<int N>
Vector<N> blas_trmv_up(const Matrix<N> &L, const Vector<N> &v) {
    if (L.rows() != L.cols()) {
        throw std::runtime_error("The matrix is not square.");
    }
    if (L.cols() != v.rows()) {
        throw std::runtime_error("The matrix and vector dimensions don't match.");
    }

    Vector<N> result = v;

    // call BLAS dtrmv
    cblas_dtrmv(
        CblasColMajor,      // Eigen uses column-major storage by default
        CblasUpper,         // upper-triangular
        CblasNoTrans,       // M * v, not M^T * v
        CblasNonUnit,       // diagonal has real values (not all 1)
        L.rows(),           // matrix dimension
        L.data(),           // pointer to L
        L.rows(),           // leading dimension of L
        result.data(),      // pointer to result (which is also the input vector v)
        1                   // increment for result
    );

    return result;
}


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


// find the corresponding y vals to the values in x (x has to be sorted)
std::vector<double> find_corresponding(const std::vector<double> &X, const std::vector<double> &Y, const std::vector<double> &x) {
    if (X.size() != Y.size()) {
        throw std::runtime_error("X and Y must have the same length.");
    }

    std::vector<double> result(x.size(), NAN);

    size_t j = 0;
    for (size_t k = 0; k < x.size(); k++) {
        while (X[j] != x[k]) {
            j++;
            if (j >= X.size()) {
                break;
            }
        }
        result[k] = Y[j];
    }

    return result;
}


// interpolate linearly
double interpolate(const std::vector<double> &X, const std::vector<double> &Y, double x) {
    if (X.size() != Y.size()) {
        throw std::runtime_error("X and Y must have the same length.");
    }

    // if x is in X return the corresponding element from Y
    std::vector<double>::const_iterator it_x = std::find(X.begin(), X.end(), x);
    size_t j_x = std::distance(X.begin(), it_x);
    if (j_x != X.size()) {
        return Y[j_x];
    }

    // else do the usual interpolation
    std::vector<double>::const_iterator it_up = std::upper_bound(X.begin(), X.end(), x);
    size_t j_up = std::distance(X.begin(), it_up); // index of first element > x

    if (j_up == 0 || j_up == X.size()) {
        throw std::runtime_error("x is out of range.");
    }

    return Y[j_up-1] + (Y[j_up]-Y[j_up-1]) /(X[j_up]-X[j_up-1]) *(x-X[j_up-1]);
}


// interpolate semi-linearly in 2 dimensions
double interpolate2d(const std::vector<double> &X, const std::vector<double> &Y, const std::vector<std::vector<double>> &grid, double x, double y) {
    if (X.size() != grid.size()) {
        throw std::runtime_error("X and grid must have the same size.");
    }

    // do the usual interpolation (skip checking whether x,y are in X,Y as it is unlikely)
    std::vector<double>::const_iterator it_j_up = std::upper_bound(X.begin(), X.end(), x);
    size_t j_up = std::distance(X.begin(), it_j_up); // index of first element > x
    std::vector<double>::const_iterator it_k_up = std::upper_bound(Y.begin(), Y.end(), y);
    size_t k_up = std::distance(Y.begin(), it_k_up); // index of first element > y

    if (j_up == 0 || j_up == X.size()) {
        throw std::runtime_error("x is out of range.");
    }
    if (k_up == 0 || k_up == Y.size()) {
        throw std::runtime_error("y is out of range.");
    }

    double dx = X[j_up] - X[j_up-1];
    double dy = Y[k_up] - Y[k_up-1];
    double z00 = grid[j_up-1].at(k_up-1);
    return z00 + (grid[j_up].at(k_up-1)-z00) /dx *(x-X[j_up-1]) + (grid[j_up-1].at(k_up)-z00) /dy *(y-Y[k_up-1])
                + (grid[j_up].at(k_up)-grid[j_up].at(k_up-1)-grid[j_up-1].at(k_up)+z00) /dx /dy *(x-X[j_up-1]) *(y-Y[k_up-1]);
}


// numerical integration of the multiplicative inverse
double integrate_inverse(const std::vector<double> &X, const std::vector<double> &Y, double x0, double x1) {
    if (X.size() != Y.size()) {
        throw std::runtime_error("X and Y must have the same length.");
    }

    std::vector<double>::const_iterator it0 = std::lower_bound(X.begin(), X.end(), x0);
    std::vector<double>::const_iterator it1 = std::upper_bound(X.begin(), X.end(), x1);
    size_t j0 = std::distance(X.begin(), it0); // index of first element >= x0
    size_t j1 = std::distance(X.begin(), it1-1); // index of last element <= x1

    if (j0 > j1 || j1 >= X.size()) {
        throw std::runtime_error("The integration bounds (x0 and x1) are out of range.");
    }

    double result = 0.0;
    if (X[j0] != x0) {
        if (j0 == 0) {
            throw std::runtime_error("The integration bounds (x0 and x1) are out of range.");
        }
        double y0 = Y[j0-1] + (Y[j0]-Y[j0-1]) /(X[j0]-X[j0-1]) *(x0-X[j0-1]);
        result += (1/y0+1/Y[j0]) /2 *(X[j0]-x0);
    }
    if (X[j1] != x1) {
        if (j1 == X.size()-1) {
            throw std::runtime_error("The integration bounds (x0 and x1) are out of range.");
        }
        double y1 = Y[j1] + (Y[j1+1]-Y[j1]) /(X[j1+1]-X[j1]) *(x1-X[j1]);
        result += (1/Y[j1]+1/y1) /2 *(x1-X[j1]);
    }
    for (size_t j = j0; j < j1; j++) {
        result += (1/Y[j]+1/Y[j+1]) /2 *(X[j+1]-X[j]);
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

