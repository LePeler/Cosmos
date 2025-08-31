#include <utility>
#include <functional>
#include <random>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>

#include <Eigen/Dense>

#include <likelihoods/LikelihoodBase.h>
#include <utils.h>


#ifndef SN1A_H_INCLUDED
#define SN1A_H_INCLUDED


template<int N>
class SN1a : public LikelihoodBase<N> {
    using LikelihoodBase<N>::z_;

public:
    SN1a(fs::path data_dir, size_t M_idx)
    :
    LikelihoodBase<N>(true),
    M_idx_(M_idx)
    {
        // check that M_idx is in range of N
        if (M_idx >= N) {
            throw std::invalid_argument("M_idx (" + std::to_string(M_idx_) + ") must be less than N (" + std::to_string(N) + ").");
        }

        // read in SN1a data
        std::vector<double> m_tmp;
        for (const std::vector<double> &datapoint : read_txt(data_dir / fs::path("SN1a_z_m.txt"), ',')) {
            z_.push_back(datapoint.at(0));
            m_tmp.push_back(datapoint.at(1));
        }
        m_ = convert_vector(m_tmp);
        std::vector<double> M_calib_tmp;
        for (const std::vector<double> &datapoint : read_txt(data_dir / fs::path("SN1a_M_calib.txt"), ',')) {
            M_calib_tmp.push_back(datapoint.at(0));
        }
        M_calib_ = convert_vector(M_calib_tmp);

        std::vector<std::vector<double>> inv_cov_signal_tmp = read_txt(data_dir / fs::path("SN1a_inv_cov_signal.txt"), ',');
        inv_cov_signal_ = convert_matrix(inv_cov_signal_tmp);
        std::vector<std::vector<double>> inv_cov_mix_tmp = read_txt(data_dir / fs::path("SN1a_inv_cov_mix.txt"), ',');
        inv_cov_mix_ = convert_nonsquarematrix(inv_cov_mix_tmp);
        std::vector<std::vector<double>> inv_cov_calib_tmp = read_txt(data_dir / fs::path("SN1a_inv_cov_calib.txt"), ',');
        inv_cov_calib_ = convert_matrix(inv_cov_calib_tmp);
    }

    ~SN1a() = default;

    // log likelihood for the SN1a data
    double log_likelihood(const Vector<N> &params,
                            const std::vector<double> &z_H,
                            const std::vector<double> &H_comp,
                            const std::vector<double> &z_D_C,
                            const std::vector<double> &D_C_comp) const override {

        // get the absolute magnitude of SN1as from the MCMC parameters
        double M = params(M_idx_);

        // interpolate the comoving distance and calculate the predicted magnitudes
        Vector<-1> m(z_.size());
        for (int j = 0; j < z_.size(); j++) {
            m(j) = 5 * log10((1.0 + z_[j]) * interpolate(z_D_C, D_C_comp, z_[j])) + 25.0 + M;
        }
        // calculate the magnitude residuals
        Vector<-1> signal_residuals = m - m_;

        // calculate the absolue magnitude calibration residuals
        Vector<-1> calib_residuals(M_calib_.size());
        for (int j = 0; j < M_calib_.size(); j++) {
            calib_residuals(j) = M - M_calib_(j);
        }

        // return the gaussian log likelihood
        return -(inv_cov_signal_*signal_residuals).dot(signal_residuals) /2
               -(inv_cov_mix_*signal_residuals).dot(calib_residuals)
               -(inv_cov_calib_*calib_residuals).dot(calib_residuals) /2;
    }


private:
    // SN1a data
    Vector<-1> m_;
    Vector<-1> M_calib_;
    Matrix<-1> inv_cov_signal_;
    NonSquareMatrix<-1,-1> inv_cov_mix_;
    Matrix<-1> inv_cov_calib_;

    // index of the SN1a absolute magnitude in the MCMC parameters
    size_t M_idx_;
};



#endif //SN1A_H_INCLUDED

