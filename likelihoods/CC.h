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


#ifndef CC_H_INCLUDED
#define CC_H_INCLUDED


template<int N>
class CC : public LikelihoodBase<N> {
    using LikelihoodBase<N>::z_;

public:
    CC(fs::path data_dir)
    :
    LikelihoodBase<N>(false)
    {
        // read in CC data
        std::vector<double> H_tmp;
        for (const std::vector<double> &datapoint : read_txt(data_dir / fs::path("CC_z_H.txt"), ',')) {
            z_.push_back(datapoint.at(0));
            H_tmp.push_back(datapoint.at(1));
        }
        H_ = convert_vector(H_tmp);

        std::vector<std::vector<double>> inv_cov_tmp = read_txt(data_dir / fs::path("CC_inv_cov.txt"), ',');
        inv_cov_ = convert_matrix(inv_cov_tmp);

        LT_ = inv_cov_.llt().matrixU();
    }

    ~CC() = default;

    // log likelihood for the CC data
    double log_likelihood(const Vector<N> &params,
                            const std::vector<double> &z_H,
                            const std::vector<double> &H_comp,
                            const std::vector<double> &z_D_C,
                            const std::vector<double> &D_C_comp) const override {

        // interpolate the predicted Hubble parameters
        std::vector<double> H_vals = find_corresponding(z_H, H_comp, z_);
        Vector<-1> H_pred(z_.size());
        for (int j = 0; j < z_.size(); j++) {
            H_pred(j) = H_vals[j];
        }
        // calculate the Hubble parameter residuals
        Vector<-1> residuals = H_pred - H_;

        // return the gaussian log likelihood
        return -(LT_.triangularView<Eigen::Upper>()*residuals).squaredNorm() /2;
    }


private:
    // CC data
    Vector<-1> H_;
    Matrix<-1> inv_cov_;
    Matrix<-1> LT_;
};



#endif //CC_H_INCLUDED

