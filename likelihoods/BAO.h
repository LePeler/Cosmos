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


#ifndef BAO_H_INCLUDED
#define BAO_H_INCLUDED


template<int N>
class BAO : public LikelihoodBase<N> {
    using LikelihoodBase<N>::z_;

public:
    BAO(fs::path data_dir, std::function<double(const Vector<N> &)> get_R_drag)
    :
    LikelihoodBase<N>(true),
    get_R_drag_(get_R_drag)
    {
        // read in BAO data
        std::vector<double> angle_1_tmp;
        for (const std::vector<double> &datapoint : read_txt(data_dir / fs::path("BAO_z_angle_1.txt"), ',')) {
            z_1_.push_back(datapoint.at(0));
            angle_1_tmp.push_back(datapoint.at(1));
        }
        angle_1_ = convert_vector(angle_1_tmp);
        std::vector<double> angle_2_tmp;
        for (const std::vector<double> &datapoint : read_txt(data_dir / fs::path("BAO_z_angle_2.txt"), ',')) {
            z_2_.push_back(datapoint.at(0));
            angle_2_tmp.push_back(datapoint.at(1));
        }
        angle_2_ = convert_vector(angle_2_tmp);
        std::vector<double> angle_3_tmp;
        for (const std::vector<double> &datapoint : read_txt(data_dir / fs::path("BAO_z_angle_3.txt"), ',')) {
            z_3_.push_back(datapoint.at(0));
            angle_3_tmp.push_back(datapoint.at(1));
        }
        angle_3_ = convert_vector(angle_3_tmp);

        // concatenate z_1, z_2 and z_3 and sort
        insert(z_, z_1_);
        insert(z_, z_2_);
        insert(z_, z_3_);
        sort(z_);

        std::vector<std::vector<double>> inv_cov_1_tmp = read_txt(data_dir / fs::path("BAO_inv_cov_1.txt"), ',');
        inv_cov_1_ = convert_matrix(inv_cov_1_tmp);
        std::vector<std::vector<double>> inv_cov_mix12_tmp = read_txt(data_dir / fs::path("BAO_inv_cov_mix12.txt"), ',');
        inv_cov_mix12_ = convert_nonsquarematrix(inv_cov_mix12_tmp);
        std::vector<std::vector<double>> inv_cov_2_tmp = read_txt(data_dir / fs::path("BAO_inv_cov_2.txt"), ',');
        inv_cov_2_ = convert_matrix(inv_cov_2_tmp);
        std::vector<std::vector<double>> inv_cov_mix23_tmp = read_txt(data_dir / fs::path("BAO_inv_cov_mix23.txt"), ',');
        inv_cov_mix23_ = convert_nonsquarematrix(inv_cov_mix23_tmp);
        std::vector<std::vector<double>> inv_cov_3_tmp = read_txt(data_dir / fs::path("BAO_inv_cov_3.txt"), ',');
        inv_cov_3_ = convert_matrix(inv_cov_3_tmp);
        std::vector<std::vector<double>> inv_cov_mix31_tmp = read_txt(data_dir / fs::path("BAO_inv_cov_mix31.txt"), ',');
        inv_cov_mix31_ = convert_nonsquarematrix(inv_cov_mix31_tmp);

        LT_1_ = inv_cov_1_.llt().matrixU();
        LT_2_ = inv_cov_2_.llt().matrixU();
        LT_3_ = inv_cov_3_.llt().matrixU();
    }

    ~BAO() = default;

    // log likelihood for the BAO data
    double log_likelihood(const Vector<N> &params,
                            const std::vector<double> &z_H,
                            const std::vector<double> &H_comp,
                            const std::vector<double> &z_D_C,
                            const std::vector<double> &D_C_comp) const override {

        // calculate radius at the end of the drag epoch
        double r_s = get_R_drag_(params);

        // calculate predicted angles for type 1 measurements
        std::vector<double> H_vals_1 = find_corresponding(z_H, H_comp, z_1_);
        std::vector<double> D_C_vals_1 = find_corresponding(z_D_C, D_C_comp, z_1_);
        Vector<-1> angle_1(z_1_.size());
        for (int j = 0; j < z_1_.size(); j++) {
            angle_1(j) = cbrt(z_1_[j] *D_C_vals_1[j] *D_C_vals_1[j] *PHYS_C/H_vals_1[j]) /r_s;
        }
        // calculate predicted angles for type 2 measurements
        std::vector<double> D_C_vals_2 = find_corresponding(z_D_C, D_C_comp, z_2_);
        Vector<-1> angle_2(z_2_.size());
        for (int j = 0; j < z_2_.size(); j++) {
            angle_2(j) = D_C_vals_2[j] /r_s;
        }
        // calculate predicted angles for type 3 measurements
        std::vector<double> H_vals_3 = find_corresponding(z_H, H_comp, z_3_);
        Vector<-1> angle_3(z_3_.size());
        for (int j = 0; j < z_3_.size(); j++) {
            angle_3(j) = PHYS_C/H_vals_3[j] /r_s;
        }

        // calculate all angle residuals
        Vector<-1> residuals_1 = angle_1 - angle_1_;
        Vector<-1> residuals_2 = angle_2 - angle_2_;
        Vector<-1> residuals_3 = angle_3 - angle_3_;

        // return the gaussian log likelihood
        return -(LT_1_.triangularView<Eigen::Upper>()*residuals_1).squaredNorm() /2
               -(inv_cov_mix12_*residuals_2).dot(residuals_1)
               -(LT_2_.triangularView<Eigen::Upper>()*residuals_2).squaredNorm() /2
               -(inv_cov_mix23_*residuals_3).dot(residuals_2)
               -(LT_3_.triangularView<Eigen::Upper>()*residuals_3).squaredNorm() /2
               -(inv_cov_mix31_*residuals_1).dot(residuals_3);
    }


private:
    // BAO data
    std::vector<double> z_1_;
    Vector<-1> angle_1_;
    std::vector<double> z_2_;
    Vector<-1> angle_2_;
    std::vector<double> z_3_;
    Vector<-1> angle_3_;
    Matrix<-1> inv_cov_1_;
    NonSquareMatrix<-1,-1> inv_cov_mix12_;
    Matrix<-1> inv_cov_2_;
    NonSquareMatrix<-1,-1> inv_cov_mix23_;
    Matrix<-1> inv_cov_3_;
    NonSquareMatrix<-1,-1> inv_cov_mix31_;
    Matrix<-1> LT_1_;
    Matrix<-1> LT_2_;
    Matrix<-1> LT_3_;

    // R_drag as a function of the parameters
    std::function<double(const Vector<N> &)> get_R_drag_;
};



#endif //BAO_H_INCLUDED

