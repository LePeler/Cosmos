#include <utility>
#include <functional>
#include <random>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>

#include <Eigen/Dense>

#include <utils.h>


#ifndef COMBINEDLIKELIHOOD_H_INCLUDED
#define COMBINEDLIKELIHOOD_H_INCLUDED


template<int N, int D>
class CombinedLikelihood {

public:
    CombinedLikelihood(std::vector<std::shared_ptr<LikelihoodBase<N>>> likelihoods,
        std::function<bool(const Vector<N> &)> prior,
        std::function<Vector<D>(const Vector<D> &, double, const Vector<N> &)> model,
        std::function<Vector<D>(const Vector<N> &)> get_y0,
        std::function<double(const Vector<D> &, double, const Vector<N> &)> get_H_z,
        double dz_max = 0.0001)
        :
        likelihoods_(likelihoods),
        prior_(prior),
        model_(model),
        get_y0_(get_y0),
        get_H_z_(get_H_z)
        {

        // combine all redshifts and sort
        std::vector<double> z_H_tmp = {0};
        for (const std::shared_ptr<LikelihoodBase<N>> &likelihood : likelihoods_) {
            insert(z_H_tmp, likelihood->GetZ());
        }
        sort(z_H_tmp);

        // combine all redshifts at which D_C is needed and sort
        for (const std::shared_ptr<LikelihoodBase<N>> &likelihood : likelihoods_) {
            if (likelihood->NeedsD_C()) {
                insert(z_D_C_, likelihood->GetZ());
            }
        }
        sort(z_D_C_);

        if (z_D_C_.size() == 0 && D == 0) {
            z_H_ = z_H_tmp;
        }
        else {
            // insert filler values where distance too large
            for (size_t j = 1; j < z_H_tmp.size(); j++) {
                double dz_init = z_H_tmp[j] - z_H_tmp[j-1];

                unsigned int N_fills = static_cast<unsigned int>(dz_init/dz_max);
                double dz = dz_init/(N_fills+1);
                for (unsigned int k = 1; k <= N_fills; k++) {
                    z_H_.push_back(z_H_tmp[j-1] + k*dz);
                }
                z_H_.push_back(z_H_tmp[j]);
            }
        }

        // compute z interval widths
        dz_H_.push_back(z_H_[0]);
        for (size_t j = 1; j < z_H_.size(); j++) {
            dz_H_.push_back(z_H_[j] - z_H_[j-1]);
        }
    }

    ~CombinedLikelihood() = default;

    // evaluate the combined log-likelihood at a set of MCMC parameters
    double log_likelihood(const Vector<N> &params) {
        // check prior
        if (!prior_(params)) {
            return -INFINITY;
        }

        // compute H(z)
        std::vector<double> H_comp;
        // if there is nothing to solve, don't solve
        if (D == 0) {
            // get the H values
            for (size_t j = 0; j < z_H_.size(); j++) {
                H_comp.push_back(get_H_z_(Vector<D>{}, z_H_[j], params));
            }
        }
        // else use the RK4 solver
        else {
            // evaluate model at the given parameters
            std::function<Vector<D>(const Vector<D> &, double)> func = [this, params](const Vector<D> &y, double z) {
                return model_(y, z, params);
            };
            // RK4 numerical solver of D dimensional differential equations
            Vector<D> y0 = get_y0_(params);
            RK4<D> solver(func, y0, 0.0);
            // get the H values from the solver
            for (size_t j = 0; j < z_H_.size(); j++) {
                solver.MakeStep(dz_H_[j]);
                H_comp.push_back(get_H_z_(solver.GetCurrentValue(), z_H_[j], params));
            }
        }

        // integrate c/H to give D_C
        std::vector<double> D_C_comp;
        if (z_D_C_.size() > 0) {
            D_C_comp.push_back(z_H_[0] *PHYS_C/H_comp[0] + PHYS_C*integrate_inverse(z_H_, H_comp, z_H_[0], z_D_C_[0]));
            for (size_t j = 1; j < z_D_C_.size(); j++) {
                D_C_comp.push_back(D_C_comp.back() + PHYS_C*integrate_inverse(z_H_, H_comp, z_D_C_[j-1], z_D_C_[j]));
            }
        }

        // call the individual log likelihoods
        double result = 0.0;
        for (const std::shared_ptr<LikelihoodBase<N>> &likelihood : likelihoods_) {
            result += likelihood->log_likelihood(params, z_H_, H_comp, z_D_C_, D_C_comp);
        }

        // return the result
        return result;
    }


private:
    // the individual likelihoods
    std::vector<std::shared_ptr<LikelihoodBase<N>>> likelihoods_;
    // the prior to check whether the parameters are valid
    std::function<bool(const Vector<N> &)> prior_;

    // the cosmological model dy/dz = f(y,z, params)
    std::function<Vector<D>(const Vector<D> &, double, const Vector<N> &)> model_;
    // the starting point for the solver y(0)
    std::function<Vector<D>(const Vector<N> &)> get_y0_;
    // the function giving H(z) in terms of the solved model, the redshift and the MCMC parameters
    std::function<double(const Vector<D> &, double, const Vector<N> &)> get_H_z_;

    // the z values for which to compute H(z)
    std::vector<double> z_H_;
    // the intervals between these z values
    std::vector<double> dz_H_;
    // the z values for which to compute D_C(z)
    std::vector<double> z_D_C_;
};



#endif //COMBINEDLIKELIHOOD_H_INCLUDED

