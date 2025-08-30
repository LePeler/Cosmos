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


#ifndef LIKELIHOODBASE_H_INCLUDED
#define LIKELIHOODBASE_H_INCLUDED


template<int N>
class LikelihoodBase {

public:
    LikelihoodBase(bool needs_D_C) : needs_D_C_(needs_D_C) {}

    ~LikelihoodBase() = default;

    // does the likelihood depend on the comoving distance?
    bool NeedsD_C() const {
        return needs_D_C_;
    }

    // get the z values in the data
    std::vector<double> GetZ() const {
        return z_;
    }

    // log likelihood
    virtual double log_likelihood(const Vector<N> &params,
                                    const std::vector<double> &z_H,
                                    const std::vector<double> &H_comp,
                                    const std::vector<double> &z_D_C,
                                    const std::vector<double> &D_C_comp) const = 0;


protected:
    // the z values in the data
    std::vector<double> z_;
    // whether the likelihood needs the comoving distance to be computed
    bool needs_D_C_;
};



#endif //LIKELIHOODBASE_H_INCLUDED

