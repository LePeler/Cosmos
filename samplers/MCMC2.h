#include <functional>
#include <random>

#include <samplers/MCMC_base.h>
#include <utils.h>


#ifndef MCMC2_H_INCLUDED
#define MCMC2_H_INCLUDED


// affine invariant modification of the Metropolis-Hastings MCMC sampler
template<int N, unsigned int W>
class MCMC2 : public MCMC_base<N, W> {

public:
    // constructor
    MCMC2(std::function<double(const Vector<N> &)> lnP, std::array<Vector<N>, W> init_states, double alpha, std::filesystem::path out_path, double beta = 1)
        :
        MCMC_base<N, W>(lnP, init_states, alpha, out_path),
        beta_(beta),
        Mu_(Vector<N>::Zero()),
        Var_(Matrix<N>::Identity()),
        L_(Matrix<N>::Identity()),
        Zinv_(1.0),
        distN01_(0.0, 1.0)
    {}

    // destructor
    ~MCMC2() = default;


private:
    // range constant for the state pdf
    double beta_;

    // mean of the current states
    Vector<N> Mu_;
    // variance matrix of the current states
    Matrix<N> Var_;
    // Cholesky decomposition of the scaled variance matrix
    Matrix<N> L_;
    // the normalization for the new state pdf
    double Zinv_;

    // normal distribution with mu=0 and sigma=1
    std::normal_distribution<double> distN01_;

    // update the Mu_, Var_ and L_ members
    void UpdateInternals() override {
        Mu_ = this->GetStateMean();
        Var_ = this->GetStateVarianceMatrix(Mu_);
        L_ = beta_ * (Var_.llt().matrixL().toDenseMatrix());
        Zinv_ = 1/pow(2*M_PI, N/2)/L_.determinant();
    }

    // sample a new state for walker w and also return the associated pdf value
    std::pair<Vector<N>, double> SampleNewState(unsigned int w, std::mt19937 &randgen) override {

        // sample a vector of standard normal distributed values
        Vector<N> z;
        for (unsigned int n = 0; n < N; n++) {
            z[n] = distN01_(randgen);
        }

        return std::make_pair(
            Mu_ + (L_.template triangularView<Eigen::Lower>()) * z,
            Zinv_ * exp(-z.dot(z) /beta_/beta_ /2)
        );
    }
};



#endif //MCMC2_H_INCLUDED

