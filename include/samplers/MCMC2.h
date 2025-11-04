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
    MCMC2(int proc, int num_procs, std::function<double(const Vector<N> &)> lnP, std::array<Vector<N>, W> init_states, double beta = 1.5)
        :
        MCMC_base<N, W>(proc, num_procs, lnP, init_states),
        beta_(beta),
        Mu_(Vector<N>::Zero()),
        Var_(Matrix<N>::Identity()),
        L_(Matrix<N>::Identity())
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
    Eigen::LLT<Matrix<N>> L_;

    // update the Mu_, Var_ and L_ members
    void UpdateInternals() override {
        Mu_ = this->GetStateMean();
        Var_ = this->GetStateVariance(Mu_);
        L_ = Var_.llt();
    }

    // sample a new state for walker w and also return the associated state prob ratio (old / new)
    std::pair<Vector<N>, double> SampleNewState(unsigned int w, std::mt19937 &randgen) const override {

        // old (i.e. current) chi2
        double old_chi2 = L_.matrixL().solve(this->states_[w] - Mu_).squaredNorm();

        // each thread gets its own standard normal distribution
        thread_local static std::normal_distribution<double> distN01(0.0, 1.0);

        // sample a vector of standard normal distributed values
        Vector<N> z;
        for (unsigned int n = 0; n < N; n++) {
            z(n) = distN01(randgen);
        }

        return std::make_pair(
            Mu_ + L_.matrixL() * z * beta_,
            exp((z.squaredNorm() - old_chi2 /beta_/beta_) /2)
        );
    }
};



#endif //MCMC2_H_INCLUDED

