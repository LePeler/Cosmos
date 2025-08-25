#include <functional>
#include <random>

#include <samplers/MCMC_base.h>
#include <utils.h>


#ifndef MCMC_H_INCLUDED
#define MCMC_H_INCLUDED


// affine invariant MCMC sampler
template<int N, unsigned int W>
class MCMC : public MCMC_base<N, W> {

public:
    // constructor
    MCMC(std::function<double(const Vector<N> &)> lnP, std::array<Vector<N>, W> init_states, double alpha, std::filesystem::path out_path, double beta = 2)
        :
        MCMC_base<N, W>(lnP, init_states, alpha, out_path),
        beta_(beta),
        dist0W2_(0, W-2)
    {}

    // destructor
    ~MCMC() = default;


private:
    // range constant for the state pdf
    double beta_;

    // uniform size_t distribution from 0 to W-2
    std::uniform_int_distribution<size_t> dist0W2_;

    // sample a new state for walker w and also return the associated pdf value
    std::pair<Vector<N>, double> SampleNewState(unsigned int w, std::mt19937 &randgen) override {

        // sample non-w walker for stretch move
        unsigned int idx = dist0W2_(randgen);
        if (idx >= w) {
            idx++;
        }

        // sample stretch factor
        double rand01 = this->dist01_(randgen);
        double stretch = ((beta_-1)*rand01 + 1) * ((beta_-1)*rand01 + 1) /beta_;

        return std::make_pair(
            this->states_[w] + (this->states_[idx] - this->states_[w]) * stretch,
            sqrt(beta_/stretch)/2/(beta_-1) /(W-1)
        );
    }
};



#endif //MCMC_H_INCLUDED

