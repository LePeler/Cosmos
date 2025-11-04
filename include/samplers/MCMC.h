#include <functional>
#include <random>

#include <samplers/MCMC_base.h>
#include <utils.h>


#ifndef MCMC_H_INCLUDED
#define MCMC_H_INCLUDED

/*
WARNING: This sampler does not work, but is still in here in case someone will fix it in the future!
*/


// affine invariant MCMC sampler (Goodman and Weare)
template<int N, unsigned int W>
class MCMC : public MCMC_base<N, W> {

public:
    // constructor
    MCMC(int proc, int num_procs, std::function<double(const Vector<N> &)> lnP, std::array<Vector<N>, W> init_states, double beta = 2)
        :
        MCMC_base<N, W>(proc, num_procs, lnP, init_states),
        beta_(beta)
    {
        throw std::logic_error("This sampler does not work, but is still in here in case someone will fix it in the future! \n Please use MCMC2 instead.");
    }

    // destructor
    ~MCMC() = default;


private:
    // range constant for the state pdf
    double beta_;

    // sample a new state for walker w and also return the associated state prob ratio (old / new)
    std::pair<Vector<N>, double> SampleNewState(unsigned int w, std::mt19937 &randgen) const override {

        // each thread gets its own uniform 0-to-(W-2) distribution and uniform 0-to-1 distribution
        thread_local static std::uniform_int_distribution<size_t> dist0W2(0, W-2);
        thread_local static std::uniform_real_distribution<double> dist01(0.0, 1.0);

        // sample non-w walker for stretch move
        unsigned int idx = dist0W2(randgen);
        if (idx >= w) {
            idx++;
        }

        // sample stretch factor
        double rand01 = dist01(randgen);
        double stretch = ((beta_-1)*rand01 + 1) * ((beta_-1)*rand01 + 1) /beta_;

        return std::make_pair(
            this->states_[w] + (this->states_[idx] - this->states_[w]) * stretch,
            1.0
        );
    }
};



#endif //MCMC_H_INCLUDED

