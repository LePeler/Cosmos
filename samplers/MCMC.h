#include <vector>
#include <functional>
#include <random>

#include <State.h>


// affine invariant MCMC sampler
template<size_t N, size_t W>
class MCMC {

public:
    // constructor
    MCMC(std::function<double(State<N>)> lnP, std::array<State<N>, W> init_states, double alpha, double beta = 2)
        :
        lnP_(lnP),
        states_(init_states),
        alpha_(alpha),
        beta_(beta),
        randgen_(std::random_device{}()),
        dist01_(0.0, 1.0),
        dist0W2_(0, W-2)
    {}

    // destructor
    ~MCMC() = default;

    // get the current walker states
    std::array<std::array<double, N>, W> GetWalkerStates() const {
        std::array<std::array<double, N>, W> result;
        for (size_t w = 0; w < W; w++) {
            result[w] = states_[w];
        }
        return result;
    }

    // get the entire sample
    std::vector<std::array<double, N>> GetSample() const {
        std::vector<std::array<double, N>> result;
        for (State<N> state : sample_) {
            result.push_back(state);
        }
        return result;
    }

    // make an iteration of the MCMC algorithm
    void MakeIter() {
        states_ = ComputeIter();

        sample_.insert(states_.begin(), states_.end(), sample_.end());
    }

    // call a loop over MakeIter for K steps
    void MakeIters(unsigned int K) {
        for (unsigned int k = 0; k < K; k++) {
            MakeIter();
        }
    }


private:
    // log-likelyhood function
    std::function<double(State<N>)> lnP_;

    // current states of the walkers
    std::array<State<N>, W> states_;
    // logprobs at the current states
    std::array<double, W> logprobs_;
    // the entire sample
    std::vector<State<N>> sample_;

    // tuning constant for acceptance probabilities
    double alpha_;
    // range constant for the state pdf
    double beta_;

    // random number generator (Mersenne Twister)
    std::mt19937 randgen_;
    // uniform double distribution from 0 to 1
    std::uniform_real_distribution<double> dist01_;
    // uniform size_t distribution from 0 to W-2
    std::uniform_int_distribution<size_t> dist0W2_;

    // pdf for the stretch factor
    double StretchP(double stretch) const;
    // sample the stretch factor from above pdf
    double SampleStretch();

    // compute an iteration of the MCMC algorithm
    std::array<State<N>, W> ComputeIter();
};


