#include <vector>
#include <functional>
#include <random>

#include <utils.h>


#ifndef MCMC_H_INCLUDED
#define MCMC_H_INCLUDED


// affine invariant MCMC sampler
template<int N, unsigned int W>
class MCMC {

public:
    // constructor
    MCMC(std::function<double(const Vector<N> &)> lnP, std::array<Vector<N>, W> init_states, double alpha, double beta = 2)
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
    std::array<Vector<N>, W> GetWalkerStates() const {
        std::array<Vector<N>, W> result;
        for (unsigned int w = 0; w < W; w++) {
            result[w] = states_[w];
        }
        return result;
    }

    // get the entire sample
    std::vector<Vector<N>> GetSample() const {
        return sample_;
    }

    // get the sample mean
    Vector<N> GetSampleMean() const {
        Vector<N> result;
        for (const Vector<N> &state : sample_) {
            result += state;
        }
        result /= sample_.size();

        return result;
    }

    // get the sample scalar variance (the trace of the covariance matrix)
    double GetSampleVariance() const {
        double result = 0;
        Vector<N> mean = GetSampleMean();
        for (const Vector<N> &state : sample_) {
            result += (state-mean)*(state-mean);
        }
        result /= sample_.size();

        return result;
    }

    // get the sample scalar covariance (the element sum of the covariance matrix)
    double GetSampleCovariance() const {
        double result = 0;
        Vector<N> mean = GetSampleMean();
        for (size_t j = 0; j < sample_.size(); j++) {
            result += (sample_[j]-mean)*(sample_[j]-mean);
            for (size_t k = 0; k < j; k++) {
                result += 2*(sample_[j]-mean)*(sample_[k]-mean);
            }
        }
        result /= sample_.size();

        return result;
    }

    // get the effective number of samples (due to the sample covariance)
    double GetEffectiveNumSamples() const {
        return sample_.size() * GetSampleVariance()/GetSampleCovariance();

    }

    /*// get the fraction of new state samples that are accepted
    double GetAcceptanceRate() const {
        return num_accepted_/sample_.size();
    }*/


    // make an iteration of the MCMC algorithm
    void MakeIter() {
        states_ = ComputeIter();

        sample_.insert(sample_.end(), states_.begin(), states_.end());
    }

    // call a loop over MakeIter for K steps
    void MakeIters(unsigned int K) {
        for (unsigned int k = 0; k < K; k++) {
            MakeIter();
        }
    }


private:
    // log-likelyhood function
    std::function<double(const Vector<N> &)> lnP_;

    // current states of the walkers
    std::array<Vector<N>, W> states_;
    // logprobs at the current states
    std::array<double, W> logprobs_;
    // the entire sample
    std::vector<Vector<N>> sample_;

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
    double StretchP(double stretch) const {
        return sqrt(beta_/stretch)/2/(beta_-1);
    }

    // sample the stretch factor from above pdf
    double SampleStretch() {
        double rand01 = dist01_(randgen_);

        return ((beta_-1)*rand01 + 1) * ((beta_-1)*rand01 + 1) /beta_;
    }

    // compute an iteration of the MCMC algorithm
    std::array<Vector<N>, W> ComputeIter() {

        // sample new states
        std::array<Vector<N>, W> new_states;
        std::array<double, W> move_prob;
        for (unsigned int w = 0; w < W; w++) {
            size_t idx = dist0W2_(randgen_);
            if (idx >= w) {
                idx++;
            }

            double stretch = SampleStretch();
            move_prob[w] = 1/(W-1) * StretchP(stretch);

            new_states[w] = states_[w] + (states_[idx] - states_[w]) * stretch;
        }

        // calculate logprobs
        std::array<double, W> new_logprobs;
        for (unsigned int w = 0; w < W; w++) {
            new_logprobs[w] = lnP_(new_states[w]);
        }

        // calculate acceptance probabilities
        std::array<double, W> acceptance_probs;
        for (unsigned int w = 0; w < W; w++) {
            acceptance_probs[w] = alpha_/move_prob[w] * exp((new_logprobs[w] - logprobs_[w])/2);
        }

        // accept or reject
        std::array<Vector<N>, W> result = states_;
        double rand01;
        for (unsigned int w = 0; w < W; w++) {
            rand01 = dist01_(randgen_);
            if (rand01 < acceptance_probs[w]) {
                result[w] = new_states[w];
            }
        }

        return result;
    }

};



#endif //MCMC_H_INCLUDED

