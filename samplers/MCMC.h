#include <vector>
#include <functional>
#include <random>

#include <omp.h>

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
        num_accepted_(0),
        alpha_(alpha),
        beta_(beta),
        dist01_(0.0, 1.0),
        dist0W2_(0, W-2)
    {
        // calculate initial logprobs
        for (unsigned int w = 0; w < W; w++) {
            logprobs_[w] = lnP_(states_[w]);
        }
    }

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

    // get the fraction of new state samples that are accepted
    double GetAcceptanceRate() const {
        return double(num_accepted_)/sample_.size();
    }


    // make an iteration of the MCMC algorithm
    void MakeIter() {
        ComputeIter();

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
    // number of accepted new states
    size_t num_accepted_;

    // tuning constant for acceptance probabilities
    double alpha_;
    // range constant for the state pdf
    double beta_;

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
    void ComputeIter() {

        #pragma omp parallel for
        for (unsigned int w = 0; w < W; w++) {

            // each thread gets its own RNG
            thread_local static std::mt19937 randgen(std::random_device{}());

            // sample new state
            unsigned int idx = dist0W2_(randgen);
            if (idx >= w) {
                idx++;
            }

            double stretch = SampleStretch();
            double move_prob = StretchP(stretch) /(W-1);

            Vector<N> new_state = states_[w] + (states_[idx] - states_[w]) * stretch;
        
            // calculate logprob
            double new_logprob = lnP_(new_state);

            // calculate acceptance probability
            double acceptance_prob = alpha_/move_prob * exp((new_logprob - logprobs_[w])/2);

            // accept or reject
            double rand01 = dist01_(randgen);
            if (rand01 < acceptance_prob) {
                states_[w] = new_state;
                logprobs_[w] = new_logprob;

                #pragma omp atomic
                num_accepted_++;
            }
        }
    }

};



#endif //MCMC_H_INCLUDED

