#include <utility>
#include <functional>
#include <random>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>

#include <omp.h>

#include <utils.h>


#ifndef MCMC_BASE_H_INCLUDED
#define MCMC_BASE_H_INCLUDED


// base class for MCMC samplers
template<int N, unsigned int W>
class MCMC_base {

public:
    // constructor
    MCMC_base(std::function<double(const Vector<N> &)> lnP, std::array<Vector<N>, W> init_states, double alpha)
        :
        lnP_(lnP),
        states_(init_states),
        num_iters_(0),
        num_accepted_(0),
        alpha_(alpha),
        dist01_(0.0, 1.0)
    {
        // calculate initial logprobs
        for (unsigned int w = 0; w < W; w++) {
            logprobs_[w] = lnP_(states_[w]);
        }

        // append initial states and logprobs to the sample
        sample_.push_back(states_);
        sample_logprobs_.push_back(logprobs_);
    }

    // destructor
    ~MCMC_base() = default;

    // get the current walker states
    std::array<Vector<N>, W> GetStates() const {
        return states_;
    }

    // get the entire sample
    std::vector<std::array<Vector<N>, W>> GetSample() const {
        return sample_;
    }

    // get the state mean
    Vector<N> GetStateMean(const std::array<Vector<N>, W> &states) const {
        Vector<N> result = Vector<N>::Zero();
        for (unsigned int w = 0; w < W; w++) {
            result += states[w];
        }
        result /= W;

        return result;
    }

    // GetStateMean(states_)
    Vector<N> GetStateMean() const {
        return GetStateMean(states_);
    }

    // get the sample mean
    Vector<N> GetSampleMean() const {
        Vector<N> result = Vector<N>::Zero();
        for (const std::array<Vector<N>, W> &states : sample_) {
            result += GetStateMean(states);
        }
        result /= sample_.size();

        return result;
    }

    // get the state variance matrix
    Matrix<N> GetStateVariance(const std::array<Vector<N>, W> &states, const Vector<N> &mean) const {
        Matrix<N> result = Matrix<N>::Zero();
        for (unsigned int w = 0; w < W; w++) {
            result += (states[w]-mean)*(states[w]-mean).transpose();
        }
        result /= W;

        return result;
    }

    // GetStateVariance(states, GetStateMean(states))
    Matrix<N> GetStateVariance(const std::array<Vector<N>, W> &states) const {
        return GetStateVariance(states, GetStateMean(states));
    }

    // GetStateVariance(states_, mean)
    Matrix<N> GetStateVariance(const Vector<N> &mean) const {
        return GetStateVariance(states_, mean);
    }

    // GetStateVariance(states_)
    Matrix<N> GetStateVariance() const {
        return GetStateVariance(states_);
    }

    // get the sample variance matrix
    Matrix<N> GetSampleVariance(const Vector<N> &mean) const {
        Matrix<N> result = Matrix<N>::Zero();
        for (const std::array<Vector<N>, W> &states : sample_) {
            result += GetStateVariance(states, mean);
        }
        result /= sample_.size();

        return result;
    }

    // GetSampleVariance(GetSampleMean())
    Matrix<N> GetSampleVariance() const {
        return GetSampleVariance(GetSampleMean());
    }

    // get the state lag-k covariance matrix
    Matrix<N> GetStatesCovariance(const std::array<Vector<N>, W> &states1, const std::array<Vector<N>, W> &states2, const Vector<N> &mean) const {
        Matrix<N> result = Matrix<N>::Zero();
        for (unsigned int w = 0; w < W; w++) {
            result += (states1[w]-mean)*(states2[w]-mean).transpose();
        }
        result /= W;

        return result;
    }

    // get the sample lag-k covariance matrix
    Matrix<N> GetSampleLagKCovariance(unsigned int k, const Vector<N> &mean) const {
        Matrix<N> result = Matrix<N>::Zero();
        for (size_t j = 0; j < sample_.size()-k; j++) {
            result += GetStatesCovariance(sample_[j], sample_[j+k], mean);
        }
        result /= sample_.size();

        return result;
    }

    // GetSampleLagKCovariance(k, GetSampleMean())
    Matrix<N> GetSampleLagKCovariance(unsigned int k) const {
        return GetSampleLagKCovariance(k, GetSampleMean());
    }

    // get the sample covariance matrix
    Matrix<N> GetSampleCovariance(const Vector<N> &mean) const {
        Matrix<N> result = GetSampleVariance(mean);
        double det_var = result.determinant();
        Matrix<N> lag_k;
        double tau;
        for (unsigned int k = 1; k < sample_.size()/2; k++) {
            lag_k = GetSampleLagKCovariance(k, mean);
            result += lag_k + lag_k.transpose();

            tau = pow(result.determinant()/det_var, 1/N);
            if (k > 5*tau) {
                break;
            }
        }

        return result;
    }

    // GetSampleCovariance(GetSampleMean())
    Matrix<N> GetSampleCovariance() const {
        return GetSampleCovariance(GetSampleMean());
    }

    // get the integrated autocorrelation time
    double GetIntegAutocorrTime(const Vector<N> &mean) const {
        Matrix<N> result = GetSampleVariance(mean);
        double det_var = result.determinant();
        Matrix<N> lag_k;
        double tau;
        for (unsigned int k = 1; k < sample_.size()/2; k++) {
            lag_k = GetSampleLagKCovariance(k, mean);
            result += lag_k + lag_k.transpose();

            tau = pow(result.determinant()/det_var, 1/N);
            if (k > 5*tau) {
                break;
            }
        }

        return tau;
    }

    // GetIntegAutocorrTime(GetSampleMean())
    double GetIntegAutocorrTime() const {
        return GetIntegAutocorrTime(GetSampleMean());

    }

    // get the fraction of new state samples that have been accepted
    double GetAcceptanceRate() const {
        return double(num_accepted_)/(num_iters_*W);
    }

    // reset the members (i.e. sample, logprobs and counters)
    void Reset() {
        sample_ = {states_};
        sample_logprobs_ = {logprobs_};
        num_iters_ = 0;
        num_accepted_ = 0;
    }

    // make an iteration of the respective MCMC algorithm
    void MakeIter() {
        // compute iteration
        ComputeIter();

        // append new states and logprobs to the sample
        sample_.push_back(states_);
        sample_logprobs_.push_back(logprobs_);
    }

    // savd the sample to a txt file
    void SaveSample(std::filesystem::path path, bool overwrite = false) {
        // check that the out file doesn't exist yet if overwrite is disabled
        if (!overwrite && std::filesystem::exists(path)) {
            throw std::runtime_error(("The path \"" + path.string() + "\" already exists and must not be overwritten.").c_str());
        }

        // open the file
        std::ofstream file;
        file.open(path);
        if (!file) {
            throw std::runtime_error(("Could not open \"" + path.string() + "\".").c_str());
        }

        // write the sample
        for (size_t j = 0; j < sample_.size(); j++) {
            for (unsigned int w = 0; w < W; w++) {
                for (int n = 0; n < N; n++) {
                    file << sample_[j][w][n] << ", ";
                }
                file << sample_logprobs_[j][w] << "\n";
            }
            file << "\n";
        }

        // close the file
        file.close();
    }


protected:
    // log-likelyhood function
    std::function<double(const Vector<N> &)> lnP_;

    // current states of the walkers
    std::array<Vector<N>, W> states_;
    // the logprobs of the current states
    std::array<double, W> logprobs_;
    // the entire sample
    std::vector<std::array<Vector<N>, W>> sample_;
    // the logprobs of the sample
    std::vector<std::array<double, W>> sample_logprobs_;
    // number of performed iterations
    size_t num_iters_;
    // number of accepted new states
    size_t num_accepted_;

    // tuning constant for acceptance probabilities
    double alpha_;

    // uniform double distribution from 0 to 1
    std::uniform_real_distribution<double> dist01_;

    // update the sampler members
    // does nothing by default
    // can be overwritten by children if needed
    virtual void UpdateInternals() {};

    // sample a new state for walker w and also return the associated pdf value
    virtual std::pair<Vector<N>, double> SampleNewState(unsigned int w, std::mt19937 &randgen) = 0;

    // compute an iteration of the respective MCMC algorithm
    void ComputeIter() {
        num_iters_++;

        // update sampler members
        UpdateInternals();

        #pragma omp parallel for
        for (unsigned int w = 0; w < W; w++) {

            // each thread gets its own RNG
            thread_local static std::mt19937 randgen(std::random_device{}());

            // sample new state
            std::pair<Vector<N>, double> new_state_tmp = SampleNewState(w, randgen);
            Vector<N> new_state = new_state_tmp.first;
            double new_state_prob = new_state_tmp.second;

            // calculate logprob for new state
            double new_logprob = lnP_(new_state);

            // calculate acceptance probability
            double acceptance_prob = alpha_/new_state_prob * exp((new_logprob - logprobs_[w])/2);

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



#endif //MCMC_BASE_H_INCLUDED

