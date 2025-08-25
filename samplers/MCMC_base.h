#include <pair>
#include <functional>
#include <random>
#include <fstream>
#include <iostream>
#include <filesystem>

#include <omp.h>

#include <utils.h>


#ifndef MCMC_BASE_H_INCLUDED
#define MCMC_BASE_H_INCLUDED


// base class for MCMC samplers
template<int N, unsigned int W>
class MCMC_base {

public:
    // constructor
    MCMC_base(std::function<double(const Vector<N> &)> lnP, std::array<Vector<N>, W> init_states, double alpha, std::filesystem::path out_path)
        :
        lnP_(lnP),
        states_(init_states),
        num_iters_(0),
        num_accepted_(0),
        alpha_(alpha),
        dist01_(0.0, 1.0),
        out_path_(out_path)
    {
        // calculate initial logprobs
        for (unsigned int w = 0; w < W; w++) {
            logprobs_[w] = lnP_(states_[w]);
        }

        // update sampler members for the first time
        UpdateInternals();
    }

    // destructor
    ~MCMC_base() = default;

    // get the current walker states
    std::array<Vector<N>, W> GetStates() const {
        return states_;
    }

    // get the state mean
    Vector<N> GetStateMean() const {
        Vector<N> result = Vector<N>::Zero();
        for (unsigned int w = 0; w < W; w++) {
            result += states_[w];
        }
        result /= W;

        return result;
    }

    // get the state scalar variance (the trace of the variance matrix)
    double GetStateVariance() const {
        double result = 0;
        Vector<N> mean = GetStateMean();
        for (unsigned int w = 0; w < W; w++) {
            result += (states_[w]-mean).dot(states_[w]-mean);
        }
        result /= W;

        return result;
    }

    // get the state scalar covariance (the trace of the covariance matrix)
    double GetStateCovariance() const {
        double result = 0;
        Vector<N> mean = GetStateMean();
        for (unsigned int w = 0; w < W; w++) {
            result += (states_[w]-mean).dot(states_[w]-mean);
            for (unsigned int j = 0; j < w; j++) {
                result += 2*(states_[w]-mean).dot(states_[j]-mean);
            }
        }
        result /= W;

        return result;
    }

    // get the state variance matrix
    double GetStateVarianceMatrix() const {
        double result = 0;
        Vector<N> mean = GetStateMean();
        for (unsigned int w = 0; w < W; w++) {
            result += (states_[w]-mean)*(states_[w]-mean).transpose();
        }
        result /= W;

        return result;
    }

    // get the state covariance matrix
    Matrix<N> GetStateCovarianceMatrix() const {
        Matrix<N> result = Matrix<N>::Zero();
        Matrix<N> temp;
        Vector<N> mean = GetStateMean();
        for (unsigned int w = 0; w < W; w++) {
            result = (states_[w]-mean)*(states_[w]-mean).transpose();
            for (unsigned int j = 0; j < w; j++) {
                temp = (states_[w]-mean)*(states_[j]-mean).transpose();
                result += temp + temp.transpose();
            }
        }
        result /= W;

        return result;
    }


    // get the effective number of independent states (due to the state covariance)
    double GetEffectiveNumStates() const {
        return W * GetStateVariance()/GetStateCovariance();

    }

    // get the fraction of new state samples that have been accepted
    double GetAcceptanceRate() const {
        return double(num_accepted_)/(num_iters_*W);
    }


    // make an iteration of the respective MCMC algorithm
    // and save the resulting states to the .txt file
    void MakeAndSaveIter() {
        ComputeIter();

        // write new states to txt file
        out_file_.open(out_path_, std::ios::app);
        if (!out_file_) {
            throw std::runtime_error(("Could not open out file: " + out_path_.string()).c_str());
        }
        for (unsigned int w = 0; w < W-1; w++) {
            for (int n = 0; n < N-1; n++) {
                out_file_ << states_[n] << ", ";
            }
            out_file_ << states_[N-1] << "\n";
        }
        out_file_ << "\n";
        out_file_.close();
    }

    // make an iteration of the respective MCMC algorithm
    // without saving the resulting states
    void MakeIter() {
        ComputeIter();
    }


protected:
    // log-likelyhood function
    std::function<double(const Vector<N> &)> lnP_;

    // current states of the walkers
    std::array<Vector<N>, W> states_;
    // logprobs at the current states
    std::array<double, W> logprobs_;
    // number of performed iterations
    size_t num_iters_;
    // number of accepted new states
    size_t num_accepted_;

    // tuning constant for acceptance probabilities
    double alpha_;

    // uniform double distribution from 0 to 1
    std::uniform_real_distribution<double> dist01_;

    // .txt path/file to save sample to
    std::filesystem::path out_path_;
    std::ofstream out_file_;

    // update the sampler members
    // does nothing by default
    // can be overwritten by children if needed
    virtual void UpdateInternals() {};

    // sample a new state for walker w and also return the associated pdf value
    virtual std::pair<Vector<N>, double> SampleNewState(unsigned int w, std::mt19937 &randgen) = 0;

    // compute an iteration of the respective MCMC algorithm
    void ComputeIter() {

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
        num_iters_++;

        // update sampler members
        UpdateInternals();
    }
};



#endif //MCMC_BASE_H_INCLUDED

