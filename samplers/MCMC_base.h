#include <utility>
#include <functional>
#include <random>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>

#include <omp.h>
#include <mpi.h>

#include <utils.h>


#ifndef MCMC_BASE_H_INCLUDED
#define MCMC_BASE_H_INCLUDED


// base class for MCMC samplers
template<int N, unsigned int W>
class MCMC_base {

public:
    // constructor
    MCMC_base(int proc, int num_procs, std::function<double(const Vector<N> &)> lnP, std::array<Vector<N>, W> init_states, double alpha)
        :
        proc_(proc),
        num_procs_(num_procs),
        lnP_(lnP),
        states_(init_states),
        num_iters_(0),
        num_accepted_(0),
        alpha_(alpha),
        dist01_(0.0, 1.0)
    {
        // walker range that belongs to this process
        walkers_per_proc_ = W /num_procs;
        start_ = walkers_per_proc_ *proc;
        stop_ = walkers_per_proc_ *(proc+1);

        // calculate initial logprobs
        for (unsigned int w = start_; w < stop_; w++) {
            logprobs_[w] = lnP_(states_[w]);
        }
        // broadcast logprobs
        Broadcast(logprobs_);

        // append initial states and logprobs to the sample (only on main process)
        if (proc_ == 0) {
            sample_.push_back(states_);
            sample_logprobs_.push_back(logprobs_);
        }
    }

    // destructor
    ~MCMC_base() = default;

    // broadcast a std::array<Vector<N>, W> (e.g. the walker states) over all processes
    void Broadcast(std::array<Vector<N>, W> &vec_arr) {
        // flatten the std::array<Vector<N>, W>
        std::array<double, W*N> flattened;
        for (unsigned int w = 0; w < W; w++) {
            for (int n = 0; n < N; n++) {
                flattened[w*N+n] = vec_arr[w](n);
            }
        }

        // broadcast the flattened array
        unsigned int start;
        unsigned int stop;
        for (unsigned int proc = 0; proc < num_procs_; proc++) {
            start = walkers_per_proc_ *proc;
            stop = walkers_per_proc_ *(proc+1);
            MPI_Bcast(&flattened[start*N], (stop-start)*N, MPI_DOUBLE, proc, MPI_COMM_WORLD);
        }

        // reconstruct the std::array<Vector<N>, W>
        for (unsigned int w = 0; w < W; w++) {
            for (int n = 0; n < N; n++) {
                vec_arr[w](n) = flattened[w*N+n];
            }
        }
    }

    // broadcast a std::array<double, W> (e.g. the logprobs) over all processes
    void Broadcast(std::array<double, W> &arr) {
        unsigned int start;
        unsigned int stop;
        for (unsigned int proc = 0; proc < num_procs_; proc++) {
            start = walkers_per_proc_ *proc;
            stop = walkers_per_proc_ *(proc+1);
            MPI_Bcast(&arr[start], stop-start, MPI_DOUBLE, proc, MPI_COMM_WORLD);
        }
    }

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

    // get the covariance between two sets of walker states
    Matrix<N> GetStatesCovariance(const std::array<Vector<N>, W> &states1, const std::array<Vector<N>, W> &states2, const Vector<N> &mean1, const Vector<N> &mean2) const {
        Matrix<N> result = Matrix<N>::Zero();
        for (unsigned int w = 0; w < W; w++) {
            result += (states1[w]-mean1)*(states2[w]-mean2).transpose();
        }
        result /= W;

        return result;
    }

    // GetStatesCovariance(states1, states2, GetStateMean(states1), GetStateMean(states2))
    Matrix<N> GetStatesCovariance(const std::array<Vector<N>, W> &states1, const std::array<Vector<N>, W> &states2) const {
        return GetStatesCovariance(states1, states2, GetStateMean(states1), GetStateMean(states2));
    }

    // get the state variance matrix
    Matrix<N> GetStateVariance(const std::array<Vector<N>, W> &states, const Vector<N> &mean) const {
        return GetStatesCovariance(states, states, mean, mean);
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

    // get the sample lag-k covariance matrix
    Matrix<N> GetSampleLagKCovariance(unsigned int k, const Vector<N> &mean) const {
        Matrix<N> result = Matrix<N>::Zero();
        for (size_t j = 0; j < sample_.size()-k; j++) {
            result += GetStatesCovariance(sample_[j], sample_[j+k], mean, mean);
        }
        result /= sample_.size();

        return result;
    }

    // GetSampleLagKCovariance(k, GetSampleMean())
    Matrix<N> GetSampleLagKCovariance(unsigned int k) const {
        return GetSampleLagKCovariance(k, GetSampleMean());
    }

    // get the sample variance matrix
    Matrix<N> GetSampleVariance(const Vector<N> &mean) const {
        return GetSampleLagKCovariance(0, mean);
    }

    // GetSampleVariance(GetSampleMean())
    Matrix<N> GetSampleVariance() const {
        return GetSampleVariance(GetSampleMean());
    }

    // get the sample covariance matrix and the integrated autocorrelation time
    std::pair<Matrix<N>, double> GetSampleCovarianceAndIntegAutocorrTime(const Vector<N> &mean) const {
        Matrix<N> result = GetSampleVariance(mean);
        double det_var = result.determinant();
        Matrix<N> lag_k;
        double tau = 1.0;
        for (unsigned int k = 1; k < sample_.size()/2; k++) {
            lag_k = GetSampleLagKCovariance(k, mean);
            result += lag_k + lag_k.transpose();

            tau = pow(result.determinant()/det_var, 1/N);
            if (k > 5*tau) {
                break;
            }
        }

        return std::make_pair(result, tau);
    }

    // GetSampleCovarianceAndIntegAutocorrTime(GetSampleMean())
    std::pair<Matrix<N>, double> GetSampleCovarianceAndIntegAutocorrTime() const {
        return GetSampleCovarianceAndIntegAutocorrTime(GetSampleMean());
    }

    // get the sample covariance matrix
    Matrix<N> GetSampleCovariance(const Vector<N> &mean) const {
        return GetSampleCovarianceAndIntegAutocorrTime(mean).first;
    }

    // GetSampleCovariance(GetSampleMean())
    Matrix<N> GetSampleCovariance() const {
        return GetSampleCovariance(GetSampleMean());
    }

    // get the integrated autocorrelation time
    double GetIntegAutocorrTime(const Vector<N> &mean) const {
        return GetSampleCovarianceAndIntegAutocorrTime(mean).second;
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
        if (proc_ == 0) {
            sample_ = {states_};
            sample_logprobs_ = {logprobs_};
            num_iters_ = 0;
            num_accepted_ = 0;
        }
    }

    // make an iteration of the respective MCMC algorithm
    void MakeIter() {
        // compute iteration
        ComputeIter();

        // append new states and logprobs to the sample
        if (proc_ == 0) {
            sample_.push_back(states_);
            sample_logprobs_.push_back(logprobs_);
        }
    }

    // save the sample to a txt file
    void SaveSample(fs::path path, bool overwrite = false) {
        if (proc_ == 0) {
            // check that the out file doesn't exist yet if overwrite is disabled
            if (!overwrite && fs::exists(path)) {
                throw std::runtime_error(("The path \"" + path.string() + "\" already exists overwrite is set to false.").c_str());
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
                        file << sample_[j][w](n) << ", ";
                    }
                    file << sample_logprobs_[j][w] << "\n";
                }
                file << "\n";
            }

            // close the file
            file.close();
        }
    }


protected:
    // MPI: process number and number of total processes
    int proc_;
    int num_procs_;
    double walkers_per_proc_;
    unsigned int start_;
    unsigned int stop_;

    // log-likelihood function
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
        unsigned int accepted = 0;

        // update sampler members
        UpdateInternals();

        #pragma omp parallel for
        for (unsigned int w = start_; w < stop_; w++) {

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
                accepted++;
            }
        }

        // broadcast states and logprobs
        Broadcast(states_);
        Broadcast(logprobs_);

        // update counters
        if (proc_ == 0) {
            num_iters_++;
            num_accepted_ += accepted;
            for (int proc = 1; proc < num_procs_; proc++) {
                MPI_Recv(&accepted, 1, MPI_UNSIGNED, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                num_accepted_ += accepted;
            }
        }
        else {
            MPI_Send(&accepted, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
        }
    }
};



#endif //MCMC_BASE_H_INCLUDED

