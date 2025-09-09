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
    }

    // destructor
    ~MCMC_base() = default;

    // broadcast a std::array<Vector<N>, W> (e.g. the walker states) over all processes
    void Broadcast(std::array<Vector<N>, W> &vec_arr) {
        unsigned int start;
        unsigned int stop;
        for (unsigned int proc = 0; proc < num_procs_; proc++) {
            start = walkers_per_proc_ *proc;
            stop = walkers_per_proc_ *(proc+1);
            MPI_Bcast(vec_arr[start].data(), (stop-start)*N, MPI_DOUBLE, proc, MPI_COMM_WORLD);
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
    std::array<std::vector<Vector<N>>, W> GetSample() const {
        return sample_;
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

    // get the state variance matrix
    Matrix<N> GetStateVariance(const Vector<N> &mean) const {
        Matrix<N> result = Matrix<N>::Zero();
        for (unsigned int w = 0; w < W; w++) {
            result += (states_[w]-mean)*(states_[w]-mean).transpose();
        }
        result /= (W-1);

        return result;
    }

    // GetStateVariance(GetStateMean())
    Matrix<N> GetStateVariance() const {
        return GetStateVariance(GetStateMean());
    }

    // get a rank-normalized and folded split-RHat measure to assess chain convergence
    // and get the effective sample size (as determined by the sample covariance) 
    // (both adapted from https://arxiv.org/pdf/1903.08008)
    std::pair<double, double> GetRHatAndESS(size_t L) const {
        // split each chain in two halves
        std::array<std::vector<Vector<N>>, 2*W> split_sample;
        # pragma omp parallel for
        for (unsigned int w = 0; w < W; w++) {
            split_sample[2*w] = std::vector<Vector<N>>(sample_[w].end() - L, sample_[w].end() - L + L/2);
            split_sample[2*w+1] = std::vector<Vector<N>>(sample_[w].end() - L/2, sample_[w].end());
        }

        // get the RHat and ESS for the rank normalized split-chains
        std::pair<double, double> RHat_and_ESS = ComputeRHatAndESS(RankNormalize(split_sample));

        // fold each split-chain about its median
        std::array<std::vector<Vector<N>>, 2*W> folded_split_sample;
        # pragma omp parallel for
        for (unsigned int w = 0; w < 2*W; w++) {
            folded_split_sample[w] = Fold(split_sample[w]);
        }

        // get the RHat and ESS for the rank normalized folded split-chains
        std::pair<double, double> folded_RHat_and_ESS = ComputeRHatAndESS(RankNormalize(folded_split_sample));

        // return the maximum of the two RHats and the minimum of the two ESSs
        return std::make_pair(std::max(RHat_and_ESS.first, folded_RHat_and_ESS.first),
                                std::min(RHat_and_ESS.second, folded_RHat_and_ESS.second));
    }

    // GetRHatAndESS on the entire available sample
    std::pair<double, double> GetRHatAndESS() const {
        return GetRHatAndESS(sample_[0].size());
    }

    // get the fraction of new state samples that have been accepted
    double GetAcceptanceRate() const {
        return double(num_accepted_)/(num_iters_*W);
    }

    // reset the members (i.e. sample, logprobs and counters)
    void Reset(size_t keep = 0) {
        if (proc_ == 0) {
            for (unsigned int w = 0; w < W; w++) {
                sample_[w] = std::vector<Vector<N>>(sample_[w].end() - keep, sample_[w].end());
                sample_logprobs_[w] = std::vector<double>(sample_logprobs_[w].end() - keep, sample_logprobs_[w].end());
            }
            num_accepted_ *= keep/num_iters_;
            num_iters_ = keep;
        }
    }

    // make an iteration of the respective MCMC algorithm
    void MakeIter() {
        // compute iteration
        ComputeIter();

        // append new states and logprobs to the sample
        if (proc_ == 0) {
            for (unsigned int w = 0; w < W; w++) {
                sample_[w].push_back(states_[w]);
                sample_logprobs_[w].push_back(logprobs_[w]);
            }
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
            for (unsigned int w = 0; w < W; w++) {
                for (size_t j = 0; j < sample_[w].size(); j++) {
                    for (int n = 0; n < N; n++) {
                        file << sample_[w][j](n) << ", ";
                    }
                    file << sample_logprobs_[w][j] << "\n";
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
    std::array<std::vector<Vector<N>>, W> sample_;
    // the logprobs of the sample
    std::array<std::vector<double>, W> sample_logprobs_;
    // number of performed iterations
    size_t num_iters_;
    // number of accepted new states
    size_t num_accepted_;

    // tuning constant for acceptance probabilities
    double alpha_;

    // uniform double distribution from 0 to 1
    std::uniform_real_distribution<double> dist01_;

    // rank-normalize the chains
    std::array<std::vector<Vector<N>>, 2*W> RankNormalize(const std::array<std::vector<Vector<N>>, 2*W> &chains) const {
        size_t L = chains[0].size();
        size_t S = L*2*W;
        // flatten the chains for each component
        std::array<std::vector<double>, N> flattened_chains;
        for (int n = 0; n < N; n++) {
            flattened_chains[n].resize(2*W*L);
        }
        # pragma omp parallel for
        for (unsigned int w = 0; w < 2*W; w++) {
            for (size_t j = 0; j < chains[w].size(); j++) {
                for (int n = 0; n < N; n++) {
                    flattened_chains[n][chains[w].size()*w+j] = chains[w][j](n);
                }
            }
        }
        // sort the flattened chains
        for (int n = 0; n < N; n++) {
            sort(flattened_chains[n]);
        }
        // calculate the rank of each value and transform into a standard normal using the probit function
        size_t rank;
        std::array<std::vector<Vector<N>>, 2*W> rank_normalized_chains;
        # pragma omp parallel for
        for (unsigned int w = 0; w < 2*W; w++) {
            rank_normalized_chains[w].resize(chains[w].size());
            for (size_t j = 0; j < chains[w].size(); j++) {
                for (int n = 0; n < N; n++) {
                    rank = get_index(chains[w][j](n), flattened_chains[n]) + 1;
                    rank_normalized_chains[w][j](n) = probit((rank - 0.375) /(S + 0.25));
                }
            }
        }

        return rank_normalized_chains;
    }

    // fold the chain about its median
    std::vector<Vector<N>> Fold(const std::vector<Vector<N>> &chain) const {
        // extract the individual components of each chain
        std::array<std::vector<double>, N> compwise_chains;
        for (int n = 0; n < N; n++) {
            for (size_t j = 0; j < chain.size(); j++) {
                compwise_chains[n].push_back(chain[j](n));
            }
        }

        // compute the componentwise median of the chain
        Vector<N> chain_median;
        for (int n = 0; n < N; n++) {
            std::nth_element(compwise_chains[n].begin(), compwise_chains[n].begin() + compwise_chains[n].size()/2, compwise_chains[n].end());
            chain_median(n) = compwise_chains[n][compwise_chains[n].size()/2];
        }

        // fold the chain abouth that median
        std::vector<Vector<N>> folded_chain(chain.size());
        for (size_t j = 0; j < chain.size(); j++) {
            folded_chain[j] = (chain[j]-chain_median).array().abs();
        }

        return folded_chain;
    }

    // compute the classic RHat and ESS from the chains
    std::pair<double, double> ComputeRHatAndESS(const std::array<std::vector<Vector<N>>, 2*W> &chains) const {
        size_t L = chains[0].size();
        // compute the chain means and variances and the global mean
        std::array<Vector<N>, 2*W> chain_means;
        std::array<double, 2*W> chain_vars;
        # pragma omp parallel for
        for (unsigned int w = 0; w < 2*W; w++) {
            chain_means[w] = get_mean(chains[w]);
            chain_vars[w] = get_variance(chains[w], chain_means[w]);
        }
        Vector<N> global_mean = get_mean(chain_means);
        // compute the between and within chain variances
        double between = 0.0;
        double within = 0.0;
        # pragma omp parallel for
        for (unsigned int w = 0; w < 2*W; w++) {
            between += (chain_means[w] - global_mean).dot(chain_means[w] - global_mean);
            within += chain_vars[w];
        }
        between /= (2*W-1);
        within /= 2*W;

        // calculate the marginal posterior variance
        double var_plus = (1-1/L)*within + between;

        // compute RHat
        double RHat = sqrt(var_plus /within);

        // calculate the integrated autocorrelation time
        double tau = 1.0;
        double rho = INFINITY;
        double rho_prev;
        double numerator;
        for (size_t k = 0; k < (L-1)/2; k++) {
            numerator = 0.0;
            // calculate lag 2k and 2k+1 autocorrelations
            # pragma omp parallel for
            for (unsigned int w = 0; w < 2*W; w++) {
                numerator -= get_lag_k_covariance(chains[w], 2*k, chain_means[w])
                            + get_lag_k_covariance(chains[w], 2*k+1, chain_means[w]);
            }
            numerator /= 2*W;
            numerator += 2*within;

            rho_prev = rho;
            rho = 2.0 - numerator / var_plus;
            // only accept rhos as long as they are monotone descreasing and positive
            if (rho < rho_prev && rho > 0) {
                tau += 2*rho;
            }
            else {
                break;
            }
        }

        // compute the ESS
        double ESS = L*2*W /tau;

        return std::make_pair(RHat, ESS);
    }

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
            //std::cout << acceptance_prob << " , " << new_state_prob << " , " << new_logprob - logprobs_[w] << std::endl;

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

