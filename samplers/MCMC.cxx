#include <cmath>

#include <samplers/MCMC.h>


template<size_t N, size_t W>
double MCMC<N, W>::SampleStretch() {
    // sample a stretch factor
    double rand01 = dist01_(randgen_);

    return ((beta_-1)*rand01 + 1) * ((beta_-1)*rand01 + 1) /beta_;
}



template<size_t N, size_t W>
double MCMC<N, W>::StretchP(double stretch) const {
    // compute the pdf value for the stretch
    return sqrt(beta_/stretch)/2/(beta_-1);
}


template<size_t N, size_t W>
std::array<State<N>, W> MCMC<N, W>::ComputeIter() {
    // make an affine invariant MCMC iteration

    // sample new states
    std::array<State<N>, W> new_states;
    std::array<double, W> move_prob;
    for (size_t w = 0; w < W; w++) {
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
    for (size_t w = 0; w < W; w++) {
        new_logprobs[w] = lnP_(new_states[w]);
    }

    // calculate acceptance probabilities
    std::array<double, W> acceptance_probs;
    for (size_t w = 0; w < W; w++) {
        acceptance_probs[w] = alpha_/move_prob[w] * exp((new_logprobs[w] - logprobs_[w])/2);
    }

    // accept or reject
    double rand01;
    for (size_t w = 0; w < W; w++) {
        rand01 = dist01_(randgen_);
        if (rand01 < acceptance_probs[w]) {
            states_[w] = new_states[w];
        }    
    }
}


