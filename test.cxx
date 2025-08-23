#include <vector>
#include <functional>
#include <random>
#include <array>
#include <cmath>

#include <samplers/MCMC.h>
#include <utils.h>


double mean = 0;
double sigma = 1;

double log_gauss(const State<1> &state) {
    return -(state[0]-mean)*(state[0]-mean)/2/sigma/sigma - log(2*M_PI*sigma*sigma)/2;
}


int main(int argc, char* argv[]) {

    std::array<State<1>, 2> init_states;
    init_states[0] = {0.1};
    init_states[1] = {0.2};

    MCMC<1, 2> sampler(log_gauss, init_states, 0.5);

    sampler.MakeIters(100);

    std::vector<std::array<double, 1>> sample = sampler.GetSample();

    save_arrayvector_as_txt(sample, "/home/aurora/sim_results/MCMC_test.txt");

}


