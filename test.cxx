#include <vector>
#include <functional>
#include <random>
#include <array>
#include <cmath>
#include <iostream>

#include <samplers/MCMC.h>
#include <samplers/MCMC2.h>
#include <ProgressBar.h>
#include <utils.h>


double mean = 0;
double sigma = 1;

double log_gauss(const Vector<1> &state) {
    return -(state[0]-mean)*(state[0]-mean)/2/sigma/sigma - log(2*M_PI*sigma*sigma)/2;
}


int main(int argc, char* argv[]) {

    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Threads used: " << omp_get_num_threads() << "\n";
    }

    std::array<Vector<1>, 10> init_states;
    init_states[0] = Vector<1>{0.1};
    init_states[1] = Vector<1>{0.2};
    init_states[2] = Vector<1>{0.3};
    init_states[3] = Vector<1>{-0.01};
    init_states[4] = Vector<1>{-0.23};
    init_states[5] = Vector<1>{-0.54};
    init_states[6] = Vector<1>{2.0};
    init_states[7] = Vector<1>{-1.5};
    init_states[8] = Vector<1>{0.07};
    init_states[9] = Vector<1>{0.85};



    MCMC2<1, 10> sampler(log_gauss, init_states, 40, "/home/aurora/sim_results/MCMC_test.txt", 2);

    unsigned int K = 100000;
    DotProgressBar progress_bar(K, "iter(s)", int(K/100), 70);

    for (unsigned int k = 0; k < K; k++) {
        sampler.MakeIter();
        progress_bar.step();
        std::cout << "mean = " << sampler.GetStateMean() << std::endl;
        std::cout << "variance = " << sampler.GetStateVariance() << std::endl;
        std::cout << std::endl;
    }

    std::cout << "acceptance_rate: " << sampler.GetAcceptanceRate() << std::endl;

}


