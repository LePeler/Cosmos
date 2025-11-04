#include <cmath>
#include <array>

#include <utils.h>
#include <ProgressBar.h>
#include <solvers/RK4.h>


// state {H, K}
// params {H0, K0, Of0, n}


double H0 = 73.0;
double K0 = 36.0;
double Of0 = 0.1;
double n = -3.0;

double dz = 1e-6;
unsigned int K = 3e6;


Vector<2> SFT(const Vector<2> &state, double z) {
    double H = state(0);
    double K = state(1);
    double a = 1/(z+1);

    double K_prime = Of0*H0*H0*H0/H/H*pow(a, n+3) - H*a*a + 3*K*a - K*K/H;
    return Vector<2>{K, K_prime};
}


int main(int argc, char* argv[]) {

    RK4<2> solver(SFT, Vector<2>{H0, K0}, 0);

    DotProgressBar progress_bar(K, "step(s)", int(K/100), 70);
    for (unsigned int k = 0; k < K; k++) {
        solver.MakeStep(dz);
        progress_bar.step();

        if (std::isnan(solver.GetCurrentValue()(0))) {
            std::cout << "Big Bang reached! Solving stopped." << std::endl;
            break;
        }
    }

    save_vector_as_txt(solver.GetTimes(), "/home/aurora/sim_results/SFT_times.txt", 10);
    save_vector_as_txt(solver.GetValues(), "/home/aurora/sim_results/SFT_values.txt", 10);
}


