#include <cmath>
#include <array>

#include <utils.h>
#include <ProgressBar.h>
#include <solvers/RK4.h>


// tau = 3.08e19 s

double H0 = 69; // 1/tau
double OmM0 = 0.3;
double OmR0 = 5e-5;
double V0 = 1 - OmM0 - OmR0;
int n = -2;

double dt = 2e-10;
unsigned int K = 2e8;

// state = {a, phi, psi}
Vector<3> quintessence(const Vector<3> &state, double t) {
    Vector<3> result;
    result[0] = H0 * sqrt(OmM0/state[0] + OmR0/state[0]/state[0] + state[0]*state[0]*(state[2]*state[2]/2 + V0*pow(state[1], n)));
    result[1] = state[2];
    result[2] = -3*result[0]/state[0]*state[2] - V0*pow(state[1], n);
    return result;
}


int main(int argc, char* argv[]) {

    RK4<3> solver(quintessence, Vector<3>{1, 1, 0}, 0);

    DotProgressBar progress_bar(K, "step(s)", int(K/100), 70);
    for (unsigned int k = 0; k < K; k++) {
        solver.MakeStep(-dt);
        progress_bar.step();

        if (std::isnan(solver.GetCurrentValue()[0])) {
            std::cout << "Big Bang reached! Solving stopped." << std::endl;
            break;
        }
    }

    save_vector_as_txt(solver.GetTimes(), "/home/aurora/sim_results/quintessence_times.txt", 10);
    save_vector_as_txt(solver.GetValues(), "/home/aurora/sim_results/quintessence_values.txt", 10);
}


