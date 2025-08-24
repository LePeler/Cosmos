#include <cmath>
#include <array>

#include <utils.h>
#include <ProgressBar.h>
#include <solvers/RK4.h>


// tau = 3.08e19 s

double H0 = 69; // 1/tau
double OmM0 = 0.3;
double OmR0 = 5e-5;
double OmL0 = 1 - OmM0 - OmR0;

double dt = 1e-10;
unsigned int K = 2e8;

Vector<1> LCDM(const Vector<1> &a, double t) {
    Vector<1> result;
    result[0] = H0 * sqrt(OmM0/a[0] + OmR0/a[0]/a[0] + OmL0*a[0]*a[0]);
    return result;
}


int main(int argc, char* argv[]) {

    RK4<1> solver(LCDM, Vector<1>{1}, 0);

    DotProgressBar progress_bar(K, "step(s)", int(K/100), 70);
    for (unsigned int k = 0; k < K; k++) {
        solver.MakeStep(-dt);
        progress_bar.step();

        if (std::isnan(solver.GetCurrentValue()[0])) {
            std::cout << "Big Bang reached! Solving stopped." << std::endl;
            break;
        }
    }

    save_vector_as_txt(solver.GetTimes(), "/home/aurora/sim_results/LCDM_times.txt", 10);
    save_vector_as_txt(solver.GetValues(), "/home/aurora/sim_results/LCDM_values.txt", 10);
}


