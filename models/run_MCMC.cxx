#include <cmath>
#include <array>

#include <utils.h>
#include <ProgressBar.h>
#include <solvers/RK4.h>
#include <likelihoods/SN1a.h>
#include <likelihoods/CC.h>
#include <likelihoods/BAO.h>
#include <likelihoods/CombinedLikelihood.h>
#include <samplers/MCMC2.h>


// tau = 3.08e19 s

// double H0 = 69.0; // 1/tau
// double Om0 = 0.3;
double Or0 = 5e-5;
// double OmL0 = 1.0 - Om0 - Or0;


Vector<0> Model(Vector<0> state, double z, Vector<2> params) {
    return Vector<0>{};
}

double GetH(Vector<0> state, double z, Vector<2> params) {
    double H0 = params(0);
    double Om0 = params(1);
    double a = 1/(1.0+z);
    return H0 * sqrt(Om0/a + Or0/a/a + (1-Om0-Or0)*a*a);
}


int main(int argc, char* argv[]) {

    std::vector<std::shared_ptr<LikelihoodBase<2>>> likelihoods;
    likelihoods.push_back(std::make_shared<CC<2>>("/home/aurora/university/ISSA/MCMC_Data/CC"));

    CombinedLikelihood<2, 0> combined_likelihood(likelihoods, Model, Vector<0>{}, GetH);

    std::cout << combined_likelihood.log_likelihood(Vector<2>{69.0, 0.3}) << std::endl;




    // DotProgressBar progress_bar(K, "step(s)", int(K/100), 70);
}


