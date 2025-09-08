#include <cmath>
#include <array>

#include <mpi.h>

#include <utils.h>
#include <Monitor.h>
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


bool prior(const Vector<3> &params) {
    return (50.0 < params(0)) && (params(0) < 100.0) && (0.0 < params(1)) && (params(1) < 1.0);
}

Vector<0> Model(const Vector<0> &state, double z, const Vector<3> &params) {
    return Vector<0>{};
}

Vector<0> GetY0(const Vector<3> &params) {
    return Vector<0>{};
}

double GetH(const Vector<0> &state, double z, const Vector<3> &params) {
    double H0 = params(0);
    double Om0 = params(1);
    double a = 1/(1.0+z);
    return H0 * sqrt(Om0/a/a/a + Or0/a/a/a/a + (1-Om0-Or0));
}


int main(int argc, char* argv[]) {
    // MPI setup
    MPI_Init(&argc, &argv);

    int proc;
    int num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // setup likelihoods
    std::vector<std::shared_ptr<LikelihoodBase<3>>> likelihoods;
    //likelihoods.push_back(std::make_shared<CC<3>>("/home/aurora/university/ISSA/MCMC_Data/CC"));
    likelihoods.push_back(std::make_shared<SN1a<3>>("/home/aurora/university/ISSA/MCMC_Data/SN1a", 2));
    //likelihoods.push_back(std::make_shared<BAO<3>>("/home/aurora/university/ISSA/MCMC_Data/BAO", 0, 1));

    CombinedLikelihood<3, 0> combined_likelihood(likelihoods, prior, Model, GetY0, GetH);

    // define initial walker states
    Vector<3> mu_init;
    mu_init << 69.0, 0.3, -19.0;
    Matrix<3> sigma_init;
    sigma_init << 3.0, 0.0, 0.0,
                  0.0, 0.1, 0.0,
                  0.0, 0.0, 1.0;
    Vector<3> z;
    std::normal_distribution<double> distN01(0.0, 1.0);
    std::mt19937 randgen(std::random_device{}());
    const unsigned int W = 50;
    std::array<Vector<3>, W> init_states;
    for (unsigned int w = 0; w < W; w++) {
        z << distN01(randgen), distN01(randgen), distN01(randgen);
        init_states[w] = mu_init + sigma_init * z;
    }

    // instantiate MCMC sampler
    std::function<double(const Vector<3> &)> log_likelihood = [&](const Vector<3> &params) {return combined_likelihood.log_likelihood(params);};
    MCMC2<3, W> sampler(proc, num_procs, log_likelihood, init_states, 1.0, 2.0);

    std::cout << "running burn-in ..." << std::endl;

    // instantiate burn-in monitor
    Monitor burn_in_monitor({"RHat", "ESS"},
        [&]() {std::pair<double, double> RHat_and_ESS = sampler.GetRHatAndESS();
            return std::vector<double>{RHat_and_ESS.first, RHat_and_ESS.second};},
        {[](double RHat) {return (RHat < 1.03);}, [](double ESS) {return (ESS > 20*W);}}, 100);

    // run burn-in
    while (true) {
        sampler.MakeIter();
        if (proc == 0) {
            if (burn_in_monitor()) {
                break;
            }
        }
    }
    sampler.Reset();

    std::cout << "running production ..." << std::endl;

    // instantiate production_monitor
    Monitor production_monitor({"RHat", "ESS"},
        [&]() {std::pair<double, double> RHat_and_ESS = sampler.GetRHatAndESS();
            return std::vector<double>{RHat_and_ESS.first, RHat_and_ESS.second};},
        {[](double RHat) {return (RHat < 1.01);}, [](double ESS) {return (ESS > 100*W);}}, 100);

    // run MCMC production
    while (true) {
        sampler.MakeIter();
        if (proc == 0) {
            if (production_monitor()) {
                break;
            }
        }
    }

    if (proc == 0) {
        std::cout << "acceptance_rate: " << sampler.GetAcceptanceRate() << std::endl;
    }

    fs::path out_path("/home/aurora/sim_results/MCMC_LCDM_SN1a_2.txt");
    sampler.SaveSample(out_path, true);

    MPI_Finalize();
    return 0;
}


