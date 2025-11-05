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


// state {T, V}
// params {H0, Om0, T0, l, M}


bool range_prior(const Vector<5> &params) {
    return (50.0 < params(0)) && (params(0) < 100.0)
        && (0.0 < params(1)) && (params(1) < 1.0)
        && (0.0 < params(2)) && (params(2) < 1.0)
        && (0.0 < params(3));
}

Vector<2> Model(const Vector<2> &state, double z, const Vector<5> &params) {
    double T = state(0);
    double V = state(1);
    double H0 = params(0);
    double Om0 = params(1);
    double l = sqrt(3)*H0*params(3);
    double Or0 = 4.1534e-1 /H0/H0;

    double H = H0*sqrt(Om0*pow(1+z, 3) + Or0*pow(1+z, 4) + T + V);
    double dz_dt = -(z+1)*H;

    double T_dot = -6*H*T + l*sqrt(2*T)*V;
    double V_dot = -l*sqrt(2*T)*V;
    return Vector<2>{T_dot/dz_dt, V_dot/dz_dt};
}

Vector<2> GetY0(const Vector<5> &params) {
    double H0 = params(0);
    double Om0 = params(1);
    double T0 = params(2);
    double Or0 = 4.1534e-1 /H0/H0;

    double V0 = 1 - Om0 - Or0 - T0;

    return Vector<2>{T0, V0};
}

double GetH(const Vector<2> &state, double z, const Vector<5> &params) {
    double T = state(0);
    double V = state(1);
    double H0 = params(0);
    double Om0 = params(1);
    double Or0 = 4.1534e-1 /H0/H0;

    return H0*sqrt(Om0*pow(1+z, 3) + Or0*pow(1+z, 4) + T + V);
}


int main(int argc, char* argv[]) {
    // MPI setup
    MPI_Init(&argc, &argv);

    int proc;
    int num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // setup likelihoods
    std::vector<std::shared_ptr<LikelihoodBase<5>>> likelihoods;
    likelihoods.push_back(std::make_shared<CC<5>>(SRC_DIR / "data/CC"));
    likelihoods.push_back(std::make_shared<SN1a<5>>(SRC_DIR / "data/SN1a", 4));

    CombinedLikelihood<5, 2> combined_likelihood(likelihoods, range_prior, Model, GetY0, GetH);

    // define initial walker states
    Vector<5> mu_init;
    mu_init << 73.0, 0.3, 0.0, 0.0, -19.0;
    Matrix<5> sigma_init;
    sigma_init << 2.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.1, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.3, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.5, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.5;
    Vector<5> z;
    std::normal_distribution<double> distN01(0.0, 1.0);
    std::mt19937 randgen(314159);
    const unsigned int W = 50;
    std::array<Vector<5>, W> init_states;
    for (unsigned int w = 0; w < W; w++) {
        z << distN01(randgen), distN01(randgen), distN01(randgen), distN01(randgen), distN01(randgen);
        init_states[w] = mu_init + sigma_init * z;
    }

    // instantiate MCMC sampler
    std::function<double(const Vector<5> &)> log_likelihood = [&](const Vector<5> &params)
    {
        return combined_likelihood.log_likelihood(params);
    };
    MCMC2<5, W> sampler(proc, num_procs, log_likelihood, init_states, 1.5);

    bool burn_in_done = false;
    // instantiate burn-in monitor
    std::shared_ptr<Monitor> burn_in_monitor = nullptr;
    if (proc == 0) {
        std::cout << "running burn-in ..." << std::endl;
        burn_in_monitor = std::make_shared<Monitor>(
            std::vector<std::string>{"RHat", "ESS"},
            [&]() {std::pair<double, double> RHat_and_ESS = sampler.GetRHatAndESS();
                    return std::vector<double>{RHat_and_ESS.first, RHat_and_ESS.second};},
            std::vector<std::function<bool(double)>>{[](double RHat) {return (RHat < 1.03);},
                                                     [](double ESS) {return (ESS > 20*W);}},
            100);
    }

    // run burn-in
    while (true) {
        sampler.MakeIter();
        if (proc == 0) {
            burn_in_done = burn_in_monitor->check();
        }
        MPI_Bcast(&burn_in_done, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        if (burn_in_done) {
            break;
        }
    }
    sampler.Reset();

    bool production_done = false;
    // instantiate production monitor
    std::shared_ptr<Monitor> production_monitor = nullptr;
    if (proc == 0) {
        std::cout << "running production ..." << std::endl;
        production_monitor = std::make_shared<Monitor>(
            std::vector<std::string>{"RHat", "ESS"},
            [&]() {std::pair<double, double> RHat_and_ESS = sampler.GetRHatAndESS();
                    return std::vector<double>{RHat_and_ESS.first, RHat_and_ESS.second};},
            std::vector<std::function<bool(double)>>{[](double RHat) {return (RHat < 1.01);},
                                                     [](double ESS) {return (ESS > 100*W);}},
            100);
    }

    // run MCMC production
    while (true) {
        sampler.MakeIter();
        if (proc == 0) {
            production_done = production_monitor->check();
        }
        MPI_Bcast(&production_done, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        if (production_done) {
            break;
        }
    }

    if (proc == 0) {
        std::cout << "acceptance_rate: " << sampler.GetAcceptanceRate() << std::endl;
    }

    fs::path out_path(SRC_DIR / "examples/quintessence_output.txt");
    sampler.SaveSample(out_path, true);

    MPI_Finalize();
    return 0;
}


