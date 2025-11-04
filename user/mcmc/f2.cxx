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


// params {H0, Om0, b2_inv, M}


bool range_prior(const Vector<4> &params) {
    return (50.0 < params(0)) && (params(0) < 100.0)
        && (0.0 < params(1)) && (params(1) < 1.0)
        && (0.0 < params(2));
}

double log_prior_SH0ES(const Vector<4> &params) {
    double H0_SH0ES = 73.04;
    double H0_sigma_SH0ES = 1.04;
    return -(params(0) - H0_SH0ES) /(H0_sigma_SH0ES*H0_sigma_SH0ES) *(params(0) - H0_SH0ES) /2;
}

double log_prior_H0liCOW(const Vector<4> &params) {
    double H0_H0liCOW = 73.3;
    double H0_sigma_H0liCOW = 1.75;
    return -(params(0) - H0_H0liCOW) /(H0_sigma_H0liCOW*H0_sigma_H0liCOW) *(params(0) - H0_H0liCOW) /2;
}

double log_prior_TRGB(const Vector<4> &params) {
    double H0_TRGB = 69.8;
    double H0_sigma_TRGB = 1.71;
    return -(params(0) - H0_TRGB) /(H0_sigma_TRGB*H0_sigma_TRGB) *(params(0) - H0_TRGB) /2;
}

Vector<0> Model(const Vector<0> &state, double z, const Vector<4> &params) {
    return Vector<0>{};
}

Vector<0> GetY0(const Vector<4> &params) {
    return Vector<0>{};
}

double LCDM_E(double z, const Vector<4> &params) {
    double H0 = params(0);
    double Om0 = params(1);
    double Or0 = 4.1534e-1 /H0/H0;
    return sqrt(Om0*pow(1+z, 3) + Or0*pow(1+z, 4) + (1-Om0-Or0));
}

double func(double E, double z, const Vector<4> &params) {
    double H0 = params(0);
    double Om0 = params(1);
    double b2_inv = params(2);
    double b2 = 1/b2_inv;
    double Or0 = 4.1534e-1 /H0/H0;
    double beta = (1 - Om0 - Or0) / ((b2 + 1)*exp(-b2) - 1);
    return E*E - Om0*pow(1+z, 3) - Or0*pow(1+z, 4) - beta*((1+b2*E)*exp(-b2*E) - 1);
}

double deriv(double E, double z, const Vector<4> &params) {
    double H0 = params(0);
    double Om0 = params(1);
    double b2_inv = params(2);
    double b2 = 1/b2_inv;
    double Or0 = 4.1534e-1 /H0/H0;
    double beta = (1 - Om0 - Or0) / ((b2 + 1)*exp(-b2) - 1);
    return 2*E + beta*b2*b2*E*exp(-b2*E);
}

double GetH(const Vector<0> &state, double z, const Vector<4> &params) {
    double H0 = params(0);
    double E_lcdm = LCDM_E(z, params);

    double root = chebyshev_root([z, params](double E) {return func(E, z, params);},
                            [z, params](double E) {return deriv(E, z, params);},
                            E_lcdm, 1e-6, 10);

    for (short i = 1; i < 15; i++) {
        if (!std::isnan(root)) {
            break;
        }
        root = chebyshev_root([z, params](double E) {return func(E, z, params);},
                            [z, params](double E) {return deriv(E, z, params);},
                            E_lcdm *pow(2, i), 1e-6, 10);
        if (!std::isnan(root)) {
            break;
        }
        root = chebyshev_root([z, params](double E) {return func(E, z, params);},
                            [z, params](double E) {return deriv(E, z, params);},
                            E_lcdm *pow(2, -i), 1e-6, 10);
    }
    if (std::isnan(root)) {
        std::cerr << "warning: could not find root" << std::endl;
        return NAN;
    }

    return H0 * root;
}

double GetRdrag(const Vector<4> &params) {
    double z_drag = 1059.94;
    double a_drag = 1.0/(1.0 + z_drag);
    double rho_b_0 = 0.02237;
    double rho_y_0 = 2.4697e-5;

    std::function<double(double a)> dr_da = [&](double a) {
        double z = 1.0/a - 1.0;
        double H_z = GetH(Vector<0>{}, z, params);

        double R_s = (3.0/4.0)*(rho_b_0/rho_y_0) *a;
        double c_s = PHYS_C /sqrt(3*(1 + R_s));

        return c_s /(a*a*H_z);
    };

    return gauss_legendre_16(dr_da, 1e-10, a_drag);
}


int main(int argc, char* argv[]) {
    // MPI setup
    MPI_Init(&argc, &argv);

    int proc;
    int num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // setup likelihoods
    std::vector<std::shared_ptr<LikelihoodBase<4>>> likelihoods;
    likelihoods.push_back(std::make_shared<CC<4>>("/home/aurora/university/ISSA/MCMC_Data/CC"));
    likelihoods.push_back(std::make_shared<SN1a<4>>("/home/aurora/university/ISSA/MCMC_Data/SN1a", 3));
    //likelihoods.push_back(std::make_shared<BAO<4>>("/home/aurora/university/ISSA/MCMC_Data/BAO", GetRdrag));

    CombinedLikelihood<4, 0> combined_likelihood(likelihoods, range_prior, Model, GetY0, GetH);

    // define initial walker states
    Vector<4> mu_init;
    mu_init << 73.0, 0.3, 0.0, -19.0;
    Matrix<4> sigma_init;
    sigma_init << 3.0, 0.0, 0.0, 0.0,
                  0.0, 0.1, 0.0, 0.0,
                  0.0, 0.0, 0.5, 0.0,
                  0.0, 0.0, 0.0, 1.0;
    Vector<4> z;
    std::normal_distribution<double> distN01(0.0, 1.0);
    std::mt19937 randgen(314159);
    const unsigned int W = 50;
    std::array<Vector<4>, W> init_states;
    for (unsigned int w = 0; w < W; w++) {
        z << distN01(randgen), distN01(randgen), distN01(randgen), distN01(randgen);
        init_states[w] = mu_init + sigma_init * z;
    }

    // instantiate MCMC sampler
    std::function<double(const Vector<4> &)> log_likelihood = [&](const Vector<4> &params)
    {
        return combined_likelihood.log_likelihood(params) + log_prior_H0liCOW(params);
    };
    MCMC2<4, W> sampler(proc, num_procs, log_likelihood, init_states, 1.5);

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

    fs::path out_path("/home/aurora/mcmc_results/f2_CC_SN1a(H0liCOW).txt");
    sampler.SaveSample(out_path, true);

    MPI_Finalize();
    return 0;
}


