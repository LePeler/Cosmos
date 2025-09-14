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


double LCDM_E(double z, const Vector<3> &params) {
    double Om0 = params(0);
    double Or0 = params(1);
    double a = 1/(1.0+z);
    return sqrt(Om0*(1+z)*(1+z)*(1+z) + Or0*(1+z)*(1+z)*(1+z)*(1+z) + (1-Om0-Or0));
}

double func(double E, double z, const Vector<3> &params) {
    double Om0 = params(0);
    double Or0 = params(1);
    double b1 = params(2);
    return E*E - Om0*(1+z)*(1+z)*(1+z) - Or0*(1+z)*(1+z)*(1+z)*(1+z) - (1-Om0-Or0)*pow(E, 2*b1);
}

double deriv(double E, double z, const Vector<3> &params) {
    double Om0 = params(0);
    double Or0 = params(1);
    double b1 = params(2);
    return 2*E - (1-Om0-Or0)*2*b1*pow(E, 2*b1-1.0);
}



int main(int argc, char* argv[]) {

    double z = 5.0;

    Vector<3> params;
    params << 0.3, 5e-5, 1.0;

    double E = chebyshev_root([z, params](double E) {return func(E, z, params);},
                              [z, params](double E) {return deriv(E, z, params);},
                              LCDM_E(z, params));

    std::cout << E << std::endl;

    return 0;
}


