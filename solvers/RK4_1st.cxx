#include <solvers/RK4_1st.h>


double RK4_1st::ComputeStep(double dt) const {
    // make an RK4 step

    double k1 = RK4_1st::func(y, t);
    double k2 = RK4_1st::func(y + k1*dt/2, t + dt/2);
    double k3 = RK4_1st::func(y + k2*dt/2, t + dt/2);
    double k4 = RK4_1st::func(y + k3*dt, t + dt);

    return y + (k1 + 2*k2 + 2*k3 + k4)/6 * dt;
}


