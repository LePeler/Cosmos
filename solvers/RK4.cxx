#include <solvers/RK4.h>

template class RK4<1>;
template class RK4<2>;
template class RK4<3>;
template class RK4<4>;
template class RK4<5>;

template<unsigned short N>
State<N> RK4<N>::ComputeStep(double dt) const {
    // make an RK4 step

    State<N> k1 = func(y, t);
    State<N> k2 = func(y + k1*dt/2, t + dt/2);
    State<N> k3 = func(y + k2*dt/2, t + dt/2);
    State<N> k4 = func(y + k3*dt, t + dt);

    return y + (k1 + k2*2 + k3*2 + k4)/6 * dt;
}


