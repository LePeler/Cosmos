#include <solvers/RK4.h>

template class RK4<1>;
template class RK4<2>;
template class RK4<3>;
template class RK4<4>;
template class RK4<5>;

template<size_t N>
State<N> RK4<N>::ComputeStep(double dt) const {
    // make an RK4 step

    State<N> k1 = func_(y_, t_);
    State<N> k2 = func_(y_ + k1*dt/2, t_ + dt/2);
    State<N> k3 = func_(y_ + k2*dt/2, t_ + dt/2);
    State<N> k4 = func_(y_ + k3*dt, t_ + dt);

    return y_ + (k1 + k2*2 + k3*2 + k4)/6 * dt;
}


