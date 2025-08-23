#include <vector>
#include <functional>

#include <State.h>


// 4th order Runge Kutta solver for 1st order differential equations
template<size_t N>
class RK4 {

public:
    // constructor
    RK4(std::function<State<N>(const State<N> &, double)> func, State<N> y0, double t0)
        :
        func_(func),
        y_(y0),
        t_(t0)
    {
        Y_.push_back(y_);
        T_.push_back(t_);
    }

    // destructor
    ~RK4() = default;

    // get the last time
    double GetCurrentTime() const {
        return t_;
    }

    // get the last computed value
    std::array<double, N> GetCurrentValue() const {
        return y_;
    }

    // get all times
    std::vector<double> GetTimes() const {
        return T_;
    }

    // get all computed values
    std::vector<std::array<double, N>> GetValues() const {
        std::vector<std::array<double, N>> result;
        for (State<N> val : Y_) {
            result.push_back(val);
        }
        return result;
    }

    // make a step of the solving algorithm
    void MakeStep(double dt) {
        t_ += dt;
        T_.push_back(t_);

        y_ = ComputeStep(dt);
        Y_.push_back(y_);
    }

    // call a loop over MakeStep for K steps at a fixed width
    void MakeSteps(double dt, unsigned int K) {
        for (unsigned int k = 0; k < K; k++) {
            MakeStep(dt);
        }
    }

    // call a loop over MakeStep for a vector of widths
    void MakeStepsVector(const std::vector<double> &dT) {
        for (double dt : dT) {
            MakeStep(dt);
        }
    }


private:
    // function that gives y'(y, t)
    std::function<State<N>(const State<N> &, double)> func_;

    // vector that stores the times for which values were calculated
    std::vector<double> T_;
    // vector that stores the computed values
    std::vector<State<N>> Y_;

    // current time
    double t_;
    // current value
    State<N> y_;

    // compute a step of the RK4 solving algorithm
    State<N> ComputeStep(double dt) const {

        State<N> k1 = func_(y_, t_);
        State<N> k2 = func_(y_ + k1*dt/2, t_ + dt/2);
        State<N> k3 = func_(y_ + k2*dt/2, t_ + dt/2);
        State<N> k4 = func_(y_ + k3*dt, t_ + dt);

        return y_ + (k1 + k2*2 + k3*2 + k4)/6 * dt;
    }

};


