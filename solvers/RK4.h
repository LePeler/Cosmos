#include <vector>
#include <functional>

#include <State.h>


// 4th order Runge Kutta solver for 1st order differential equations
template<unsigned short N>
class RK4 {

public:
    // constructor
    RK4(std::function<State<N>(State<N>, double)> f, State<N> y0, double t0) {
        func = f;
        y = y0;
        t = t0;
        Y.push_back(y);
        T.push_back(t);
    }

    // destructor
    ~RK4() = default;

    // get the last time
    double GetCurrentTime() const {
        return t;
    }

    // get the last computed value
    std::array<double, N> GetCurrentValue() const {
        return y;
    }

    // get all times
    std::vector<double> GetTimes() const {
        return T;
    }

    // get all computed values
    std::vector<std::array<double, N>> GetValues() const {
        std::vector<std::array<double, N>> result;
        for (State<N> val : Y) {
            result.push_back(val);
        }
        return result;
    }

    // make a step of the solving algorithm
    void MakeStep(double dt) {
        t += dt;
        T.push_back(t);

        y = ComputeStep(dt);
        Y.push_back(y);
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
    std::function<State<N>(State<N>, double)> func;

    // vector that stores the times for which values were calculated
    std::vector<double> T;
    // vector that stores the computed values
    std::vector<State<N>> Y;

    // current time
    double t;
    // current value
    State<N> y;

    // compute a step of the solving algorithm
    State<N> ComputeStep(double dt) const;
};


