#include <vector>
#include <functional>


// 4th order Runge Kutta solver for 1st order differential equations
class RK4_1st {

public:
    // constructor
    RK4_1st(std::function<double(double, double)> f, double y0, double t0) {
        func = f;
        y = y0;
        t = t0;
        times.push_back(t);
        values.push_back(y);
    }

    // destructor
    ~RK4_1st() = default;

    // compute a step of the solving algorithm
    double ComputeStep(double dt) const;

    // make a step of the solving algorithm
    void MakeStep(double dt) {
        t += dt;
        times.push_back(t);

        y = ComputeStep(dt);
        values.push_back(y);
    }

    // get the last time
    double GetCurrentTime() const {
        return t;
    }

    // get the last computed value
    double GetCurrentValue() const {
        return y;
    }

    // get all times
    std::vector<double> GetTimes() const {
        return times;
    }

    // get all computed values
    std::vector<double> GetValues() const {
        return values;
    }


private:
    // function that gives y'(y, t)
    std::function<double(double, double)> func;

    // vector that stores the times for which values were calculated
    std::vector<double> times;
    // vector that stores the computed values
    std::vector<double> values;

    // current time
    double t;
    // current value
    double y;
};


