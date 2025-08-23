#include <array>
#include <string>
#include <stdexcept>


template <size_t N>
struct State {
    // array to hold all dynamic variable values
    std::array<double, N> values;

    // constructors
    State() = default;
    State(const std::array<double, N> &vals) : values(vals) {}
    State(std::initializer_list<double> init) {
        if (init.size() != N) {
            throw std::invalid_argument(("State<" + std::to_string(N) + "> has to be initialized with " + std::to_string(N) + "values, not " + std::to_string(init.size())).c_str());
        }
        std::copy(init.begin(), init.end(), values.begin());
    }

    // convertion operator to std::array
    operator std::array<double, N>() const {
        return values;
    }

    // fast (but unsafe) element access
    double operator[](size_t n) const {
        return values[n];
    }

    // safe element access
    double at(size_t n) const {
        return values.at(n);
    }

    // element-wise addition
    State<N> operator+(const State<N> &rhs) const {
        State<N> result;
        for (size_t n = 0; n < N; n++)
            result.values[n] = values[n] + rhs.values[n];
        return result;
    }

    // element-wise subtraction
    State<N> operator-(const State<N> &rhs) const {
        State<N> result;
        for (size_t n = 0; n < N; n++)
            result.values[n] = values[n] - rhs.values[n];
        return result;
    }

    // element-wise multiplication
    template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    State<N> operator*(T scalar) const {
        State<N> result;
        for (size_t n = 0; n < N; n++)
            result.values[n] = values[n] * scalar;
        return result;
    }

    // element-wise division
    template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    State<N> operator/(T scalar) const {
        State<N> result;
        for (size_t n = 0; n < N; n++)
            result.values[n] = values[n] / scalar;
        return result;
    }
};


