#include <array>


template <unsigned short N>
struct State {
    // array to hold all dynamic variable values
    std::array<double, N> values;

    // constructors
    State() = default;
    State(const std::array<double, N> &vals) : values(vals) {}

    // convertion operator to std::array
    operator std::array<double, N>() const {
        return values;
    }

    // element-wise addition
    State<N> operator+(const State<N> &rhs) const {
        State<N> result;
        for (unsigned short n = 0; n < N; n++)
            result.values[n] = values[n] + rhs.values[n];
        return result;
    }

    // element-wise subtraction
    State<N> operator-(const State<N> &rhs) const {
        State<N> result;
        for (unsigned short n = 0; n < N; n++)
            result.values[n] = values[n] - rhs.values[n];
        return result;
    }

    // element-wise multiplication
    template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    State<N> operator*(T scalar) const {
        State<N> result;
        for (unsigned short n = 0; n < N; n++)
            result.values[n] = values[n] * scalar;
        return result;
    }

    // element-wise division
    template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    State<N> operator/(T scalar) const {
        State<N> result;
        for (unsigned short n = 0; n < N; n++)
            result.values[n] = values[n] / scalar;
        return result;
    }
};


