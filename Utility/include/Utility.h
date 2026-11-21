#ifndef UTILITY_H
#define UTILITY_H
#include <type_traits>
#include <Config.h>
#include <cstdint>
#include <limits>
// Utility.h - Header file for utility functions and classes
// Everything here is constexpr so it can run inside the compiler during compile-time training.
namespace utility
{
    inline constexpr double E   = 2.718281828459045235;
    inline constexpr double LN2 = 0.693147180559945309;

    constexpr float factorial(int x)
    {
        if(x < 0)
            return 0;
        else if(x == 0 || x == 1){
            return 1;
        }
        else{
            return x * factorial(x - 1);
        }

    }

    //pow function for where exponent is an integer
    template <typename BaseType, typename ExpType>
    requires std::is_arithmetic_v<BaseType> and std::is_integral_v<ExpType>
    constexpr float pow(BaseType base, ExpType exponent){
        if constexpr (std::is_signed_v<ExpType>) {
            if(exponent < 0)
                return 1.0f / utility::pow(base, -exponent); // x^-n = 1 / x^n
        }
        if(exponent == 0){
            return 1;
        }
        else{
            float result = base;
            for(ExpType i = 1; i < exponent; i++){
                result *= base;
            }
            return result;
        }
    }

    // e^x
    // A plain Taylor series is only accurate for small |x|: for negative x the terms alternate in sign
    // and cancel each other out (e^-15 used to come out as -485). So we:
    //   1. use e^-x = 1 / e^x to only ever sum positive terms
    //   2. split x into integer and fractional parts, e^x = e^n * e^f, so the series only sees f in [0, 1)
    //   3. do the arithmetic in double and round to float once at the end
    constexpr float exp(float x, int iterationDepth = config::MATH_ITERATION_DEPTH){
        if(x != x) return x; // NaN in, NaN out
        if(x > 88.72f)  return std::numeric_limits<float>::infinity(); // e^x no longer fits in a float
        if(x < -103.97f) return 0.0f;                                   // smaller than the smallest float

        const bool negative = x < 0.0f;
        const double magnitude = negative ? -static_cast<double>(x) : static_cast<double>(x);
        const int integerPart = static_cast<int>(magnitude);
        const double fractionalPart = magnitude - integerPart;

        // e^f = 1 + f + f^2/2! + ... ; each term is built from the previous one, no pow/factorial needed
        double term = 1.0;
        double fractionalExp = 1.0;
        for(int i = 1; i < iterationDepth; i++){
            term *= fractionalPart / i;
            fractionalExp += term;
        }

        double integerExp = 1.0;
        for(int i = 0; i < integerPart; i++){
            integerExp *= E;
        }

        const double result = integerExp * fractionalExp;
        return static_cast<float>(negative ? 1.0 / result : result);
    }

    // ln(x)
    // The series 2*atanh((x-1)/(x+1)) converges very slowly when x is far from 1 (ln(1000) used to be 5.25).
    // We first scale x into [1, 2) with powers of two: ln(m * 2^k) = ln(m) + k*ln(2).
    // For m in [1, 2), |(m-1)/(m+1)| <= 1/3 so every term is at least 9x smaller than the previous one.
    constexpr float ln(float x, int iterationDepth = config::MATH_ITERATION_DEPTH){
        if(x != x || x < 0.0f) return std::numeric_limits<float>::quiet_NaN();
        if(x == 0.0f) return -std::numeric_limits<float>::infinity();
        if(x == std::numeric_limits<float>::infinity()) return x;

        double mantissa = x;
        int exponent = 0;
        while(mantissa >= 2.0){ mantissa /= 2.0; exponent++; }
        while(mantissa < 1.0){ mantissa *= 2.0; exponent--; }

        const double y = (mantissa - 1.0) / (mantissa + 1.0);
        const double ySquared = y * y;
        double term = y;
        double sum = 0.0;
        for(int i = 0; i < iterationDepth; i++){
            sum += term / (2.0 * i + 1.0);
            term *= ySquared;
        }
        return static_cast<float>(2.0 * sum + exponent * LN2);
    }

    //pow function for where exponent is not an integer
    template <typename BaseType, typename ExpType>
    requires (std::is_arithmetic_v<BaseType> and not std::is_integral_v<ExpType>)
    constexpr float pow(BaseType base, ExpType exponent){
        return utility::exp(exponent * utility::ln(base));
    }
}

// Activation functions
namespace utility::af{
    constexpr float Sigmoid(float x) {
        return 1.0f / (1.0f + utility::exp(-x));
    }
    // Cheap sigmoid-like curve built from softsign: 0.5 * (x / (1 + |x|) + 1)
    // It has the same shape as the sigmoid (0.5 at 0, between 0 and 1) but it is NOT the sigmoid:
    // PseudoSigmoid(1) = 0.75 while Sigmoid(1) = 0.731, and it approaches 0 and 1 much more slowly.
    constexpr float PseudoSigmoid(float x) {
        return 0.5f * (x / (1.0f + ((x < 0.0f) ? -x : x)) + 1.0f);
    }
    constexpr float Relu(float x) {
        return x > 0.0f ? x : 0.0f;
    }

    // tanh(x) = 1 - 2 / (e^2x + 1). Unlike (e^x - e^-x) / (e^x + e^-x) this never computes inf / inf for large x
    constexpr float Tanh(float x) {
        return 1.0f - 2.0f / (utility::exp(2.0f * x) + 1.0f);
    }

    constexpr float Linear(float x) {
        return x;
    }

    constexpr float LeakyRelu(float x, float alpha = 0.01f) {
        return x > 0.0f ? x : alpha * x;
    }
}

// Derivatives of the activation functions.
// Backpropagation already knows each perceptron's output y = f(x), so every derivative is written in terms of y.
namespace utility::af::derivative{
    constexpr float Linear(float) {
        return 1.0f;
    }
    // d/dx sigmoid(x) = y * (1 - y)
    constexpr float Sigmoid(float y) {
        return y * (1.0f - y);
    }
    // d/dx PseudoSigmoid(x) = 0.5 / (1 + |x|)^2, which works out to 2 * (1 - y)^2 for x >= 0 and 2 * y^2 for x < 0
    constexpr float PseudoSigmoid(float y) {
        const float distanceToEdge = (y < 1.0f - y) ? y : 1.0f - y;
        return 2.0f * distanceToEdge * distanceToEdge;
    }
    // d/dx tanh(x) = 1 - y^2
    constexpr float Tanh(float y) {
        return 1.0f - y * y;
    }
    constexpr float Relu(float y) {
        return y > 0.0f ? 1.0f : 0.0f;
    }
    constexpr float LeakyRelu(float y, float alpha = 0.01f) {
        return y > 0.0f ? 1.0f : alpha;
    }
}

// PCG32 random number generator, https://www.pcg-random.org
// The seed is a fixed number (config::RANDOM_SEED) instead of __TIME__ so every build trains the exact same network.
namespace utility::random {
    struct PCG
    {
        typedef std::uint32_t result_type;

        // Same seeding as the reference pcg32_srandom_r. Different streams give independent sequences,
        // so every perceptron can have its own stream while sharing one global seed.
        constexpr PCG(std::uint64_t seed, std::uint64_t stream = 0)
        {
            state = 0;
            inc = (stream << 1u) | 1u;
            pcg32_random_r();
            state += seed;
            pcg32_random_r();
        }

        constexpr result_type operator()()
        {
            return pcg32_random_r();
        }

        static result_type constexpr min()
        {
            return std::numeric_limits<result_type>::min();
        }

        static result_type constexpr max()
        {
            return std::numeric_limits<result_type>::max();
        }

    private:
        std::uint64_t state = 0;
        std::uint64_t inc = 1;

        constexpr std::uint32_t pcg32_random_r()
        {
            std::uint64_t oldstate = state;
            // Advance internal state
            state = oldstate * 6364136223846793005ULL + inc;
            // Calculate output function (XSH RR), uses old state for max ILP
            std::uint32_t xorshifted = static_cast<std::uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
            std::uint32_t rot = static_cast<std::uint32_t>(oldstate >> 59u);
            return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
        }

    };

    // Uniform float in [lower, upper)
    constexpr float GetRandomUniform(PCG& rng, float lower = -0.5f, float upper = 0.5f)
    {
        // Keep the top 24 bits: a float has exactly 24 bits of precision, so this maps evenly onto [0, 1)
        const float randomValue = static_cast<float>(rng() >> 8) / 16777216.0f; // 2^24
        return lower + randomValue * (upper - lower);
    }
}

#endif // UTILITY_H
