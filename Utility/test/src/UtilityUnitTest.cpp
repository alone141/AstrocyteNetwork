#include <gtest/gtest.h>
#include <Utility.h>
#include <cmath>
#include <initializer_list>
#include <limits>

// Test for factorial function
TEST(FactorialTest, HandlesPositiveInput) {
    constexpr int result1 = utility::factorial(0);  // 0! = 1
    constexpr int result2 = utility::factorial(1);  // 1! = 1
    constexpr int result3 = utility::factorial(5);  // 5! = 120
    EXPECT_EQ(result1, 1);
    EXPECT_EQ(result2, 1);
    EXPECT_EQ(result3, 120);
}

TEST(FactorialTest, HandlesNegativeInput) {
    constexpr int result1 = utility::factorial(-1);  // Negative input returns 0
    constexpr int result2 = utility::factorial(-10); // Negative input returns 0
    EXPECT_EQ(result1, 0);
    EXPECT_EQ(result2, 0);
}

TEST(FactorialTest, HandlesBigValues) {
    constexpr int result = utility::factorial(10); // 10! = 3628800
    EXPECT_EQ(result, 3628800);
}

// Test for pow function
TEST(PowTest, HandlesPositiveExponent) {
    constexpr float result1 = utility::pow(2.0f, 3);  
    constexpr float result2 = utility::pow(5.0f, 0);  
    constexpr float result3 = utility::pow(3.0f, 1);  
    EXPECT_EQ(result1, std::pow(2.0f, 3));
    EXPECT_EQ(result2, std::pow(5.0f, 0));
    EXPECT_EQ(result3, std::pow(3.0f, 1));
}

TEST(PowTest, HandlesNegativeExponent) {
    constexpr float result = utility::pow(2.0f, -3); // 2^-3 = 1 / 2^3
    EXPECT_EQ(result, 0.125f);
}

TEST(PowTest, HandlesZeroBase) {
    constexpr float result1 = utility::pow(0.0f, 3);  
    constexpr float result2 = utility::pow(0.0f, 0);  // 0^0 = 1 (by convention)
    EXPECT_EQ(result1, std::pow(0.0f, 3));
    EXPECT_EQ(result2, 1.0f);
}

TEST(PowTest, HandlesBigValues) {
    constexpr float result = utility::pow(5.0f, 10); 
    EXPECT_EQ(result, std::pow(5.0f, 10));
}

// Test for exp function
TEST(ExpTest, ApproximatesExponential) {
    constexpr float result1 = utility::exp(1.0f);  
    constexpr float result2 = utility::exp(0.0f);  
    constexpr float result3 = utility::exp(-1.0f); 
    EXPECT_NEAR(result1, std::exp(1.0f), 0.001f);
    EXPECT_NEAR(result2, std::exp(0.0f), 0.001f);
    EXPECT_NEAR(result3, std::exp(-1.0f), 0.001f);
}

TEST(ExpTest, HandlesBigValues) {
    constexpr float result = utility::exp(10.0f); 
    // e^10 = 22026.46..., at that size neighbouring floats are 0.002 apart,
    // so an absolute tolerance of 0.001 can't work. Compare in ULPs instead.
    EXPECT_FLOAT_EQ(result, std::exp(10.0f));
}

TEST(ExpTest, HandlesNegativeValues) {
    // The plain Taylor series returned -485 here: its alternating terms cancel each other out
    constexpr float result = utility::exp(-15.0f);
    EXPECT_FLOAT_EQ(result, std::exp(-15.0f));
}

TEST(ExpTest, HandlesOverflowAndUnderflow) {
    constexpr float tooBig = utility::exp(100.0f);
    constexpr float tooSmall = utility::exp(-200.0f);
    EXPECT_EQ(tooBig, std::numeric_limits<float>::infinity());
    EXPECT_EQ(tooSmall, 0.0f);
}

// Test for ln function
TEST(LnTest, ApproximatesNaturalLogarithm) {
    constexpr float result1 = utility::ln(2.71828f); 
    constexpr float result2 = utility::ln(1.0f);     
    constexpr float result3 = utility::ln(0.5f);     
    EXPECT_NEAR(result1, std::log(2.71828f), 0.001f);
    EXPECT_NEAR(result2, std::log(1.0f), 0.001f);
    EXPECT_NEAR(result3, std::log(0.5f), 0.001f);
}

TEST(LnTest, HandlesBigValues) {
    constexpr float result = utility::ln(1000.0f); 
    EXPECT_NEAR(result, std::log(1000.0f), 0.001f);
}

TEST(LnTest, HandlesSmallValues) {
    constexpr float result = utility::ln(0.001f);
    EXPECT_NEAR(result, std::log(0.001f), 0.001f);
}

TEST(SigmoidTest, MatchesStandardSigmoid) {
    constexpr float result1 = utility::af::Sigmoid(0.0f);
    constexpr float result2 = utility::af::Sigmoid(1.0f);
    constexpr float result3 = utility::af::Sigmoid(-20.0f);
    EXPECT_NEAR(result1, 0.5f, 0.0001f);
    EXPECT_NEAR(result2, 1.0f / (1.0f + std::exp(-1.0f)), 0.0001f);
    EXPECT_NEAR(result3, 1.0f / (1.0f + std::exp(20.0f)), 0.0001f);
}

// PseudoSigmoid is its own function, 0.5 * (x / (1 + |x|) + 1), not an approximation of the sigmoid
TEST(PseudoSigmoidTest, MatchesDefinition) {
    constexpr float result1 = utility::af::PseudoSigmoid(0.0f);  
    constexpr float result2 = utility::af::PseudoSigmoid(1.0f);  
    constexpr float result3 = utility::af::PseudoSigmoid(-1.0f); 
    EXPECT_FLOAT_EQ(result1, 0.5f);  // 0.5 * (0 / 1 + 1)
    EXPECT_FLOAT_EQ(result2, 0.75f); // 0.5 * (1 / 2 + 1)
    EXPECT_FLOAT_EQ(result3, 0.25f); // 0.5 * (-1 / 2 + 1)
}

TEST(PseudoSigmoidTest, IsSymmetricAroundHalf) {
    for (float x : {0.1f, 0.5f, 2.0f, 10.0f}) {
        EXPECT_NEAR(utility::af::PseudoSigmoid(-x), 1.0f - utility::af::PseudoSigmoid(x), 1e-6f);
    }
}

TEST(PseudoSigmoidTest, HandlesLargeInputs) {
    constexpr float result1 = utility::af::PseudoSigmoid(100.0f);
    constexpr float result2 = utility::af::PseudoSigmoid(-100.0f);
    // It approaches 0 and 1 slowly, like 1/x instead of e^-x: PseudoSigmoid(100) is still 0.995
    EXPECT_FLOAT_EQ(result1, 0.5f * (100.0f / 101.0f + 1.0f));
    EXPECT_FLOAT_EQ(result2, 0.5f * (-100.0f / 101.0f + 1.0f));
    EXPECT_LT(result1, 1.0f);
    EXPECT_GT(result2, 0.0f);
}

TEST(ReluTest, HandlesPositiveAndNegativeInputs) {
    constexpr float result1 = utility::af::Relu(1.0f);  
    constexpr float result2 = utility::af::Relu(-1.0f); 
    constexpr float result3 = utility::af::Relu(0.0f);  
    EXPECT_EQ(result1, 1.0f);  // ReLU(1) = 1
    EXPECT_EQ(result2, 0.0f);  // ReLU(-1) = 0
    EXPECT_EQ(result3, 0.0f);  // ReLU(0) = 0
}

TEST(ReluTest, HandlesLargeInputs) {
    constexpr float result1 = utility::af::Relu(1000.0f);  // Large positive input
    constexpr float result2 = utility::af::Relu(-1000.0f); // Large negative input
    EXPECT_EQ(result1, 1000.0f);  // ReLU(1000) = 1000
    EXPECT_EQ(result2, 0.0f);     // ReLU(-1000) = 0
}


TEST(TanhTest, ApproximatesTanh) {
    constexpr float result1 = utility::af::Tanh(0.5f);  
    constexpr float result2 = utility::af::Tanh(1.0f);  
    constexpr float result3 = utility::af::Tanh(-1.0f); 
    EXPECT_NEAR(result1, std::tanh(0.5f), 0.001f); // Tanh(0) = 0
    EXPECT_NEAR(result2, std::tanh(1.0f), 0.001f); // Tanh(1)
    EXPECT_NEAR(result3, std::tanh(-1.0f), 0.001f); // Tanh(-1)
}

TEST(TanhTest, HandlesLargeInputs) {
    constexpr float result1 = utility::af::Tanh(10.0f);   // Large positive input
    constexpr float result2 = utility::af::Tanh(-10.0f); // Large negative input
    EXPECT_NEAR(result1, 1.0f, 0.001f);  // Tanh(10) ≈ 1
    EXPECT_NEAR(result2, -1.0f, 0.001f); // Tanh(-10) ≈ -1
}

TEST(LeakyReluTest, HandlesPositiveAndNegativeInputs) {
    constexpr float result1 = utility::af::LeakyRelu(1.0f);  
    constexpr float result2 = utility::af::LeakyRelu(-1.0f); 
    constexpr float result3 = utility::af::LeakyRelu(0.0f);  
    EXPECT_EQ(result1, 1.0f);  // LeakyReLU(1) = 1
    EXPECT_EQ(result2, -0.01f); // LeakyReLU(-1) = -0.01 (default alpha = 0.01)
    EXPECT_EQ(result3, 0.0f);  // LeakyReLU(0) = 0
}

TEST(LeakyReluTest, HandlesLargeInputs) {
    constexpr float result1 = utility::af::LeakyRelu(1000.0f);  // Large positive input
    constexpr float result2 = utility::af::LeakyRelu(-1000.0f); // Large negative input
    EXPECT_EQ(result1, 1000.0f);       // LeakyReLU(1000) = 1000
    EXPECT_EQ(result2, -10.0f);        // LeakyReLU(-1000) = -10 (default alpha = 0.01)
}

// Backpropagation is only correct if every derivative matches its activation function.
// Compare each analytic derivative, written in terms of the output y = f(x), with a numerical one.
template <typename Function, typename Derivative>
void ExpectDerivativeMatches(Function function, Derivative derivative, std::initializer_list<float> points) {
    constexpr float h = 1e-3f;
    for (float x : points) {
        const float numerical = (function(x + h) - function(x - h)) / (2.0f * h);
        EXPECT_NEAR(derivative(function(x)), numerical, 1e-3f) << "at x = " << x;
    }
}

TEST(DerivativeTest, MatchesNumericalDerivative) {
    const auto points = {-3.0f, -1.0f, -0.3f, 0.2f, 0.7f, 2.5f};
    ExpectDerivativeMatches(utility::af::Linear,        utility::af::derivative::Linear,        points);
    ExpectDerivativeMatches(utility::af::Sigmoid,       utility::af::derivative::Sigmoid,       points);
    ExpectDerivativeMatches(utility::af::PseudoSigmoid, utility::af::derivative::PseudoSigmoid, points);
    ExpectDerivativeMatches(utility::af::Tanh,          utility::af::derivative::Tanh,          points);
    ExpectDerivativeMatches(utility::af::Relu,          utility::af::derivative::Relu,          points);
    ExpectDerivativeMatches([](float x) { return utility::af::LeakyRelu(x); },
                            [](float y) { return utility::af::derivative::LeakyRelu(y); }, points);
}

TEST(DerivativeTest, SigmoidDerivativeIsWrongForPseudoSigmoid) {
    // The bug this project used to have: at x = 0 the real slope of PseudoSigmoid is 0.5, y * (1 - y) gives 0.25
    constexpr float y = utility::af::PseudoSigmoid(0.0f);
    EXPECT_FLOAT_EQ(utility::af::derivative::PseudoSigmoid(y), 0.5f);
    EXPECT_FLOAT_EQ(utility::af::derivative::Sigmoid(y), 0.25f);
}

TEST(RandomTest, StaysInRange) {
    utility::random::PCG rng(1234);
    for (int i = 0; i < 100000; ++i) {
        const float value = utility::random::GetRandomUniform(rng, -0.5f, 0.5f);
        ASSERT_GE(value, -0.5f);
        ASSERT_LT(value, 0.5f);
    }
}

TEST(RandomTest, IsReproducible) {
    utility::random::PCG first(42, 7);
    utility::random::PCG second(42, 7);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(first(), second());
    }
}

TEST(RandomTest, StreamsAreDifferent) {
    utility::random::PCG first(42, 1);
    utility::random::PCG second(42, 2);
    int equalValues = 0;
    for (int i = 0; i < 100; ++i) {
        equalValues += (first() == second()) ? 1 : 0;
    }
    EXPECT_EQ(equalValues, 0);
}

TEST(RandomTest, WorksAtCompileTime) {
    constexpr float value = [] {
        utility::random::PCG rng(42);
        return utility::random::GetRandomUniform(rng);
    }();
    EXPECT_GE(value, -0.5f);
    EXPECT_LT(value, 0.5f);
}

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}