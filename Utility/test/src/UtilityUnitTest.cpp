#include <gtest/gtest.h>
#include <Utility.h>
#include <cmath>

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
    constexpr float result = utility::pow(2.0f, -3); // Negative exponent returns 0 (custom behavior)
    EXPECT_EQ(result, 0.0f);
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
    EXPECT_NEAR(result, std::exp(10.0f), 0.001f);
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

TEST(SigmoidTest, ApproximatesSigmoid) {
    constexpr float result1 = utility::af::Sigmoid(0.0f);  
    constexpr float result2 = utility::af::Sigmoid(1.0f);  
    constexpr float result3 = utility::af::Sigmoid(-1.0f); 
    EXPECT_NEAR(result1, 0.5f, 0.001f); // Sigmoid(0) = 0.5
    EXPECT_NEAR(result2, 1.0f / (1.0f + std::exp(-1.0f)), 0.001f);
    EXPECT_NEAR(result3, 1.0f / (1.0f + std::exp(1.0f)), 0.001f);
}

TEST(SigmoidTest, HandlesLargeInputs) {
    constexpr float result1 = utility::af::Sigmoid(100.0f);  // Large positive input
    constexpr float result2 = utility::af::Sigmoid(-100.0f); // Large negative input
    EXPECT_NEAR(result1, 1.0f, 0.001f); // Sigmoid(100) ≈ 1
    EXPECT_NEAR(result2, 0.0f, 0.001f); // Sigmoid(-100) ≈ 0
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

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}