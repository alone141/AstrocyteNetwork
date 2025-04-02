#include <gtest/gtest.h>
#include <Utility.h>
#include <cmath>
TEST(UtilityTests, FactorialFunction) {
    constexpr float factorialResult = utility::factorial(5);
    EXPECT_EQ(120, factorialResult);
}
TEST(UtilityTests, PowFunction) {
    constexpr float powResult = utility::pow(2.556, 3);
    EXPECT_FLOAT_EQ(std::pow(2.556, 3), powResult);
}
// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}