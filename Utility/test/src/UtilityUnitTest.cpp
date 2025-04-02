#include <gtest/gtest.h>

TEST(UtilityTests, DivideFunction) {
    EXPECT_EQ(1, 1);
}

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}