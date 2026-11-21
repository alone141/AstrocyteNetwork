#include <gtest/gtest.h>
#include <AstrocyteNetwork.h>
#include <array>
#include <cmath>
#include <utility>

namespace {
    // XOR with only 2 hidden perceptrons: the smallest network that can solve it
    using XorNetwork = an::AstrocyteNetwork<2, 2, 1>;
    constexpr float xorLearningRate = 2.0f;
    constexpr int xorEpochs = 1000;

    constexpr auto xorData = std::to_array({
        std::make_pair(std::to_array({0.0f, 0.0f}), std::to_array({0.0f})),
        std::make_pair(std::to_array({0.0f, 1.0f}), std::to_array({1.0f})),
        std::make_pair(std::to_array({1.0f, 0.0f}), std::to_array({1.0f})),
        std::make_pair(std::to_array({1.0f, 1.0f}), std::to_array({0.0f}))
    });

    consteval auto TrainXorAtCompileTime() {
        XorNetwork network;
        network.Train(xorData, xorLearningRate, xorEpochs);
        return network.SerializeWeights();
    }
    constexpr auto xorWeights = TrainXorAtCompileTime();

    constexpr bool ToBit(float x) {
        return x >= 0.5f;
    }

    // Training AND inference both run inside the compiler.
    // If the network does not learn XOR, this file does not compile.
    consteval bool CompileTimeNetworkSolvesXor() {
        XorNetwork network(xorWeights);
        for (const auto& [input, target] : xorData) {
            if (ToBit(network.Predict(input)[0]) != ToBit(target[0])) {
                return false;
            }
        }
        return true;
    }
    static_assert(CompileTimeNetworkSolvesXor(), "The network trained at compile time did not learn XOR");

    template <typename Network, std::size_t weightCount, std::size_t inputCount, std::size_t outputCount>
    float Loss(const std::array<float, weightCount>& weights, const std::array<float, inputCount>& input,
               const std::array<float, outputCount>& target) {
        Network network(weights);
        const auto output = network.Predict(input);
        float loss = 0.0f;
        for (std::size_t k = 0; k < outputCount; ++k) {
            loss += 0.5f * (output[k] - target[k]) * (output[k] - target[k]);
        }
        return loss;
    }
}

TEST(CompileTimeTraining, WeightsSolveXorAtRuntime) {
    XorNetwork network(xorWeights);
    for (const auto& [input, target] : xorData) {
        EXPECT_NEAR(network.Predict(input)[0], target[0], 0.3f) << "input {" << input[0] << ", " << input[1] << "}";
    }
}

TEST(CompileTimeTraining, MatchesRuntimeTraining) {
    // The compiler and the CPU run the same code, so they should train (almost) the same network
    XorNetwork network;
    network.Train(xorData, xorLearningRate, xorEpochs);
    const auto runtimeWeights = network.SerializeWeights();
    for (std::size_t i = 0; i < runtimeWeights.size(); ++i) {
        EXPECT_NEAR(runtimeWeights[i], xorWeights[i], 1e-3f) << "weight " << i;
    }
}

TEST(FeedForward, MatchesHandCalculation) {
    // 1 input, 1 hidden, 1 output. Weight order: hidden {bias, w}, output {bias, w}
    an::AstrocyteNetwork<1, 1, 1> network(std::to_array({0.5f, 2.0f, -1.0f, 1.0f}));
    // hidden: 0.5 + 2 * 1 = 2.5           -> PseudoSigmoid(2.5)       = 0.5 * (2.5 / 3.5 + 1)          = 0.857143
    // output: -1 + 1 * 0.857143 = -0.142857 -> PseudoSigmoid(-0.142857) = 0.5 * (-0.142857 / 1.142857 + 1) = 0.4375
    // (when the input layer added +1 to every input, the hidden sum was 4.5 instead of 2.5)
    EXPECT_NEAR(network.Predict(std::to_array({1.0f}))[0], 0.4375f, 1e-5f);
}

TEST(Backpropagation, MatchesNumericalGradient) {
    // Gradient check: compare backpropagation's gradient with the slope of the loss measured numerically.
    // With learning rate 1, one Backpropagation step changes every weight by exactly -gradient.
    using Network = an::AstrocyteNetwork<3, 4, 2>;
    const auto input = std::to_array({0.3f, -0.7f, 0.9f});
    const auto target = std::to_array({1.0f, 0.0f});

    Network network;
    const auto weights = network.SerializeWeights();
    network.FeedForward(input);
    network.Backpropagation(target, 1.0f);
    const auto updatedWeights = network.SerializeWeights();

    constexpr float h = 1e-3f;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        auto plus = weights;
        auto minus = weights;
        plus[i] += h;
        minus[i] -= h;
        const float numericalGradient = (Loss<Network>(plus, input, target) - Loss<Network>(minus, input, target)) / (2.0f * h);
        const float backpropagationGradient = weights[i] - updatedWeights[i];
        EXPECT_NEAR(backpropagationGradient, numericalGradient, 1e-3f) << "weight " << i;
    }
}

TEST(Initialization, IsReproducibleAndNotSymmetric) {
    XorNetwork first;
    XorNetwork second;
    const auto weights = first.SerializeWeights();
    EXPECT_EQ(weights, second.SerializeWeights());

    for (float weight : weights) {
        EXPECT_GE(weight, -0.5f);
        EXPECT_LT(weight, 0.5f);
    }
    // Hidden perceptrons must start different from each other, or they stay identical forever
    EXPECT_NE(weights[0], weights[3]);
    EXPECT_NE(weights[1], weights[4]);
    EXPECT_NE(weights[2], weights[5]);
}

TEST(ResetNetwork, RestoresInitialWeights) {
    XorNetwork network;
    const auto initialWeights = network.SerializeWeights();
    network.Train(xorData, xorLearningRate, 100);
    ASSERT_NE(network.SerializeWeights(), initialWeights);

    network.ResetNetwork();
    EXPECT_EQ(network.SerializeWeights(), initialWeights);
}

TEST(ResetNetwork, CanStillLearn) {
    XorNetwork network;
    network.ResetNetwork();
    network.Train(xorData, xorLearningRate, xorEpochs);
    for (const auto& [input, target] : xorData) {
        EXPECT_EQ(ToBit(network.Predict(input)[0]), ToBit(target[0]));
    }
}

TEST(Serialization, RoundTrip) {
    XorNetwork network;
    network.Train(xorData, xorLearningRate, 100);
    const auto weights = network.SerializeWeights();

    XorNetwork copy(weights);
    EXPECT_EQ(copy.SerializeWeights(), weights);
    for (const auto& [input, target] : xorData) {
        EXPECT_EQ(copy.Predict(input), network.Predict(input));
    }
}

TEST(CopyConstructor, MakesIndependentCopy) {
    XorNetwork original;
    const auto originalWeights = original.SerializeWeights();

    XorNetwork copy(original);
    copy.Train(xorData, xorLearningRate, 100);

    EXPECT_EQ(original.SerializeWeights(), originalWeights);
    EXPECT_NE(copy.SerializeWeights(), originalWeights);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
