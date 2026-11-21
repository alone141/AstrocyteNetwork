#include <AstrocyteNetwork.h>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace {
    constexpr int inputCount = 3;
    constexpr int hiddenCount = 5;
    constexpr int outputCount = 2;
    using Network = an::AstrocyteNetwork<inputCount, hiddenCount, outputCount>;

    constexpr float learningRate = 0.5f;
    constexpr int epochs = 1000;

    constexpr auto trainingData = std::to_array({
        std::make_pair(std::to_array({0.0f, 0.0f, 1.0f}), std::to_array({0.0f, 0.0f})),
        std::make_pair(std::to_array({0.0f, 1.0f, 1.0f}), std::to_array({1.0f, 0.0f})),
        std::make_pair(std::to_array({1.0f, 0.0f, 1.0f}), std::to_array({1.0f, 0.0f})),
        std::make_pair(std::to_array({1.0f, 1.0f, 0.5f}), std::to_array({0.0f, 1.0f}))
    });

    // consteval: this function can ONLY run inside the compiler.
    // The network uses heap memory (std::unique_ptr), which C++20 allows during constant evaluation
    // as long as it is freed before the evaluation ends. That is why the network itself cannot be
    // the result, only its weights, copied out into a plain std::array.
    consteval auto TrainAtCompileTime() {
        Network network;
        network.Train(trainingData, learningRate, epochs);
        return network.SerializeWeights();
    }

    // The compiler runs the whole training here. The binary only contains the resulting numbers.
    constexpr auto trainedWeights = TrainAtCompileTime();

    constexpr int Round(float x) {
        return (x >= 0.0f) ? static_cast<int>(x + 0.5f) : static_cast<int>(x - 0.5f);
    }
}

int main() {
    // No training at runtime: the network is built directly from the weights computed by the compiler
    Network network(trainedWeights);

    std::cout << "Network trained at compile time (" << epochs << " epochs):\n";
    int failures = 0;
    for (const auto& [input, targetOutput] : trainingData) {
        const auto output = network.Predict(input);
        const bool correct = Round(output[0]) == Round(targetOutput[0]) and Round(output[1]) == Round(targetOutput[1]);
        failures += correct ? 0 : 1;

        std::cout << "  Input: {" << input[0] << ", " << input[1] << ", " << input[2] << "}"
                  << " -> Predicted: {" << output[0] << ", " << output[1] << "}"
                  << ", Target: {" << targetOutput[0] << ", " << targetOutput[1] << "}"
                  << (correct ? "" : "  <-- WRONG") << '\n';
    }

    // For comparison: the same training at runtime
    Network runtimeNetwork;
    const auto start = std::chrono::steady_clock::now();
    runtimeNetwork.Train(trainingData, learningRate, epochs);
    const std::chrono::duration<double, std::milli> trainingTime = std::chrono::steady_clock::now() - start;
    std::cout << "\nThe same training at runtime takes " << trainingTime.count() << " ms\n";

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
