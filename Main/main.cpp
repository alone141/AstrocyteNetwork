#include <iostream>
#include <Utility.h>
#include <Perceptron.h>
#include <AstrocyteNetwork.h>
#include <initializer_list>
#include <vector>
consteval float fun(){

    perceptron::Perceptron <1, perceptron::ActivationFunctionEnum::Sigmoid> p;
    an::AstrocyteNetwork<2,2,1> an;
    auto input = std::to_array({10.0f,7.0f});
    auto backprop = std::to_array({10.0f});
    an.FeedForward(input);
    float output = an.GetOutputLayer()[0];
    an.Backpropagation(backprop,0.01);
    return output;
}
int main() {

    an::AstrocyteNetwork<2, 2, 1> network;
    std::vector<std::pair<std::array<float, 2>, std::array<float, 1>>> trainingData = {
        {{0.0f, 0.0f}, {0.0f}},
        {{0.0f, 1.0f}, {1.0f}},
        {{1.0f, 0.0f}, {1.0f}},
        {{1.0f, 1.0f}, {0.0f}}
    };
    // Training parameters
    float learningRate = 0.01f;
    int epochs = 1000;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalError = 0.0f;

        for (auto& [input, targetOutput] : trainingData) {
            // Forward pass
            network.FeedForward(input);

            // Backpropagation
            network.Backpropagation(targetOutput, learningRate);

            // Calculate error for monitoring
            auto output = network.GetOutputLayer();
            totalError += 0.5f * (targetOutput[0] - output[0]) * (targetOutput[0] - output[0]); // Mean Squared Error
        }

        // Print error every 1000 epochs
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Total Error: " << totalError << std::endl;
        }
    }

    std::cout << "\nTesting the trained network:\n";
    for (auto& [input, targetOutput] : trainingData) {
        network.FeedForward(input);
        auto output = network.GetOutputLayer();
        std::cout 
        << "Input: {" << input[0] << ", " << input[1] 
        << "} -> Predicted Output: " << output[0] 
        << ", Target Output: " << targetOutput[0] << std::endl;
    }
    

    return 0;
}