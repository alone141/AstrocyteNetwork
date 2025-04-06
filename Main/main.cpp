#include <iostream>
#include <Utility.h>
#include <Perceptron.h>
#include <AstrocyteNetwork.h>
#include <initializer_list>
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


void TrainXOR() {
    // Define the network: 2 input perceptrons, 2 hidden perceptrons, 1 output perceptron
    an::AstrocyteNetwork<3, 5, 2> network;
    std::cout << " --------------------------------- Before Training --------------------------------- " << std::endl;
    network.Print();
    std::cout << " --------------------------------- Before Training --------------------------------- " << std::endl;

    // XOR training data
    std::array<std::pair<std::array<float, 3>, std::array<float, 2>>, 4> trainingData = {
        std::make_pair(std::array<float, 3>{0.0f, 0.0f, 1.0f}, std::array<float, 2>{0.0f,0.0f}),
        std::make_pair(std::array<float, 3>{0.0f, 1.0f, 1.0f}, std::array<float, 2>{1.0f,0.0f}),
        std::make_pair(std::array<float, 3>{1.0f, 0.0f, 1.0f}, std::array<float, 2>{1.0f,0.0f}),
        std::make_pair(std::array<float, 3>{1.0f, 1.0f, 0.5f}, std::array<float, 2>{0.0f,1.0f})
    };

    // Training parameters
    float learningRate = 0.1f;
    int epochs = 10000;

    // Training loop
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
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", Total Error: " << totalError << std::endl;
        }
    }

    std::cout << " --------------------------------- After Training --------------------------------- " << std::endl;
    network.Print();
    std::cout << " --------------------------------- After Training --------------------------------- " << std::endl;

    // Test the network after training
    std::cout << "\nTesting the trained network:\n";
    for (auto& [input, targetOutput] : trainingData) {
        network.FeedForward(input);
        auto output = network.GetOutputLayer();
        std::cout << "Input: {" << input[0] << ", " << input[1] << "} -> Predicted Output: " << output[0] << " || " << output[1] 
                    << ", Target Output: " << targetOutput[0] << " || "<< targetOutput[1] << std::endl;
    }
}


int main() {


    TrainXOR();

    /*
    an::AstrocyteNetwork<2,2,1> an;
    auto input = std::to_array({5.0f,7.0f});
    an.FeedForward(input);
    float output = an.GetOutputLayer()[0];
    std::cout << "Before learning:" << output << std::endl;
    auto backprop = std::to_array({0.0f});
    an.Print();

    for (size_t i = 0; i < 10000; i++)
    {
        an.Backpropagation(backprop,0.1);
    }
    
    auto input2 = std::to_array({5.0f,7.0f});
    an.FeedForward(input2);
    float output2 = an.GetOutputLayer()[0];

    std::cout << std::endl << "------------------------------------------" << std::endl;
    std::cout << "After learning:" << output2 << std::endl;
    an.Print();

    */



    return 0;
}