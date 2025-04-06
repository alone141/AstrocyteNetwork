#include <iostream>
#include <Utility.h>
#include <Perceptron.h>
#include <AstrocyteNetwork.h>
#include <initializer_list>
#include <memory>
consteval float fun(){

    perceptron::Perceptron <1, perceptron::ActivationFunctionEnum::Sigmoid> p(5);
    an::AstrocyteNetwork<2,2,1> an;
    auto input = std::to_array({10.0f,7.0f});
    auto backprop = std::to_array({10.0f});
    an.FeedForward(input);
    float output = an.GetOutputLayer()[0];
    an.Backpropagation(backprop,0.01);
    return output;
}
constexpr int cround(double x) {
    return (x >= 0.0) ? int(x + 0.5) : int(x - 0.5);
  }

consteval void TestConsEval(){
    float learningRate = 0.2f;
    int epochs = 3000;
    std::array<std::pair<std::array<float, 3>, std::array<float, 2>>, 4> trainingData = {
        std::make_pair(std::array<float, 3>{0.0f, 0.0f, 1.0f}, std::array<float, 2>{0.0f,0.0f}),
        std::make_pair(std::array<float, 3>{0.0f, 1.0f, 1.0f}, std::array<float, 2>{1.0f,0.0f}),
        std::make_pair(std::array<float, 3>{1.0f, 0.0f, 1.0f}, std::array<float, 2>{1.0f,0.0f}),
        std::make_pair(std::array<float, 3>{1.0f, 1.0f, 0.5f}, std::array<float, 2>{0.0f,1.0f})
    };

    an::AstrocyteNetwork<3, 5, 2> network;
    //network.Train(trainingData,learningRate,epochs);
    //return network;
}

int main() {
    TestConsEval();
    float learningRate = 0.2f;
    int epochs = 3000;
    std::array<std::pair<std::array<float, 3>, std::array<float, 2>>, 4> trainingData = {
        std::make_pair(std::array<float, 3>{0.0f, 0.0f, 1.0f}, std::array<float, 2>{0.0f,0.0f}),
        std::make_pair(std::array<float, 3>{0.0f, 1.0f, 1.0f}, std::array<float, 2>{1.0f,0.0f}),
        std::make_pair(std::array<float, 3>{1.0f, 0.0f, 1.0f}, std::array<float, 2>{1.0f,0.0f}),
        std::make_pair(std::array<float, 3>{1.0f, 1.0f, 0.5f}, std::array<float, 2>{0.0f,1.0f})
    };
    
    an::AstrocyteNetwork<3, 5, 2> network;
    std::cout << " --------------------------------- Before Training --------------------------------- " << std::endl;
    network.Print();
    std::cout << " --------------------------------- Before Training --------------------------------- " << std::endl;

    network.Train(trainingData,learningRate,epochs);

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

        if(cround(output[0]) != cround(targetOutput[0]) or 
        cround(output[1]) != cround(targetOutput[1])){
            std::cout << "ERROR" << std::endl;
        }
    }


    constexpr auto a = utility::random::GetRandomUniform();
    //constexpr auto result = TestConsEval();
    //std::cout << "Calculated Output 1: " << result[0] << std::endl;
    //std::cout << "Calculated Output 2: " << result[1] << std::endl;

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