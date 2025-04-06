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
int main() {


    an::AstrocyteNetwork<2,2,1> an;
    auto input = std::to_array({5.0f,7.0f});
    an.FeedForward(input);
    float output = an.GetOutputLayer()[0];
    std::cout << "Before learning:" << output << std::endl;
    auto backprop = std::to_array({1.0f});
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
    return 0;
}