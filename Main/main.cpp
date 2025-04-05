#include <iostream>
#include <Utility.h>
#include <Perceptron.h>
#include <AstrocyteNetwork.h>
#include <initializer_list>
consteval float fun(){

    perceptron::Perceptron <1, perceptron::ActivationFunctionEnum::Sigmoid> p;
    an::AstrocyteNetwork<2,2,1> an;
    auto input = std::to_array({10.0f,7.0f});
    an.FeedForward(input);
    float output = an.GetOutputLayer()[0];

    return output;
}
int main() {
    perceptron::Perceptron <1, perceptron::ActivationFunctionEnum::Sigmoid> p;
    p.CalculateOutput();
    //static_assert((int)fun() == 12);
    std::cout << "Hello, World!" << fun() << std::endl;
    
    constexpr float rand = utility::random::GetRandomUniform(2);
    std::cout << "Rand:" << rand << std::endl;

    an::AstrocyteNetwork<2,2,1> an;
    auto input = std::to_array({10.0f,7.0f});
    an.FeedForward(input);
    auto output = an.GetOutputLayer()[0];

    constexpr int factorial = utility::factorial(5); 
    constexpr int power = utility::pow(2, 3); 
    constexpr float power2 = utility::pow(2.5, 3.5); 
    constexpr float e = utility::exp(1.5);
    constexpr float ln = utility::ln(10);

    constexpr float sig = utility::af::Sigmoid(8.64f);
    std::cout << "Hello, World!" << power2 << std::endl;



    return 0;
}