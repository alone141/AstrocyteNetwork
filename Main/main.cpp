#include <iostream>
#include <Utility.h>
#include <Perceptron.h>
#include <AstrocyteNetwork.h>
#include <initializer_list>
consteval float fun(){

    perceptron::Perceptron <1, perceptron::ActivationFunctionEnum::Sigmoid> p;
/*     an::AstrocyteNetwork<1,1,1> an;
    constexpr auto output = an.GetOutputLayer(); */

    return p.ActivationFunction(2);
}
int main() {
    perceptron::Perceptron <1, perceptron::ActivationFunctionEnum::Sigmoid> p;
    p.CalculateOutput();
    constexpr float c = fun();
    std::cout << "Hello, World!" << fun() << std::endl;

    an::AstrocyteNetwork<2,2,1> an;
    auto input = std::to_array({6.0f,6.0f});
    an.FeedForward(input);
    auto output = an.GetOutputLayer();


    constexpr int factorial = utility::factorial(5); 
    constexpr int power = utility::pow(2, 3); 
    constexpr float power2 = utility::pow(2.5, 3.5); 
    constexpr float e = utility::exp(1.5);
    constexpr float ln = utility::ln(10);

    constexpr float sig = utility::af::Sigmoid(8.64f);
    std::cout << "Hello, World!" << power2 << std::endl;



    return 0;
}