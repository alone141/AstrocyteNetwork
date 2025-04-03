#include <iostream>
#include <Utility.h>
#include <Perceptron.h>
#include <AstrocyteNetwork.h>
consteval float fun(){

    perceptron::Perceptron <1, perceptron::ActivationFunctionEnum::Sigmoid> p;
    an::AstrocyteNetwork<2,2,1,1> an;
    return p.ActivationFunction(2);
}
int main() {
    perceptron::Perceptron <1, perceptron::ActivationFunctionEnum::Sigmoid> p;
    p.CalculateOutput();
    constexpr float c = fun();
    std::cout << "Hello, World!" << fun() << std::endl;

    
    constexpr int factorial = utility::factorial(5); 
    constexpr int power = utility::pow(2, 3); 
    constexpr float power2 = utility::pow(2.5, 3.5); 
    constexpr float e = utility::exp(1.5);
    constexpr float ln = utility::ln(10);
    std::cout << "Hello, World!" << power2 << std::endl;



    return 0;
}