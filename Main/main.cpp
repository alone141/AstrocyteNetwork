#include <iostream>
#include <Perceptron.h>
#include <Utility.h>
consteval float fun(){

    perceptron::Perceptron <1, perceptron::ActivationFunctionEnum::Sigmoid> p;
    return p.ActivationFunction(2);
}
int main() {
    perceptron::Perceptron <1, perceptron::ActivationFunctionEnum::Sigmoid> p;
    p.CalculateOutput();
    constexpr float c = fun();
    std::cout << "Hello, World!" << fun() << std::endl;

    
    constexpr int factorial = utility::factorial(5); 
    constexpr int power = utility::pow(2, 3); 
    constexpr float e = utility::exp(1);
    constexpr float ln = utility::ln(10);
    std::cout << "Hello, World!" << ln << std::endl;
    



    std::cin.get(); // Wait for user input before closing the console window
    return 0;
}