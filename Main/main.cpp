#include <iostream>
#include <Perceptron.h>
#include <Utility.h>
using namespace perceptron;
consteval float fun(){

    Perceptron <1, ActivationFunctionEnum::Sigmoid> p;
    return p.ActivationFunction(2);
}
int main() {
    //Perceptron<1> p;
    constexpr float c = fun();
    std::cout << "Hello, World!" << fun() << std::endl;
    std::cin.get(); // Wait for user input before closing the console window
    constexpr int a = utility::factorial(5); 
    constexpr int b = utility::pow(2, 3); 
    return 0;
}