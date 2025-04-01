#include <iostream>
#include <Perceptron.h>
#include <Utility.h>
int main() {
    Perceptron<1> p;
    std::cout << "Hello, World!" << std::endl;
    std::cin.get(); // Wait for user input before closing the console window
    constexpr int a = utility::factorial(5); // Example usage of the Sigmoid function
    constexpr int b = utility::pow(2, 3); // Example usage of the Sigmoid function
    return 0;
}