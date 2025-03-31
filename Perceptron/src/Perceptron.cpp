#include <Perceptron.h>
#include <iostream>


Perceptron::Perceptron(int bias, int weight) : bias(bias), weight(weight) {
    std::cout << "Perceptron created!" << std::endl;
}