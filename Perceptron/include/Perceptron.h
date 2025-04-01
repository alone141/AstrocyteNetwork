#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "IPerceptron.h"
#include <vector>
#include <utility>
#include <Config.h>
template<int inputCount>
class Perceptron : public IPerceptron {
public:
    Perceptron(int bias = 0) : bias(bias){

    }
    ~Perceptron() = default; 
    float CalculateOutput(float input) {
        float sum = 0.0f;
        for(int i = 0; i < inputCount; i++){
            sum += inputs[i] * weights[i];
        }
        sum += bias; // Add bias to the sum
        return 0;
    }

    float ActivationFunction(float input){

    }
private:
    int bias;  
    float weights[inputCount]; // Weights for each input
    float inputs[inputCount]; // Inputs to the perceptron
    float output; // Output of the perceptron
};

#endif