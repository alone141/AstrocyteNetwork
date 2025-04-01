#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "IPerceptron.h"
#include <vector>
#include <utility>
#include <Config.h>
#include <array>
#include <type_traits>
template<int inputCount, int inputCountWithBias = inputCount + 1>
class Perceptron : public IPerceptron {
    static_assert(inputCount > 0, "Input count must be greater than 0.");
    static_assert(inputCount +1 == inputCountWithBias, "Input with bias must be equal to input count + 1.");
public:
    constexpr Perceptron(int bias = 0) : bias(bias){
    }
    ~Perceptron() = default; 
    constexpr float CalculateOutput(float input) {
        float sum = 0.0f;
        for(int i = 0; i < inputCount; i++){
            sum += inputs[i] * weights[i];
        }
        sum += bias; // Add bias to the sum
        return 0;
    }

    constexpr float ActivationFunction(float input){
        return input;
    }
private:
    int bias;  
    std::array<float, inputCount+1> weights; //weights for each input and bias(+1)
    std::array<float, inputCount> inputs; 
    float output;
};

#endif