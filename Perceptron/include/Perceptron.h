#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "IPerceptron.h"

class Perceptron : public IPerceptron {
public:
    Perceptron(int bias = 0, int weight = 0);
    ~Perceptron() = default; 
private:
    int bias;  
    int weight; 
};

#endif