#ifndef ACTIVATIONFUNCTIONENUM_H
#define ACTIVATIONFUNCTIONENUM_H

namespace perceptron {
    enum class ActivationFunctionEnum {
        None = -1,
        Linear,
        Sigmoid,
        ReLU,
        LeakyReLU,
        Tanh
        //Softmax,
        //ELU,
        //SELU,
        //Swish,
        //GELU
    };
    
}

#endif // ACTIVATIONFUNCTIONENUM_H