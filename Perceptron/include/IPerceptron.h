#ifndef IPERCEPTRON_H
#define IPERCEPTRON_H

namespace perceptron {
    class IPerceptron {
        public:
            virtual constexpr ~IPerceptron() = default; // Destructor
            virtual constexpr float CalculateOutput() = 0; // Calculate output based on input
            virtual constexpr float ActivationFunction(const float input) const = 0;
            virtual constexpr float ActivationDerivative() const = 0; // Derivative of the activation function at the current output
            virtual constexpr float GetOutput() const = 0;
            virtual constexpr float GetWeight(const unsigned int weightIndex) const = 0;
            virtual constexpr void SetInput(const unsigned int inputIndex, float inputValue) = 0;
            virtual constexpr void SetWeight(const unsigned int weightIndex, float weightValue) = 0;
            virtual constexpr void ResetWeights() = 0; // Back to the perceptron's initial random weights
            virtual void Print() const = 0;
        };
        
}

#endif // IPERCEPTRON_H