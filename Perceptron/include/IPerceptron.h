#ifndef IPERCEPTRON_H
#define IPERCEPTRON_H
namespace perceptron {
    class IPerceptron {
        public:
            virtual constexpr ~IPerceptron() = default; // Destructor
            virtual constexpr float CalculateOutput() = 0; // Calculate output based on input
            virtual constexpr float ActivationFunction(float input) = 0;
            virtual constexpr float GetOutput() const = 0;
            virtual constexpr void SetInput(unsigned int inputIndex, float inputValue) = 0;
        };
        
}

#endif // IPERCEPTRON_H