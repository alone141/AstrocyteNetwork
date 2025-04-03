#ifndef IPERCEPTRON_H
#define IPERCEPTRON_H
namespace perceptron {
    class IPerceptron {
        public:
            virtual constexpr ~IPerceptron() = default; // Destructor
            virtual constexpr float CalculateOutput() = 0; // Calculate output based on input
        
        };
        
}

#endif // IPERCEPTRON_H