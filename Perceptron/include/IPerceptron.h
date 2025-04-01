#ifndef IPERCEPTRON_H
#define IPERCEPTRON_H
namespace perceptron {
    class IPerceptron {
        public:
            virtual ~IPerceptron() = default; // Destructor
            virtual float CalculateOutput(float input) = 0; // Calculate output based on input
        
        };
        
}

#endif // IPERCEPTRON_H