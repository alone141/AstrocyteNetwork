#ifndef IPERCEPTRON_H
#define IPERCEPTRON_H

class IPerceptron {
public:
    virtual ~IPerceptron() = default; // Destructor
    virtual int GetBias() const = 0; // Getter for bias
    virtual int GetWeight() const = 0; // Getter for weight
    virtual void SetBias(int bias) = 0; // Setter for bias
    virtual void SetWeight(int weight) = 0; // Setter for weight
    virtual int CalculateOutput(int input) const = 0; // Calculate output based on input

};

#endif // IPERCEPTRON_H