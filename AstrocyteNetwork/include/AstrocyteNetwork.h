#ifndef ASTROCYTE_NETWORK_H
#define ASTROCYTE_NETWORK_H

#include <IAstrocyteNetwork.h>
#include <Perceptron.h>
#include <memory>



namespace an{
    template<int inputPerceptronCount, int hiddenPerceptronCount, int outputPerceptronCount, int layerCount = 1>
    class AstrocyteNetwork : public IAstrocyteNetwork {
        using inputPerceptronType = perceptron::Perceptron<1,perceptron::ActivationFunctionEnum::Linear>;
        using hiddenPerceptronType = perceptron::Perceptron<inputPerceptronCount, perceptron::ActivationFunctionEnum::ReLU>;
        using outputPerceptronType = perceptron::Perceptron<hiddenPerceptronCount,perceptron::ActivationFunctionEnum::Sigmoid>;
        public:
            constexpr AstrocyteNetwork(){
                static_assert(inputPerceptronCount > 0,  "Input layer perceptron count must be bigger than 0");
                static_assert(hiddenPerceptronCount > 0, "Hidden layer perceptron count must be bigger than 0");
                static_assert(layerCount > 0,            "Layer count must be bigger than 0");
                static_assert(outputPerceptronCount > 0, "Output layer perceptron count must be bigger than 0");

                InitializeNetwork();
            }
            constexpr ~AstrocyteNetwork() = default;

            constexpr void InitializeNetwork() override{
                for (auto &&perceptron : inputLayer)
                {
                    perceptron = std::make_unique<inputPerceptronType>();
                }
                for (auto &&perceptron : hiddenLayer)
                {
                    perceptron = std::make_unique<hiddenPerceptronType>();
                }
                for (auto &&perceptron : outputLayer)
                {
                    perceptron = std::make_unique<outputPerceptronType>();
                }
            }

            constexpr void FeedForward(std::array<float,inputPerceptronCount>& inputArray) {
                for (std::size_t i = 0; i < inputPerceptronCount; ++i) {
                    inputLayer[i]->SetInput(1, inputArray[i]); // assign each input to the corresponding perceptron
                    inputLayer[i]->CalculateOutput(); //this step might be unnecessary
                }
                for (std::size_t hiddenIndex = 0; hiddenIndex < hiddenPerceptronCount; ++hiddenIndex) {
                    for (std::size_t inputIndex = 0; inputIndex < inputPerceptronCount; ++inputIndex) {
                        hiddenLayer[hiddenIndex]->SetInput(inputIndex+1, inputLayer[inputIndex]->GetOutput());
                    }
                }
                
            }
            constexpr void ResetNetwork() override{

            }
        private:
            std::array<std::unique_ptr<perceptron::IPerceptron>, inputPerceptronCount> inputLayer; 
            std::array<std::unique_ptr<perceptron::IPerceptron>, hiddenPerceptronCount> hiddenLayer; 
            std::array<std::unique_ptr<perceptron::IPerceptron>, outputPerceptronCount> outputLayer; 
        };
}

#endif // ASTROCYTE_NETWORK_H