#ifndef ASTROCYTE_NETWORK_H
#define ASTROCYTE_NETWORK_H

#include <IAstrocyteNetwork.h>
#include <Perceptron.h>
#include <memory>



namespace an{
    template<int inputPerceptronCount, int hiddenPerceptronCount, int outputPerceptronCount, int layerCount = 1>
    class AstrocyteNetwork : public IAstrocyteNetwork {
        template<int perceptronCount>
        using Layer = std::array<std::unique_ptr<perceptron::IPerceptron>, perceptronCount>;
        using inputPerceptronType = perceptron::Perceptron<1,perceptron::ActivationFunctionEnum::Linear>;
        using hiddenPerceptronType = perceptron::Perceptron<inputPerceptronCount, perceptron::ActivationFunctionEnum::Sigmoid>;
        using outputPerceptronType = perceptron::Perceptron<hiddenPerceptronCount,perceptron::ActivationFunctionEnum::Sigmoid>;

        public:
            constexpr AstrocyteNetwork(){
                static_assert(inputPerceptronCount > 0,  "Input layer perceptron count must be bigger than 0");
                static_assert(hiddenPerceptronCount > 0, "Hidden layer perceptron count must be bigger than 0");
                static_assert(layerCount > 0,            "Layer count must be bigger than 0");
                static_assert(layerCount == 1,           "Currently, only networks with 1 hidden layer is supported");
                static_assert(outputPerceptronCount > 0, "Output layer perceptron count must be bigger than 0");

                InitializeNetwork();
            }
            constexpr ~AstrocyteNetwork() = default;

            constexpr void InitializeNetwork() override{
                for (auto &&perceptron : inputLayer)
                {
                    perceptron = std::make_unique<inputPerceptronType>();
                    perceptron->ResetWeights();
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
                for (std::size_t h = 0; h < hiddenPerceptronCount; ++h) {
                    for (std::size_t i = 0; i < inputPerceptronCount; ++i) {
                        hiddenLayer[h]->SetInput(i+1, inputLayer[i]->GetOutput());
                    }
                    hiddenLayer[h]->CalculateOutput();
                }

                for (std::size_t o = 0; o < outputPerceptronCount; ++o) {
                    for (std::size_t h = 0; h < hiddenPerceptronCount; ++h) {
                        outputLayer[o]->SetInput(h+1, hiddenLayer[h]->GetOutput());
                    }
                    outputLayer[o]->CalculateOutput();
                }    
            }

            constexpr void Backpropagation(std::array<float, outputPerceptronCount>& expectedArray, float learningRate) {

                //https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

                
                // Step 1: Compute gradients for the output layer
                std::array<float, outputPerceptronCount> outputGradients;
                for (std::size_t o = 0; o < outputPerceptronCount; ++o) {
                    const float& out = outputLayer[o]->GetOutput();
                    const float& target = expectedArray[o];
                    outputGradients[o] = (out - target) * out * (1 - out); // Gradient for output perceptron
                }
            
                // Step 2: Compute gradients for the hidden layer
                std::array<float, hiddenPerceptronCount> hiddenGradients;
                for (std::size_t h = 0; h < hiddenPerceptronCount; ++h) {
                    float gradientSum = 0.0f;
                    for (std::size_t o = 0; o < outputPerceptronCount; ++o) {
                        gradientSum += outputGradients[o] * outputLayer[o]->GetWeight(h + 1); // h+1 because weight[0] is bias
                    }
                    const float& hiddenOutput = hiddenLayer[h]->GetOutput();
                    hiddenGradients[h] = gradientSum * hiddenOutput * (1 - hiddenOutput); // Gradient for hidden perceptron
                }
            
                // Step 3: Update weights for the output layer
                for (std::size_t o = 0; o < outputPerceptronCount; ++o) {
                    for (std::size_t h = 0; h < hiddenPerceptronCount; ++h) {
                        const float& hiddenOutput = hiddenLayer[h]->GetOutput();
                        const float oldWeight = outputLayer[o]->GetWeight(h + 1); // h+1 because weight[0] is bias
                        const float newWeight = oldWeight - learningRate * outputGradients[o] * hiddenOutput;
                        outputLayer[o]->SetWeight(h + 1, newWeight);
                    }
                    // Update bias weight for output perceptron
                    const float oldBiasWeight = outputLayer[o]->GetWeight(0);
                    const float newBiasWeight = oldBiasWeight - learningRate * outputGradients[o];
                    outputLayer[o]->SetWeight(0, newBiasWeight);
                }
            
                // Step 4: Update weights for the hidden layer (input to hidden)
                for (std::size_t h = 0; h < hiddenPerceptronCount; ++h) {
                    for (std::size_t i = 0; i < inputPerceptronCount; ++i) {
                        const float& input = inputLayer[i]->GetOutput();
                        const float oldWeight = hiddenLayer[h]->GetWeight(i + 1); // i+1 because weight[0] is bias
                        const float newWeight = oldWeight - learningRate * hiddenGradients[h] * input;
                        hiddenLayer[h]->SetWeight(i + 1, newWeight);
                    }
                    // Update bias weight for hidden perceptron
                    const float oldBiasWeight = hiddenLayer[h]->GetWeight(0);
                    const float newBiasWeight = oldBiasWeight - learningRate * hiddenGradients[h];
                    hiddenLayer[h]->SetWeight(0, newBiasWeight);
                }
            }

            
            constexpr std::array<float,outputPerceptronCount> GetOutputLayer() const {
                std::array<float,outputPerceptronCount> result;
                for (std::size_t i = 0; i < outputPerceptronCount; ++i) {
                    result[i] = outputLayer[i]->GetOutput();
                }
                
                return result;
            }
            constexpr void ResetNetwork() override{

            }
            void Print(){
                std::cout << "Input Layer:" << std::endl;
                for (std::size_t i = 0; i < inputPerceptronCount; ++i) {
                    std::cout << "Perceptron " << i << ": " << std::endl;
                    inputLayer[i]->Print();
                }
            
                std::cout << "Hidden Layer:" << std::endl;
                for (std::size_t h = 0; h < hiddenPerceptronCount; ++h) {
                    std::cout << "Perceptron " << h << ": " << std::endl;
                    hiddenLayer[h]->Print();
                }
            
                std::cout << "Output Layer:" << std::endl;
                for (std::size_t o = 0; o < outputPerceptronCount; ++o) {
                    std::cout << "Perceptron " << o << ": " << std::endl;
                    outputLayer[o]->Print();
                }
            }
        private:
            Layer<inputPerceptronCount> inputLayer; 
            Layer<hiddenPerceptronCount> hiddenLayer; 
            Layer<outputPerceptronCount> outputLayer; 
        };
}

#endif // ASTROCYTE_NETWORK_H