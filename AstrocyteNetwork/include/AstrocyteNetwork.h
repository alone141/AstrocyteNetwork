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
        using hiddenPerceptronType = perceptron::Perceptron<inputPerceptronCount, perceptron::ActivationFunctionEnum::ReLU>;
        using outputPerceptronType = perceptron::Perceptron<hiddenPerceptronCount,perceptron::ActivationFunctionEnum::Linear>;

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
                }
                for (auto &&perceptron : hiddenLayer)
                {
                    perceptron = std::make_unique<hiddenPerceptronType>();
                }
                for (auto &&perceptron : outputLayer)
                {
                    perceptron = std::make_unique<outputPerceptronType>();
                    perceptron->ResetWeights();
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

            constexpr void Backpropagation(std::array<float,outputPerceptronCount>& expectedArray, float learningRate){
                std::array<float, outputPerceptronCount> gradient;                
                for (std::size_t h = 0; h < hiddenPerceptronCount; h++)
                {
                    //https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

                    //NOT READY
                    //backpropagation from hidden to input
                    for (std::size_t h = 0; h < hiddenPerceptronCount; h++)
                    {
                        for (std::size_t i = 0; i < outputPerceptronCount; i++)
                        {
                            const float& out = outputLayer[i]->GetOutput();
                            const float& target = expectedArray[i];
                            const float& oldWeight = hiddenLayer[h]->GetWeight(i);
    
                            const float gradient = -(target - out) * out * (1 - out) * out;
    
                            const float newWeight = oldWeight - learningRate * gradient;
                        }
                    }


                    //backpropagation from output to hidden
                    for (std::size_t h = 0; h < hiddenPerceptronCount; h++)
                    {
                        for (std::size_t i = 0; i < outputPerceptronCount; i++)
                        {
                            const float& out = outputLayer[i]->GetOutput();
                            const float& target = expectedArray[i];
                            const float& oldWeight = hiddenLayer[h]->GetWeight(i);
    
                            const float gradient = -(target - out) * out * (1 - out) * out;
    
                            const float newWeight = oldWeight - learningRate * gradient;
                            hiddenLayer[h]->SetWeight(i, newWeight);
                        }
                    }
                    

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
        private:
            Layer<inputPerceptronCount> inputLayer; 
            Layer<hiddenPerceptronCount> hiddenLayer; 
            Layer<outputPerceptronCount> outputLayer; 
        };
}

#endif // ASTROCYTE_NETWORK_H