#ifndef ASTROCYTE_NETWORK_H
#define ASTROCYTE_NETWORK_H

#include <IAstrocyteNetwork.h>
#include <Perceptron.h>
#include <memory>



namespace an{
    template<int inputPerceptronCount, int hiddenPerceptronCount, int outputPerceptronCount, int layerCount = 1,
    int totalWeightCount = ((inputPerceptronCount+1)*hiddenPerceptronCount + (hiddenPerceptronCount+1)*outputPerceptronCount)>
    class AstrocyteNetwork : public IAstrocyteNetwork {
        template<int perceptronCount>
        using Layer = std::array<std::unique_ptr<perceptron::IPerceptron>, perceptronCount>;
        using inputPerceptronType = perceptron::Perceptron<1,perceptron::ActivationFunctionEnum::Linear>;
        using hiddenPerceptronType = perceptron::Perceptron<inputPerceptronCount, perceptron::ActivationFunctionEnum::PseudoSigmoid>;
        using outputPerceptronType = perceptron::Perceptron<hiddenPerceptronCount,perceptron::ActivationFunctionEnum::PseudoSigmoid>;

        static_assert(inputPerceptronCount > 0,  "Input layer perceptron count must be bigger than 0");
        static_assert(hiddenPerceptronCount > 0, "Hidden layer perceptron count must be bigger than 0");
        static_assert(layerCount > 0,            "Layer count must be bigger than 0");
        static_assert(layerCount == 1,           "Currently, only networks with 1 hidden layer is supported");
        static_assert(outputPerceptronCount > 0, "Output layer perceptron count must be bigger than 0");
        static_assert(totalWeightCount == 
            (inputPerceptronCount+1)*hiddenPerceptronCount + (hiddenPerceptronCount+1)*outputPerceptronCount, 
            "Please do not use this template argument, it is calculated automatically");

        public:
            constexpr AstrocyteNetwork(){
                InitializeNetwork();
            }
            constexpr ~AstrocyteNetwork() = default;
            constexpr AstrocyteNetwork(const AstrocyteNetwork& other) {
                // Copy input layer
                for (std::size_t i = 0; i < inputPerceptronCount; ++i) {
                    inputLayer[i] = std::make_unique<inputPerceptronType>(*static_cast<inputPerceptronType*>(other.inputLayer[i].get()));
                }
        
                // Copy hidden layer
                for (std::size_t h = 0; h < hiddenPerceptronCount; ++h) {
                    hiddenLayer[h] = std::make_unique<hiddenPerceptronType>(*static_cast<hiddenPerceptronType*>(other.hiddenLayer[h].get()));
                }
        
                // Copy output layer
                for (std::size_t o = 0; o < outputPerceptronCount; ++o) {
                    outputLayer[o] = std::make_unique<outputPerceptronType>(*static_cast<outputPerceptronType*>(other.outputLayer[o].get()));
                }
            }
            constexpr AstrocyteNetwork(std::array<float,totalWeightCount> weights) {
                InitializeNetwork(); //this is suboptimal
                DeserializeWeights(weights);
            }
            constexpr void InitializeNetwork() override{
                int perceptronSeed = 1; //used for random initialization of weights
                for (auto &&perceptron : inputLayer)
                {
                    perceptron = std::make_unique<inputPerceptronType>(perceptronSeed);
                    perceptron->ResetWeights();
                    perceptronSeed++;
                }

                perceptronSeed = 100;
                for (auto &&perceptron : hiddenLayer)
                {
                    perceptron = std::make_unique<hiddenPerceptronType>(perceptronSeed);
                    perceptronSeed++;
                }

                perceptronSeed = 1000;
                for (auto &&perceptron : outputLayer)
                {
                    perceptron = std::make_unique<outputPerceptronType>(perceptronSeed);
                    perceptronSeed++;
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
                std::array<float, outputPerceptronCount> outputGradients;
                for (std::size_t o = 0; o < outputPerceptronCount; ++o) {
                    const float& out = outputLayer[o]->GetOutput();
                    const float& target = expectedArray[o];
                    outputGradients[o] = (out - target) * out * (1 - out);
                }
            
                std::array<float, hiddenPerceptronCount> hiddenGradients;
                for (std::size_t h = 0; h < hiddenPerceptronCount; ++h) {
                    float gradientSum = 0.0f;
                    for (std::size_t o = 0; o < outputPerceptronCount; ++o) {
                        gradientSum += outputGradients[o] * outputLayer[o]->GetWeight(h + 1); // h+1 because weight[0] is bias
                    }
                    const float& hiddenOutput = hiddenLayer[h]->GetOutput();
                    hiddenGradients[h] = gradientSum * hiddenOutput * (1 - hiddenOutput);
                }
            
                for (std::size_t o = 0; o < outputPerceptronCount; ++o) {
                    for (std::size_t h = 0; h < hiddenPerceptronCount; ++h) {
                        const float& hiddenOutput = hiddenLayer[h]->GetOutput();
                        const float oldWeight = outputLayer[o]->GetWeight(h + 1); 
                        const float newWeight = oldWeight - learningRate * outputGradients[o] * hiddenOutput;
                        outputLayer[o]->SetWeight(h + 1, newWeight);
                    }
                    const float oldBiasWeight = outputLayer[o]->GetWeight(0);
                    const float newBiasWeight = oldBiasWeight - learningRate * outputGradients[o];
                    outputLayer[o]->SetWeight(0, newBiasWeight);
                }
            
                //  (input to hidden)
                for (std::size_t h = 0; h < hiddenPerceptronCount; ++h) {
                    for (std::size_t i = 0; i < inputPerceptronCount; ++i) {
                        const float& input = inputLayer[i]->GetOutput();
                        const float oldWeight = hiddenLayer[h]->GetWeight(i + 1); 
                        const float newWeight = oldWeight - learningRate * hiddenGradients[h] * input;
                        hiddenLayer[h]->SetWeight(i + 1, newWeight);
                    }
                    // Update bias weight for hidden perceptron
                    const float oldBiasWeight = hiddenLayer[h]->GetWeight(0);
                    const float newBiasWeight = oldBiasWeight - learningRate * hiddenGradients[h];
                    hiddenLayer[h]->SetWeight(0, newBiasWeight);
                }
            }
            constexpr void Train(std::array<std::pair<std::array<float, 3>, std::array<float, 2>>, 4> trainingData, float learningRate, int epochs){
                for (int epoch = 0; epoch < epochs; ++epoch) {
                    float totalError = 0.0f;

                    for (auto& [input, targetOutput] : trainingData) {
                        FeedForward(input);

                        Backpropagation(targetOutput, learningRate);

                        // Calculate error for monitoring
                        //auto output = GetOutputLayer();
                        //totalError += 0.5f * (targetOutput[0] - output[0]) * (targetOutput[0] - output[0]); // Mean Squared Error
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
                for (auto &&perceptron : inputLayer)
                {
                    perceptron->ResetWeights();
                }
                for (auto &&perceptron : hiddenLayer)
                {
                    perceptron->ResetWeights();
                }
                for (auto &&perceptron : outputLayer)
                {
                    perceptron->ResetWeights();
                }
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

            constexpr std::array<float,totalWeightCount> SerializeWeights(){
                std::array<float,totalWeightCount> weights;
                int index = 0;
                for (std::size_t i = 0; i < hiddenPerceptronCount; i++)
                {
                    for (std::size_t w = 0; w < inputPerceptronCount+1; w++)
                    {
                        weights[index] = hiddenLayer[i]->GetWeight(w);
                        index++;
                    }
                }
                for (std::size_t i = 0; i < outputPerceptronCount; i++)
                {
                    for (std::size_t w = 0; w < hiddenPerceptronCount+1; w++)
                    {
                        weights[index] = outputLayer[i]->GetWeight(w);
                        index++;
                    }
                }
                return weights;
            }

            void DeserializeWeights(std::array<float,totalWeightCount> weights){

                int index = 0;
                for (std::size_t i = 0; i < hiddenPerceptronCount; i++)
                {
                    for (std::size_t w = 0; w < inputPerceptronCount+1; w++)
                    {
                        hiddenLayer[i]->SetWeight(w, weights[index]);
                        index++;
                    }
                }
                for (std::size_t i = 0; i < outputPerceptronCount; i++)
                {
                    for (std::size_t w = 0; w < hiddenPerceptronCount+1; w++)
                    {
                        outputLayer[i]->SetWeight(w, weights[index]);
                        index++;
                    }
                }
            }
        private:
            Layer<inputPerceptronCount> inputLayer; 
            Layer<hiddenPerceptronCount> hiddenLayer; 
            Layer<outputPerceptronCount> outputLayer; 
        };
}

#endif // ASTROCYTE_NETWORK_H