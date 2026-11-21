# AstrocyteNetwork - Rediscovering Neural Networks
A header-only, template-based C++23 neural network that trains entirely at compile time.

The compiler runs the whole training (forward pass, backpropagation, weight updates), and the program ships with the trained weights already baked in as constants. An educational project exploring how far C++ `constexpr` can be pushed.

```cpp
#include <AstrocyteNetwork.h>

using Network = an::AstrocyteNetwork<3, 5, 2>; // 3 inputs, 5 hidden perceptrons, 2 outputs

constexpr auto trainingData = std::to_array({
    std::make_pair(std::to_array({0.0f, 0.0f, 1.0f}), std::to_array({0.0f, 0.0f})),
    std::make_pair(std::to_array({0.0f, 1.0f, 1.0f}), std::to_array({1.0f, 0.0f})),
    std::make_pair(std::to_array({1.0f, 0.0f, 1.0f}), std::to_array({1.0f, 0.0f})),
    std::make_pair(std::to_array({1.0f, 1.0f, 0.5f}), std::to_array({0.0f, 1.0f}))
});

consteval auto TrainAtCompileTime() {
    Network network;
    network.Train(trainingData, 0.5f, 1000); // learning rate, epochs
    return network.SerializeWeights();
}

// Computed by the compiler. Only the resulting numbers end up in the binary.
constexpr auto trainedWeights = TrainAtCompileTime();

int main() {
    Network network(trainedWeights);  // no training at runtime
    auto output = network.Predict({0.0f, 1.0f, 1.0f}); // ~{0.85, 0.10}
}
```

Output of [`Main/main.cpp`](Main/main.cpp):
```
Network trained at compile time (1000 epochs):
  Input: {0, 0, 1} -> Predicted: {0.152054, 0.065941}, Target: {0, 0}
  Input: {0, 1, 1} -> Predicted: {0.850369, 0.103192}, Target: {1, 0}
  Input: {1, 0, 1} -> Predicted: {0.86626, 0.0988428}, Target: {1, 0}
  Input: {1, 1, 0.5} -> Predicted: {0.141003, 0.871584}, Target: {0, 1}

The same training at runtime takes 1.53966 ms
```

## How it works
- **Everything is `constexpr`.** Perceptrons, layers, training and the math functions can all be evaluated by the compiler. `consteval` functions are forced to run at compile time.
- **Heap memory and virtual functions inside the compiler.** Since C++20, constant evaluation allows `new`/`delete` and virtual calls, so the network can use `std::unique_ptr<IPerceptron>` layers. The catch: memory allocated during constant evaluation must be freed before it ends. That is why the network object itself cannot be a `constexpr` variable, and its weights are copied out into a plain `std::array` with `SerializeWeights()`.
- **Its own math.** `std::exp` is not `constexpr` before C++26, so `exp`, `ln` and the activation functions are implemented with range reduction and series expansions in [`Utility.h`](Utility/include/Utility.h).
- **Reproducible randomness.** Initial weights come from a `constexpr` PCG32 generator with a fixed seed ([`Config.h`](Utility/include/Config.h)), one stream per perceptron. Every build trains the exact same network, and GCC and Clang produce identical weights.

## Getting Started

### Prerequisites
- A C++23 compiler. Tested with **GCC 13** and **Clang 18**; MSVC 2022 is supported through its own flags.
- **CMake** 3.20 or newer.

### Building
```bash
git clone https://github.com/alone141/AstrocyteNetwork.git
cd AstrocyteNetwork
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/Main/Main
```

Compilers stop evaluating a constant expression after a fixed amount of work, far less than a training run needs. The `astrocyte_enable_compile_time_training()` CMake function raises that limit with each compiler's own flag (`-fconstexpr-ops-limit` for GCC, `-fconstexpr-steps` for Clang, `/constexpr:steps` for MSVC). Use it for your own targets that train at compile time.

### Running the tests
```bash
cmake -B build -DENABLE_TEST=ON
cmake --build build
ctest --test-dir build
```
GoogleTest is taken from your system if installed, otherwise downloaded.

Some tests worth a look in [`AstrocyteNetworkUnitTest.cpp`](AstrocyteNetwork/test/src/AstrocyteNetworkUnitTest.cpp):
- a `static_assert` that trains XOR **and** checks the predictions at compile time: if the network stops learning, the test file no longer compiles
- a gradient check that compares backpropagation with a numerical derivative of the loss
- a check that compile-time and runtime training produce the same weights

## Cost of training in the compiler
Training the 3-5-2 network above for 1000 epochs:

| | Time |
|---|---|
| At runtime | ~1.5 ms |
| At compile time (GCC 13 / Clang 18) | ~14 s / ~12 s |

The compiler's constant evaluator interprets the code step by step, roughly 10,000 times slower than the compiled program. Compile time grows linearly with epochs, training samples and weights, so this approach suits small networks.

## Limitations
- One hidden layer
- Hidden and output layers use `PseudoSigmoid`, `0.5 * (x / (1 + |x|) + 1)`, a cheap sigmoid-shaped function
- Fully connected layers only, trained with plain stochastic gradient descent on mean squared error
