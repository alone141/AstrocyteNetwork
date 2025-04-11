# AstrocyteNetwork - Rediscovering Neural Networks 
An archaic, header-only and template based neural network project that completely trains on compile time

... at least thats what I'm hoping

Educational purposes for C++ constexpr capabilities. 
## Getting Started

### Prerequisites

- **C++ Compiler**: Requires a compiler with C++23 support.
- **CMake**: Build system generator (minimum version 3.20 recommended).

### Building the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/AstrocyteNetwork.git
   cd AstrocyteNetwork 

2. Create a build directory and configure the project:
   ```bash
    mkdir build
    cd build
    cmake ..
    cmake --build .

## Creating a network
To create a neural network, use the AstrocyteNetwork class template. For example:

   ```cpp
    #include <AstrocyteNetwork.h>

    constexpr int inputCount = 3;
    constexpr int hiddenCount = 4;
    constexpr int outputCount = 2;

    an::AstrocyteNetwork<inputCount, hiddenCount, outputCount> network;
```

### Training on compile time
https://godbolt.org/z/h3hE5T454
