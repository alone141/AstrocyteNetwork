#ifndef ASTROCYTE_NETWORK_H
#define ASTROCYTE_NETWORK_H

#include "IAstrocyteNetwork.h"

namespace an{
    class AstrocyteNetwork : public IAstrocyteNetwork {
        public:
            constexpr AstrocyteNetwork() = default;
            constexpr ~AstrocyteNetwork() = default;

            constexpr void InitializeNetwork(){

            }
            constexpr void ResetNetwork(){

            }
        };
}

#endif // ASTROCYTE_NETWORK_H