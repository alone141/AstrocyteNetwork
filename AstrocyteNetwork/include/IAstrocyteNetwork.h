#ifndef IASTROCYTE_NETWORK_H
#define IASTROCYTE_NETWORK_H

namespace an{
    class IAstrocyteNetwork {
        public:
            virtual constexpr ~IAstrocyteNetwork() = default;
        
            virtual constexpr void InitializeNetwork() = 0;
            virtual constexpr void ResetNetwork() = 0;
            virtual void Print() const = 0;
        };
}


#endif // IASTROCYTE_NETWORK_H