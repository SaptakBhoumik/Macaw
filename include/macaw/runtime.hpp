#ifndef _MACAW_RUNTIME_HPP
#define _MACAW_RUNTIME_HPP
#include "type.hpp"
namespace macaw{
class Runtime{
    private:
    NeuralNetwork m_network;
    FloatArray m_parameters;
    public:
    Runtime();
    Runtime(NeuralNetwork);
    Runtime(NeuralNetwork,FloatArray);
    FloatArray execute(FloatArray);
    FloatArray execute();
};
}
#endif