#ifndef _MACAW_TYPE_HPP
#define _MACAW_TYPE_HPP
#include <vector>
namespace macaw {
typedef std::vector<float> FloatArray;
typedef std::vector<FloatArray> Matrix;
struct NeuralLayer {
  Matrix m_weights;
  FloatArray m_biases;
};
typedef std::vector<NeuralLayer> NeuralNetwork;
}  // namespace macaw
#endif