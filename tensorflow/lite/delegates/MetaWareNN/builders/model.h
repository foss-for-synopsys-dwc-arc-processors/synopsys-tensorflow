#ifndef ONNXRUNTIME_METAWARENN_IMPLEMENTATION_H_
#define ONNXRUNTIME_METAWARENN_IMPLEMENTATION_H_

#include "tensorflow/lite/delegates/MetaWareNN/MetaWareNN_lib/NeuralNetworksWrapper.h"

#include <vector>

namespace tflite {
namespace delegates {
namespace metawarenn{

class Model {
  friend class ModelBuilder;

 public:
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  const std::vector<std::string>& GetInputs() const;
  const std::vector<std::string>& GetOutputs() const;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  Model();

 private:
  const MetaWareNN* metawarenn_{nullptr};

  MetaWareNNModel* model_{nullptr};
  MetaWareNNCompilation* compilation_{nullptr};

};

} // namespace metawarenn
} // namespace delegates
} // namespace tflite

#endif //TFLITE_METAWARENN_IMPLEMENTATION_H_
