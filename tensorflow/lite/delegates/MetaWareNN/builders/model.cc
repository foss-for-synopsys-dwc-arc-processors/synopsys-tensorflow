//#include <core/common/logging/logging.h>

#include "model.h"
#include "tensorflow/lite/delegates/MetaWareNN/MetaWareNN_lib/MetaWareNN_implementation.h"

namespace tflite {
namespace delegates {
namespace metawarenn{

Model::Model() : metawarenn_(MetaWareNNImplementation()) {}

const std::vector<std::string>& Model::GetInputs() const {
  return input_names_;
}

const std::vector<std::string>& Model::GetOutputs() const {
  return output_names_;
}

} // namespace metawarenn
} // namespace delegates
} // namespace tflite
