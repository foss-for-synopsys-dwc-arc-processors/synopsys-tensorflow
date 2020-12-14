#ifndef METAWARENN_TENSOR_H_
#define METAWARENN_TENSOR_H_

#include "metawarenn_model.h"

namespace metawarenn {

class MWNNTensor {
  public:
    MWNNTensor(std::string m_name, std::vector<int> m_dims, std::vector<float> m_tensor);
    std::string get_name() { return name; }
    std::vector<int> get_dims() { return dims; }
    std::vector<float> get_tensor() { return tensor; }
  private:
    std::string name;
    std::vector<int> dims;
    std::vector<float> tensor;
};

} //namespace metawarenn

#endif //METAWARENN_TENSOR_H_
