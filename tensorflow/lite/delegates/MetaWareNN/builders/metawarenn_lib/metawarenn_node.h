#ifndef METAWARENN_NODE_H_
#define METAWARENN_NODE_H_

#include "metawarenn_model.h"

namespace metawarenn {

class MWNNNode {
  public:
    MWNNNode(std::string m_name, std::string m_op_type, std::vector<std::string> m_inputs,  std::vector<std::string> m_outputs);
    std::string get_name() { return name; }
    std::string get_op_type() { return op_type; }
    std::vector<std::string> get_inputs() { return inputs; }
    std::vector<std::string> get_outputs() { return outputs; }
  private:
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

} //namespace metawarenn

#endif //METAWARENN_NODE_H_
