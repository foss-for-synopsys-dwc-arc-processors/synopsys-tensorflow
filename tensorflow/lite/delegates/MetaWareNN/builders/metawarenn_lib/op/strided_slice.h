#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class StridedSlice : public Node {
  public:
    StridedSlice();
    StridedSlice(std::string n_name, std::vector<std::string> n_inputs,
        std::vector<std::string> n_outputs,  std::vector<int> n_begin_mask,
        std::vector<int> n_ellipsis_mask, std::vector<int> n_end_mask,
        std::vector<int> n_new_axis_mask, std::vector<int> n_shrink_axis_mask);
    void fill_attributes(DataSerialization &layer_serializer) override;
  private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<int> begin_mask;
    std::vector<int> ellipsis_mask;
    std::vector<int> end_mask;
    std::vector<int> new_axis_mask;
    std::vector<int> shrink_axis_mask;

};

} //namespace op

} //namespace metawarenn
