#pragma once

#include "metawarenn_optimizer.h"

namespace metawarenn {

namespace optimizer {

class CalculateOffset : public MWNNOptimizer {
  public:
    CalculateOffset();
    CalculateOffset(std::shared_ptr<MWNNGraph> mwnn_graph);
    void RunPass() override;
  private:
    uint32_t bytes_size = 4 + 4 + 4 + 4; //(initializer index + name_length + data_type + dim_size)
};

} //namespace optimizer

} //namespace metawarenn
