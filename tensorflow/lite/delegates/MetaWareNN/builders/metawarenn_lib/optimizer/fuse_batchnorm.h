#pragma once

#include "metawarenn_optimizer.h"

namespace metawarenn {

namespace optimizer {

class FuseBatchNorm : public MWNNOptimizer {
  public:
    FuseBatchNorm();
    FuseBatchNorm(std::shared_ptr<MWNNGraph> mwnn_graph, MWNNNode mwnn_node);
    void RunPass() override;
};

} //namespace optimizer

} //namespace metawarenn
