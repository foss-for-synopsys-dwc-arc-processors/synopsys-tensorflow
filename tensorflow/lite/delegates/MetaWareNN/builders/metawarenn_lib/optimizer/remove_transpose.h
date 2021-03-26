#pragma once

#include "metawarenn_optimizer.h"

namespace metawarenn {

namespace optimizer {

class RemoveTranspose : public MWNNOptimizer {
  public:
    RemoveTranspose();
    RemoveTranspose(MWNNGraph* mwnn_graph, MWNNNode mwnn_node);
    void RunPass() override;
};

} //namespace optimizer

} //namespace metawarenn
