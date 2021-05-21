#pragma once

#include "metawarenn_optimizer.h"

namespace metawarenn {

namespace optimizer {

class ConvertLayout : public MWNNOptimizer {
  public:
    ConvertLayout();
    ConvertLayout(std::shared_ptr<MWNNGraph> mwnn_graph, MWNNTensor mwnn_tensor, bool to_hwc, bool to_chw);
    ConvertLayout(std::shared_ptr<MWNNGraph> mwnn_graph, MWNNValueInfo mwnn_value_info, bool to_hwc, bool to_chw);
    void RunPass() override;
  private:
    bool CHW_to_HWC = 0;
    bool HWC_to_CHW = 0;
    bool is_tensor = false;
    bool is_value_info = false;
};

} //namespace optimizer

} //namespace metawarenn
