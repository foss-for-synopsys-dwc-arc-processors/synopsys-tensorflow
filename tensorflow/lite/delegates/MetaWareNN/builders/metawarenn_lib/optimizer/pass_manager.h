#pragma once

#include "metawarenn_optimizer.h"
#include "remove_reshape.h"
#include "fuse_relu.h"
#include "convert_layout.h"
#include "remove_transpose.h"
#include "calculate_offset.h"
#include "fuse_batchnorm.h"
#include <vector>
#include <memory>

namespace metawarenn {

namespace optimizer {

class PassManager {
  public:
    PassManager();
    template <class T>
    void register_pass(T &pass) {
      auto p = std::make_shared<T>(pass);
      pass_list.push_back(std::static_pointer_cast<MWNNOptimizer>(p));
      std::cout << "\n In Register Pass : " << pass.get_name();
    }
    void run_passes();
  private:
    std::vector<std::shared_ptr<MWNNOptimizer>> pass_list;
};
} //namespace optimizer

} //namespace metawarenn
