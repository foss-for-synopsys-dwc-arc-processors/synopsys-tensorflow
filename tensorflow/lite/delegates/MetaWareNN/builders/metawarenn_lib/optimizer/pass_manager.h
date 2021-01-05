#pragma once

#include "metawarenn_optimizer.h"
#include "dummy_pass_1.h"
#include "dummy_pass_2.h"
#include "dummy_pass_3.h"
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