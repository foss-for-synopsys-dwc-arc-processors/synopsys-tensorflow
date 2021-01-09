#include "pass_manager.h"

namespace metawarenn {

namespace optimizer {

PassManager::PassManager() { std::cout << "\n In PassManager Constructor!!"; }
void PassManager::run_passes() {
  for (auto& pass : pass_list) {
    std::cout  << "\n PassName : " << pass->get_name();
    if(auto pass1 = std::dynamic_pointer_cast<DummyPass1>(pass)) {
      std::cout << "\n Value : " << pass1->get_value();
    }
    else if (auto remove_reshape_pass = std::dynamic_pointer_cast<RemoveReshape>(pass)) {
      std::cout << "\n Name : " << remove_reshape_pass->get_name();
      remove_reshape_pass->RunPass();
    }
    else if (auto fuse_relu_pass = std::dynamic_pointer_cast<FuseRelu>(pass)) {
      std::cout << "\n Name : " << fuse_relu_pass->get_name();
      fuse_relu_pass->RunPass();
    }
  }
}

} //namespace optimizer

} //namespace metawarenn
