#include "pass_manager.h"

namespace metawarenn {

namespace optimizer {

PassManager::PassManager() { std::cout << "\n In PassManager Constructor!!"; }
void PassManager::run_passes() {
  for (auto& pass : pass_list) {
    if (auto remove_reshape_pass = std::dynamic_pointer_cast<RemoveReshape>(pass)) {
      std::cout << "\n Name : " << remove_reshape_pass->get_name();
      remove_reshape_pass->RunPass();
    }
    else if (auto fuse_relu_pass = std::dynamic_pointer_cast<FuseRelu>(pass)) {
      std::cout << "\n Name : " << fuse_relu_pass->get_name();
      fuse_relu_pass->RunPass();
    }
    else if (auto convert_layout_pass = std::dynamic_pointer_cast<ConvertLayout>(pass)) {
      std::cout << "\n Name : " << convert_layout_pass->get_name();
      convert_layout_pass->RunPass();
    }
  }
}

} //namespace optimizer

} //namespace metawarenn
