#include "pass_manager.h"

namespace metawarenn {

namespace optimizer {

PassManager::PassManager() { /*std::cout << "\n In PassManager Constructor!!";*/ }
void PassManager::run_passes() {
  for (auto& pass : pass_list) {
    if (auto remove_reshape_pass = std::dynamic_pointer_cast<RemoveReshape>(pass)) {
      remove_reshape_pass->RunPass();
    }
    else if (auto fuse_relu_pass = std::dynamic_pointer_cast<FuseRelu>(pass)) {
      fuse_relu_pass->RunPass();
    }
    else if (auto convert_layout_pass = std::dynamic_pointer_cast<ConvertLayout>(pass)) {
      convert_layout_pass->RunPass();
    }
    else if (auto remove_transpose_pass = std::dynamic_pointer_cast<RemoveTranspose>(pass)) {
      remove_transpose_pass->RunPass();
    }
    else if (auto calculate_offset_pass = std::dynamic_pointer_cast<CalculateOffset>(pass)) {
      calculate_offset_pass->RunPass();
    }
  }
}

} //namespace optimizer

} //namespace metawarenn
