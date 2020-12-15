#pragma once

#include "metawarenn_optimizer.h"

namespace metawarenn {

namespace optimizer {

class DummyPass2 : public MWNNOptimizer {
  public:
    DummyPass2();
    int get_value() { return value; }
  private:
    int value = 20;
};

} //namespace optimizer

} //namespace metawarenn
