#pragma once

#include "metawarenn_optimizer.h"

namespace metawarenn {

namespace optimizer {

class DummyPass3 : public MWNNOptimizer {
  public:
    DummyPass3();
    DummyPass3(int val);
    int get_value() { return value; }
  private:
    int value = 30;
};

} //namespace optimizer

} //namespace metawarenn
