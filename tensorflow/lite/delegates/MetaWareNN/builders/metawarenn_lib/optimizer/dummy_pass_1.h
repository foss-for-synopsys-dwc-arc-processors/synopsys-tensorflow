#pragma once

#include "metawarenn_optimizer.h"

namespace metawarenn {

namespace optimizer {

class DummyPass1 : public MWNNOptimizer {
  public:
    DummyPass1();
    DummyPass1(int val);
    int get_value() { return value; }
  private:
    int value = 10;
};

} //namespace optimizer

} //namespace metawarenn
