#pragma once

#include <iostream>

namespace metawarenn {

namespace optimizer {

class MWNNOptimizer {
  public:
    MWNNOptimizer();
    virtual ~MWNNOptimizer() {}
    void set_name(const std::string& name) { pass_name = name; }
    std::string get_name() { return pass_name; }
  private:
    std::string pass_name;
};
} //namespace optimizer

} //namespace metawarenn
