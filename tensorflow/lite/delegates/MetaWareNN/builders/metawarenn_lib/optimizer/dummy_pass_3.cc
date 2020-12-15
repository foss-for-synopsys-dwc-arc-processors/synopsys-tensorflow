#include "dummy_pass_3.h"

namespace metawarenn {

namespace optimizer {

DummyPass3::DummyPass3() {
  std::cout << "\n In DummyPass3 Constructor!!";
  set_name("DummyPass3");
}

DummyPass3::DummyPass3(int val) {
  std::cout << "\n In DummyPass3 Constructor!! Setting value";
  set_name("DummyPass3");
  value = val;
}

} //namespace optimizer

} //namespace metawarenn
