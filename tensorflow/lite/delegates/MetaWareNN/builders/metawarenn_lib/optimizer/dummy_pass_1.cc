#include "dummy_pass_1.h"

namespace metawarenn {

namespace optimizer {

DummyPass1::DummyPass1() {
  std::cout << "\n In DummyPass1 Constructor!!";
  set_name("DummyPass1");
}

DummyPass1::DummyPass1(int val) {
  std::cout << "\n In DummyPass1 Constructor!! Setting value";
  set_name("DummyPass1");
  value = val;
}

} //namespace optimizer

} //namespace metawarenn
