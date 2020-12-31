#ifndef METAWARENN_MODEL_H_
#define METAWARENN_MODEL_H_

#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <map>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"

namespace metawarenn {

class MWNNModel {
  public:
    MWNNModel() = default;
};

} //namespace metawarenn
#endif //METAWARENN_MODEL_H_
