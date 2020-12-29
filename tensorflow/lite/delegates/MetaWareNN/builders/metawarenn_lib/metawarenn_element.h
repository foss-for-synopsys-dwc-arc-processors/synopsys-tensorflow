#ifndef METAWARENN_ELEMENT_H_
#define METAWARENN_ELEMENT_H_

#include "tensorflow/lite/c/common.h"

namespace metawarenn {

class ElementType {
  public:
    enum class element_type
      {
          //Common ONNX & TF Types
          boolean_,
          double_,
          float16_,
          float_,
          int8_,
          int16_,
          int32_,
          int64_,
          uint8_,
          //ONNX Specific
          uint16_,
          uint32_,
          uint64_,
          //TF Specific
          string_,
          complex64_,
          //Common ONNX & TF Types
          dynamic_
      };

    static element_type get_mwnn_type(int tf_type) {
        switch (tf_type) {
            case kTfLiteBool:
                return element_type::boolean_;
            case kTfLiteFloat64:
                return element_type::double_;
            case kTfLiteFloat16:
                return element_type::float16_;
            case kTfLiteFloat32:
                return element_type::float_;
            case kTfLiteInt8:
                return element_type::int8_;
            case kTfLiteInt16:
                return element_type::int16_;
            case kTfLiteInt32:
                return element_type::int32_;
            case kTfLiteInt64:
                return element_type::int64_;
            case kTfLiteUInt8:
                return element_type::uint8_;
            case kTfLiteNoType:
                return element_type::dynamic_;
            case kTfLiteString:
                return element_type::string_;
            case kTfLiteComplex64:
                return element_type::complex64_;
            default:
                return element_type::dynamic_;
        }
    }
};

} //namespace metawarenn

#endif //METAWARENN_ELEMENT_H_
