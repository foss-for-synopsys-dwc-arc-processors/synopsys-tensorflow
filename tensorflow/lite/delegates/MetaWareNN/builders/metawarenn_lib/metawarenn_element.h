#ifndef METAWARENN_ELEMENT_H_
#define METAWARENN_ELEMENT_H_

#include "metawarenn_common.h"

#if ONNX
using namespace onnx;
#endif

namespace metawarenn {

class ElementType {
  public:
    enum class element_type
      {
          //Common Types
          boolean_,   //0
          double_,    //1
          float16_,   //2
          float_,     //3
          int8_,      //4
          int16_,     //5
          int32_,     //6
          int64_,     //7
          uint8_,     //8
          //ONNX & TVM Specific
          uint16_,    //9
          uint32_,    //10
          uint64_,    //11
          //TF Specific
          string_,    //12
          complex64_, //13
          //Common ONNX & TF Types
          dynamic_    //14
      };

    #if ONNX
    static element_type get_mwnn_type_onnx(int onnx_type) {
        switch (onnx_type) {
            case onnx::TensorProto_DataType_BOOL:
                return element_type::boolean_;
            case onnx::TensorProto_DataType_DOUBLE:
                return element_type::double_;
            case onnx::TensorProto_DataType_FLOAT16:
                return element_type::float16_;
            case onnx::TensorProto_DataType_FLOAT:
                return element_type::float_;
            case onnx::TensorProto_DataType_INT8:
                return element_type::int8_;
            case onnx::TensorProto_DataType_INT16:
                return element_type::int16_;
            case onnx::TensorProto_DataType_INT32:
                return element_type::int32_;
            case onnx::TensorProto_DataType_INT64:
                return element_type::int64_;
            case onnx::TensorProto_DataType_UINT8:
                return element_type::uint8_;
            case onnx::TensorProto_DataType_UINT16:
                return element_type::uint16_;
            case onnx::TensorProto_DataType_UINT32:
                return element_type::uint32_;
            case onnx::TensorProto_DataType_UINT64:
                return element_type::uint64_;
            case onnx::TensorProto_DataType_UNDEFINED:
                return element_type::dynamic_;
            default:
                return element_type::dynamic_;
        }
    }

    static element_type get_mwnn_attr_type_onnx(int onnx_attr_type) {
        switch (onnx_attr_type) {
            case onnx::AttributeProto_AttributeType_FLOAT:
                return element_type::float_;
            case onnx::AttributeProto_AttributeType_INT:
                return element_type::int32_;
            case onnx::AttributeProto_AttributeType_STRING:
                return element_type::string_;
            case onnx::AttributeProto_AttributeType_FLOATS:
                return element_type::float_;
            case onnx::AttributeProto_AttributeType_INTS:
                return element_type::int32_;
            case onnx::AttributeProto_AttributeType_STRINGS:
                return element_type::string_;
            default:
                return element_type::dynamic_;
        }
    }
    #endif

    #if TFLITE
    static element_type get_mwnn_type_tf(int tf_type) {
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
    #endif

    #if GLOW
    static element_type get_mwnn_type_glow(ElemKind glow_type) {
        switch (glow_type) {
            case ElemKind::BoolTy:
                return element_type::boolean_;
            case ElemKind::Float16Ty:
                return element_type::float16_;
            case ElemKind::FloatTy:
                return element_type::float_;
            case ElemKind::Int8QTy:
                return element_type::int8_;
            case ElemKind::Int16QTy:
                return element_type::int16_;
            case ElemKind::Int32QTy:
                return element_type::int32_;
            case ElemKind::Int64ITy:
                return element_type::int64_;
            case ElemKind::UInt8QTy:
                return element_type::uint8_;
            default:
                return element_type::dynamic_;
        }
    }
    #endif

    #if TVM
    static element_type get_mwnn_type_tvm(int tvm_type) {
        switch (tvm_type) {
            case kDLFloat:
                return element_type::float_;
            case kDLInt:
                return element_type::int32_;
            case kDLUInt:
                return element_type::uint32_;
            default:
                return element_type::dynamic_;
        }
    }
    #endif
};

enum ActivationType {
  Activation_None = 0,
  Activation_Relu = 1,
  Activation_Relu6 = 2,
};

} //namespace metawarenn

#endif //METAWARENN_ELEMENT_H_
