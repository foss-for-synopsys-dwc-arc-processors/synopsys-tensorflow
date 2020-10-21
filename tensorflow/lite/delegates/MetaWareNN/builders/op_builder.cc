#include <iostream>

#include "model_builder.h"
#include "op_builder.h"
#include "tensorflow/lite/builtin_ops.h"

namespace tflite {
namespace delegates {
namespace metawarenn {

class BaseOpBuilder : public IOpBuilder {
 public:
  virtual ~BaseOpBuilder() = default;

  TfLiteStatus AddToModelBuilder(ModelBuilder& model_builder, const int32_t op_type) override;

 protected:
  virtual TfLiteStatus AddToModelBuilderImpl(ModelBuilder& model_builder, int32_t op_type)  = 0;
};

TfLiteStatus BaseOpBuilder::AddToModelBuilder(ModelBuilder& model_builder, int32_t op_type) {
  TF_LITE_ENSURE_STATUS(AddToModelBuilderImpl(model_builder, op_type));
  return kTfLiteOk;
}

class ConvOpBuilder : public BaseOpBuilder {
  TfLiteStatus AddToModelBuilderImpl(ModelBuilder& model_builder, int32_t op_type) override ;
};

TfLiteStatus ConvOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, int32_t op_type) {
  int32_t op_code;
  if (op_type == kTfLiteBuiltinConv2d) {
    op_code = METAWARENN_CONV_2D;
  }
  TF_LITE_ENSURE_STATUS(model_builder.AddOperation(op_code));
  return kTfLiteOk;
}


class DepthwiseConvOpBuilder : public BaseOpBuilder {
	  TfLiteStatus AddToModelBuilderImpl(ModelBuilder& model_builder, int32_t op_type) override ;
};

TfLiteStatus DepthwiseConvOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, int32_t op_type) {	
  int32_t op_code;
  if (op_type == kTfLiteBuiltinDepthwiseConv2d) {
    op_code = METAWARENN_DEPTHWISECONV_2D;
  }
  TF_LITE_ENSURE_STATUS(model_builder.AddOperation(op_code));
  return kTfLiteOk;
}


class PoolOpBuilder : public BaseOpBuilder {
 private:
  TfLiteStatus AddToModelBuilderImpl(ModelBuilder& model_builder, int32_t op_type) override;
};

TfLiteStatus PoolOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, int32_t op_type) {
  int32_t op_code;
  if (op_type == kTfLiteBuiltinAveragePool2d) {
    op_code = METAWARENN_GLOBAL_AVERAGE_POOL_2D; //GlobalAveragePool is coming under avgpool in TFLite
  }
  TF_LITE_ENSURE_STATUS(model_builder.AddOperation(op_code));
  return kTfLiteOk;
}

class BinaryOpBuilder : public BaseOpBuilder {
 private:
  TfLiteStatus AddToModelBuilderImpl(ModelBuilder& model_builder, int32_t op_type) override;
};

TfLiteStatus BinaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, int32_t op_type) {
  int32_t op_code;
  if (op_type == kTfLiteBuiltinAdd) {
    op_code = METAWARENN_ADD;
  }
  TF_LITE_ENSURE_STATUS(model_builder.AddOperation(op_code));
  return kTfLiteOk;
}

class ReluOpBuilder : public BaseOpBuilder {
 private:
  TfLiteStatus AddToModelBuilderImpl(ModelBuilder& model_builder, int32_t op_type) override;
};

TfLiteStatus ReluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, int32_t op_type) {
  int32_t op_code;
  if (op_type == kTfLiteBuiltinRelu) {
    op_code = METAWARENN_RELU;
  }
  TF_LITE_ENSURE_STATUS(model_builder.AddOperation(op_code));
  return kTfLiteOk;
}

class ReshapeOpBuilder : public BaseOpBuilder {
 private:
  TfLiteStatus AddToModelBuilderImpl(ModelBuilder& model_builder, int32_t op_type) override;
};

TfLiteStatus ReshapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, int32_t op_type) {
  int32_t op_code;
  if (op_type == kTfLiteBuiltinReshape) {
    op_code = METAWARENN_RESHAPE;
  }
  TF_LITE_ENSURE_STATUS(model_builder.AddOperation(op_code));
  return kTfLiteOk;
}

class SoftmaxOpBuilder : public BaseOpBuilder {
 private:
  TfLiteStatus AddToModelBuilderImpl(ModelBuilder& model_builder, int32_t op_type) override;
};

TfLiteStatus SoftmaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, int32_t op_type) {
  int32_t op_code;
  if (op_type == kTfLiteBuiltinSoftmax) {
    op_code = METAWARENN_SOFTMAX;
  }
  TF_LITE_ENSURE_STATUS(model_builder.AddOperation(op_code));
  return kTfLiteOk;
}


std::unordered_map<std::int32_t, std::shared_ptr<IOpBuilder>>
CreateOpBuilders() {
  std::cout<<"\nCreateOpBuilders!!!"<<std::endl;
  std::unordered_map<std::int32_t, std::shared_ptr<IOpBuilder>> op_map;

  op_map.emplace(kTfLiteBuiltinConv2d, std::make_shared<ConvOpBuilder>());
  op_map.emplace(kTfLiteBuiltinDepthwiseConv2d, std::make_shared<DepthwiseConvOpBuilder>());  

  auto pool_op_builder = std::make_shared<PoolOpBuilder>();
  op_map.emplace(kTfLiteBuiltinAveragePool2d, pool_op_builder);

  auto binary_op_builder = std::make_shared<BinaryOpBuilder>();
  op_map.emplace(kTfLiteBuiltinAdd, binary_op_builder);

  op_map.emplace(kTfLiteBuiltinRelu, std::make_shared<ReluOpBuilder>());
  op_map.emplace(kTfLiteBuiltinReshape, std::make_shared<ReshapeOpBuilder>());
  op_map.emplace(kTfLiteBuiltinSoftmax, std::make_shared<SoftmaxOpBuilder>());

  std::cout<<"\nOp map Created...\n";
  return op_map;
}

}  // namespace metawarenn
}  // namespace delegates
}  // namespace onnxruntime
