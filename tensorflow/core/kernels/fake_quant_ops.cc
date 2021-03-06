/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/fake_quant_ops_functor.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/protobuf.h"

using tensorflow::BinaryElementWiseOp;
using tensorflow::DEVICE_CPU;
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
using tensorflow::DEVICE_GPU;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TTypes;  // NOLINT This is needed in CUDA mode, do not remove.
using tensorflow::UnaryElementWiseOp;
using tensorflow::errors::InvalidArgument;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace {
bool IsNumBitsValid(int num_bits) { return num_bits >= 2 && num_bits <= 16; }
}  // namespace

// -----------------------------------------------------------------------------
// Implementation of FakeQuantWithMinMaxArgsOp, see its documentation in
// core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxArgsOp
    : public UnaryElementWiseOp<float, FakeQuantWithMinMaxArgsOp<Device>> {
 public:
  typedef UnaryElementWiseOp<float, FakeQuantWithMinMaxArgsOp<Device>> Base;
  explicit FakeQuantWithMinMaxArgsOp(OpKernelConstruction* context)
      : Base::UnaryElementWiseOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("min", &min_));
    OP_REQUIRES_OK(context, context->GetAttr("max", &max_));
    OP_REQUIRES(context, min_ < max_,
                InvalidArgument("min has to be smaller than max, was: ", min_,
                                " >= ", max_));
    int num_bits;
    OP_REQUIRES_OK(context, context->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(
        context, IsNumBitsValid(num_bits),
        InvalidArgument("num_bits must be between 2 and 16, inclusive"));
    bool narrow_range;
    OP_REQUIRES_OK(context, context->GetAttr("narrow_range", &narrow_range));
    OP_REQUIRES_OK(context, context->GetAttr("tensor_type", &tensor_type_));
    OP_REQUIRES_OK(context, context->GetAttr("ev_quant", &ev_quant_));
    quant_min_ = narrow_range ? 1 : 0;
    quant_max_ = (1 << num_bits) - 1;
  }

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    FakeQuantWithMinMaxArgsFunctor<Device> functor;
    functor(context->eigen_device<Device>(), input.flat<float>(), min_, max_,
            quant_min_, quant_max_, tensor_type_, ev_quant_, output->flat<float>());
  }

 private:
  float min_;
  float max_;
  int quant_min_;
  int quant_max_;
  int tensor_type_;
  bool ev_quant_;
};

// Implementation of FakeQuantWithMinMaxArgsGradientOp, see its documentation in
// core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxArgsGradientOp
    : public BinaryElementWiseOp<float,
                                 FakeQuantWithMinMaxArgsGradientOp<Device>> {
 public:
  typedef BinaryElementWiseOp<float, FakeQuantWithMinMaxArgsGradientOp<Device>>
      Base;
  explicit FakeQuantWithMinMaxArgsGradientOp(OpKernelConstruction* context)
      : Base::BinaryElementWiseOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("min", &min_));
    OP_REQUIRES_OK(context, context->GetAttr("max", &max_));
    OP_REQUIRES(context, min_ < max_,
                InvalidArgument("min has to be smaller than max, was: ", min_,
                                " >= ", max_));
    int num_bits;
    OP_REQUIRES_OK(context, context->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(
        context, IsNumBitsValid(num_bits),
        InvalidArgument("num_bits must be between 2 and 16, inclusive"));
    bool narrow_range;
    OP_REQUIRES_OK(context, context->GetAttr("narrow_range", &narrow_range));
    OP_REQUIRES_OK(context, context->GetAttr("tensor_type", &tensor_type_));
    OP_REQUIRES_OK(context, context->GetAttr("ev_quant", &ev_quant_));
    quant_min_ = narrow_range ? 1 : 0;
    quant_max_ = (1 << num_bits) - 1;
  }

  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& gradient,
               const Tensor& input, Tensor* output) {
    OperateNoTemplate(context, gradient, input, output);
  }

  void OperateNoTemplate(OpKernelContext* context, const Tensor& gradient,
                         const Tensor& input, Tensor* output) {
    OP_REQUIRES(context, input.IsSameSize(gradient),
                InvalidArgument("gradient and input must be the same size"));
    FakeQuantWithMinMaxArgsGradientFunctor<Device> functor;
    functor(context->eigen_device<Device>(), gradient.flat<float>(),
            input.flat<float>(), min_, max_, quant_min_, quant_max_,
            tensor_type_, ev_quant_, output->flat<float>());
  }

 private:
  float min_;
  float max_;
  int quant_min_;
  int quant_max_;
  int tensor_type_;
  bool ev_quant_;
};

REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxArgs").Device(DEVICE_CPU),
                        FakeQuantWithMinMaxArgsOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(
    Name("FakeQuantWithMinMaxArgsGradient").Device(DEVICE_CPU),
    FakeQuantWithMinMaxArgsGradientOp<CPUDevice>);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
typedef Eigen::GpuDevice GPUDevice;

// Forward declarations for functor specializations for GPU.
template <>
void FakeQuantWithMinMaxArgsFunctor<GPUDevice>::operator()(
    const GPUDevice& d, typename TTypes<float>::ConstFlat inputs,
    const float min, const float max, const int quant_min,
    const int quant_max, const int tensor_type,
    const bool ev_quant, typename TTypes<float>::Flat outputs);
extern template struct FakeQuantWithMinMaxArgsFunctor<GPUDevice>;
REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxArgs").Device(DEVICE_GPU),
                        FakeQuantWithMinMaxArgsOp<GPUDevice>);

template <>
void FakeQuantWithMinMaxArgsGradientFunctor<GPUDevice>::operator()(
    const GPUDevice& d, typename TTypes<float>::ConstFlat gradients,
    typename TTypes<float>::ConstFlat inputs, const float min, const float max,
    const int quant_min, const int quant_max, const int tensor_type,
    const bool ev_quant, typename TTypes<float>::Flat backprops);
REGISTER_KERNEL_BUILDER(
    Name("FakeQuantWithMinMaxArgsGradient").Device(DEVICE_GPU),
    FakeQuantWithMinMaxArgsGradientOp<GPUDevice>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


// -----------------------------------------------------------------------------
// Implementation of FakeQuantWithMinMaxVarsOp, see its documentation in
// core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxVarsOp : public OpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsOp(OpKernelConstruction* context)
      : OpKernel::OpKernel(context) {
    int num_bits;
    OP_REQUIRES_OK(context, context->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(
        context, IsNumBitsValid(num_bits),
        InvalidArgument("num_bits must be between 2 and 16, inclusive"));
    bool narrow_range;

    OP_REQUIRES_OK(context, context->GetAttr("narrow_range", &narrow_range));
    OP_REQUIRES_OK(context, context->GetAttr("tensor_type", &tensor_type_));
    OP_REQUIRES_OK(context, context->GetAttr("ev_quant", &ev_quant_));
    quant_min_ = narrow_range ? 1 : 0;
    quant_max_ = (1 << num_bits) - 1;
  }

  void Compute(OpKernelContext* context) override {
    CHECK_EQ(5, context->num_inputs());
    const Tensor& input = context->input(0);
    const Tensor& min = context->input(1);
    const Tensor& max = context->input(2);
    const Tensor& w_scale = context->input(3);
    const Tensor& ip_scale = context->input(4);

    Tensor* output;
    Tensor* op_w_scale;
    Tensor* op_ip_scale;

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, w_scale.shape(), &op_w_scale));
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, ip_scale.shape(), &op_ip_scale));

    FakeQuantWithMinMaxVarsFunctor<Device> functor;
    functor(context->eigen_device<Device>(), input.flat<float>(),
            min.scalar<float>(), max.scalar<float>(), quant_min_, quant_max_,
            tensor_type_, ev_quant_, w_scale.scalar<float>(), ip_scale.scalar<float>(),
            output->flat<float>(), &op_w_scale->flat<float>()(0), &op_ip_scale->flat<float>()(0));
  }

 private:
  int quant_min_;
  int quant_max_;
  int tensor_type_;
  bool ev_quant_;
};

// Implementation of FakeQuantWithMinMaxVarsGradientOp, see its documentation in
// core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxVarsGradientOp : public OpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsGradientOp(OpKernelConstruction* context)
      : OpKernel::OpKernel(context) {
    int num_bits;
    OP_REQUIRES_OK(context, context->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(
        context, IsNumBitsValid(num_bits),
        InvalidArgument("num_bits must be between 2 and 16, inclusive"));
    bool narrow_range;
    OP_REQUIRES_OK(context, context->GetAttr("narrow_range", &narrow_range));
    OP_REQUIRES_OK(context, context->GetAttr("tensor_type", &tensor_type_));
    OP_REQUIRES_OK(context, context->GetAttr("ev_quant", &ev_quant_));
    quant_min_ = narrow_range ? 1 : 0;
    quant_max_ = (1 << num_bits) - 1;
  }

  void Compute(OpKernelContext* context) override {
    CHECK_EQ(6, context->num_inputs());
    const Tensor& gradient = context->input(0);
    const Tensor& input = context->input(1);
    OP_REQUIRES(context, input.IsSameSize(gradient),
                InvalidArgument("gradient and input must be the same size"));
    const Tensor& min = context->input(2);
    const Tensor& max = context->input(3);
    const Tensor& w_scale = context->input(4);
    const Tensor& ip_scale = context->input(5);

    Tensor* grad_wrt_input;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &grad_wrt_input));

    TensorShape scalar_shape;
    Tensor* grad_wrt_min;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, scalar_shape, &grad_wrt_min));

    Tensor* grad_wrt_max;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, scalar_shape, &grad_wrt_max));

    Tensor* grad_wrt_w_scale;
    OP_REQUIRES_OK(context,
                   context->allocate_output(3, scalar_shape, &grad_wrt_w_scale));

    Tensor* grad_wrt_ip_scale;
    OP_REQUIRES_OK(context,
                   context->allocate_output(4, scalar_shape, &grad_wrt_ip_scale));
    FakeQuantWithMinMaxVarsGradientFunctor<Device> functor;
    functor(context->eigen_device<Device>(), gradient.flat<float>(),
            input.flat<float>(), min.scalar<float>(), max.scalar<float>(),
            quant_min_, quant_max_, tensor_type_, ev_quant_, w_scale.scalar<float>(),
            ip_scale.scalar<float>(), grad_wrt_input->flat<float>(),
            grad_wrt_min->scalar<float>(), grad_wrt_max->scalar<float>(),
            grad_wrt_w_scale->scalar<float>(), grad_wrt_ip_scale->scalar<float>());
  }

 private:
  int quant_min_;
  int quant_max_;
  int tensor_type_;
  bool ev_quant_;
};

REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVars").Device(DEVICE_CPU),
                        FakeQuantWithMinMaxVarsOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(
    Name("FakeQuantWithMinMaxVarsGradient").Device(DEVICE_CPU),
    FakeQuantWithMinMaxVarsGradientOp<CPUDevice>);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
template <>
void FakeQuantWithMinMaxVarsFunctor<GPUDevice>::operator()(
    const GPUDevice& d, typename TTypes<float>::ConstFlat inputs,
    typename TTypes<float>::ConstScalar min,
    typename TTypes<float>::ConstScalar max, const int quant_min,
    const int quant_max, const int tensor_type, const bool ev_quant,
    typename TTypes<float>::ConstScalar w_scale,
    typename TTypes<float>::ConstScalar ip_scale,
    typename TTypes<float>::Flat output,
    float* op_w_scale, float* op_ip_scale);
extern template struct FakeQuantWithMinMaxVarsFunctor<GPUDevice>;
REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVars")
                            .Device(DEVICE_GPU)
                            .HostMemory("min")
                            .HostMemory("max")
                            .HostMemory("w_scale")
                            .HostMemory("ip_scale")
                            .HostMemory("op_w_scale")
                            .HostMemory("op_ip_scale"),
                        FakeQuantWithMinMaxVarsOp<GPUDevice>);

template <>
void FakeQuantWithMinMaxVarsGradientFunctor<GPUDevice>::operator()(
    const GPUDevice& d, typename TTypes<float>::ConstFlat gradients,
    typename TTypes<float>::ConstFlat inputs,
    typename TTypes<float>::ConstScalar min,
    typename TTypes<float>::ConstScalar max, const int quant_min,
    const int quant_max, const int tensor_type, const bool ev_quant,
    typename TTypes<float>::ConstScalar w_scale,
    typename TTypes<float>::ConstScalar ip_scale,
    typename TTypes<float>::Flat backprops_wrt_input,
    typename TTypes<float>::Scalar backprop_wrt_min,
    typename TTypes<float>::Scalar backprop_wrt_max,
    typename TTypes<float>::Scalar backprop_wrt_w_scale,
    typename TTypes<float>::Scalar backprop_wrt_ip_scale);
extern template struct FakeQuantWithMinMaxVarsGradientFunctor<GPUDevice>;
REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVarsGradient")
                            .Device(DEVICE_GPU)
                            .HostMemory("min")
                            .HostMemory("max")
                            .HostMemory("w_scale")
                            .HostMemory("ip_scale"),
                        FakeQuantWithMinMaxVarsGradientOp<GPUDevice>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// -----------------------------------------------------------------------------
// Implementation of FakeQuantWithMinMaxVarsPerChannelOp, see its documentation
// in core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxVarsPerChannelOp : public OpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsPerChannelOp(OpKernelConstruction* context)
      : OpKernel::OpKernel(context) {
    int num_bits;
    OP_REQUIRES_OK(context, context->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(
        context, IsNumBitsValid(num_bits),
        InvalidArgument("num_bits must be between 2 and 16, inclusive"));
    bool narrow_range;
    OP_REQUIRES_OK(context, context->GetAttr("narrow_range", &narrow_range));
    OP_REQUIRES_OK(context, context->GetAttr("tensor_type", &tensor_type_));
    OP_REQUIRES_OK(context, context->GetAttr("ev_quant", &ev_quant_));
    quant_min_ = narrow_range ? 1 : 0;
    quant_max_ = (1 << num_bits) - 1;
  }

  void Compute(OpKernelContext* context) override {
    CHECK_EQ(3, context->num_inputs());
    const Tensor& input = context->input(0);
    const int depth = input.dim_size(input.dims() - 1);  // last dimension size.
    const Tensor& min = context->input(1);
    OP_REQUIRES(context, min.dim_size(0) == depth,
                InvalidArgument("min has incorrect size, expected ", depth,
                                " was ", min.dim_size(0)));
    const Tensor& max = context->input(2);
    OP_REQUIRES(context, max.dim_size(0) == depth,
                InvalidArgument("max has incorrect size, expected ", depth,
                                " was ", max.dim_size(0)));

    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    FakeQuantWithMinMaxVarsPerChannelFunctor<Device> functor;
    functor(context->eigen_device<Device>(), input.flat_inner_dims<float, 2>(),
            min.vec<float>(), max.vec<float>(), quant_min_, quant_max_, tensor_type_,
            ev_quant_, output->flat_inner_dims<float, 2>());
  }

 private:
  int quant_min_;
  int quant_max_;
  int tensor_type_;
  bool ev_quant_;
};

// Implementation of FakeQuantWithMinMaxVarsPerChannelGradientOp, see its
// documentation in core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxVarsPerChannelGradientOp : public OpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsPerChannelGradientOp(
      OpKernelConstruction* context)
      : OpKernel::OpKernel(context) {
    int num_bits;
    OP_REQUIRES_OK(context, context->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(
        context, IsNumBitsValid(num_bits),
        InvalidArgument("num_bits must be between 2 and 16, inclusive"));
    bool narrow_range;
    OP_REQUIRES_OK(context, context->GetAttr("narrow_range", &narrow_range));
    quant_min_ = narrow_range ? 1 : 0;
    quant_max_ = (1 << num_bits) - 1;
    OP_REQUIRES_OK(context, context->GetAttr("tensor_type", &tensor_type_));
    OP_REQUIRES_OK(context, context->GetAttr("ev_quant", &ev_quant_));
  }

  void Compute(OpKernelContext* context) override {
    CHECK_EQ(4, context->num_inputs());
    const Tensor& gradient = context->input(0);
    const Tensor& input = context->input(1);
    OP_REQUIRES(context, input.IsSameSize(gradient),
                InvalidArgument("gradient and input must be the same size"));
    const int depth = input.dim_size(input.dims() - 1);  // last dimension size.
    const Tensor& min = context->input(2);
    OP_REQUIRES(context, min.dim_size(0) == depth,
                InvalidArgument("min has incorrect size, expected ", depth,
                                " was ", min.dim_size(0)));
    const Tensor& max = context->input(3);
    OP_REQUIRES(context, max.dim_size(0) == depth,
                InvalidArgument("max has incorrect size, expected ", depth,
                                " was ", max.dim_size(0)));

    Tensor* grad_wrt_input;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &grad_wrt_input));

    TensorShape min_max_shape({input.dim_size(input.dims() - 1)});
    Tensor* grad_wrt_min;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, min_max_shape, &grad_wrt_min));

    Tensor* grad_wrt_max;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, min_max_shape, &grad_wrt_max));
    FakeQuantWithMinMaxVarsPerChannelGradientFunctor<Device> functor;
    functor(
        context->eigen_device<Device>(), gradient.flat_inner_dims<float, 2>(),
        input.flat_inner_dims<float, 2>(), min.vec<float>(), max.vec<float>(),
        quant_min_, quant_max_, tensor_type_, ev_quant_,
        grad_wrt_input->flat_inner_dims<float, 2>(),
        grad_wrt_min->vec<float>(), grad_wrt_max->vec<float>());
  }

 private:
  int tensor_type_;
  int quant_min_;
  int quant_max_;
  bool ev_quant_;
};

REGISTER_KERNEL_BUILDER(
    Name("FakeQuantWithMinMaxVarsPerChannel").Device(DEVICE_CPU),
    FakeQuantWithMinMaxVarsPerChannelOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(
    Name("FakeQuantWithMinMaxVarsPerChannelGradient").Device(DEVICE_CPU),
    FakeQuantWithMinMaxVarsPerChannelGradientOp<CPUDevice>);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
template <>
void FakeQuantWithMinMaxVarsPerChannelFunctor<GPUDevice>::operator()(
    const GPUDevice& d, typename TTypes<float>::ConstMatrix inputs,
    typename TTypes<float>::ConstFlat min,
    typename TTypes<float>::ConstFlat max, const int quant_min,
    const int quant_max,const int tensor_type,
    const bool ev_quant, typename TTypes<float>::Matrix outputs);
extern template struct FakeQuantWithMinMaxVarsPerChannelFunctor<GPUDevice>;

REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVarsPerChannel")
                            .Device(DEVICE_GPU)
                            .HostMemory("min")
                            .HostMemory("max"),
                        FakeQuantWithMinMaxVarsPerChannelOp<GPUDevice>);

template <>
void FakeQuantWithMinMaxVarsPerChannelGradientFunctor<GPUDevice>::operator()(
    const GPUDevice& d, typename TTypes<float>::ConstMatrix gradients,
    typename TTypes<float>::ConstMatrix inputs,
    typename TTypes<float>::ConstVec min, typename TTypes<float>::ConstVec max,
    const int quant_min, const int quant_max, const int tensor_type,
    const bool ev_quant, typename TTypes<float>::Matrix backprops_wrt_input,
    typename TTypes<float>::Vec backprop_wrt_min,
    typename TTypes<float>::Vec backprop_wrt_max);
extern template struct FakeQuantWithMinMaxVarsPerChannelGradientFunctor<
    GPUDevice>;

REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVarsPerChannelGradient")
                            .Device(DEVICE_GPU)
                            .HostMemory("min")
                            .HostMemory("max"),
                        FakeQuantWithMinMaxVarsPerChannelGradientOp<GPUDevice>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
