
#include "Graph/Nodes.h"


namespace glow {
/// Specifies a node whose Input will be copied to Output.This node prevents graph optimizations from eliminating this node and all of its ancestor nodes. Generally intended to save the final result of a network.
class SaveNode final : public Node {
  NodeHandle Input_;
  NodeHandle Output_;

 public:
  enum InputIndices {
    InputIdx = 0,
    OutputIdx = 1,
  };

  enum ResultIndices {
  };

  SaveNode(llvm::StringRef name, NodeValue Input, NodeValue Output)
      : Node(Kinded::Kind::SaveNodeKind, name), Input_(this, Input), Output_(this, Output) {
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getOutput() const { return Output_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SaveNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    if (idx == 1) return true;
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 1; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const SaveNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  Placeholder *getPlaceholder() const;};
} // namespace glow


namespace glow {
/// Performs padding of a given input tensor. The Padding information must be specified for each dimension of the tensor in Pads (start and end padding). In case the padding is negative, it means that the tensor must be cropped. Mode defines how extra padding elements are created. Supported modes are defined in the PaddingMode enum: CONSTANT, REFLECT, EDGE. Value is only used with the CONSTANT mode.
class PadNode final : public Node {
  NodeHandle Input_;
  unsigned_t Mode_;
  std::vector<int> Pads_;
  float Value_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  PadNode(llvm::StringRef name, TypeRef Result , NodeValue Input, unsigned_t Mode, std::vector<int> Pads, float Value)
      : Node(Kinded::Kind::PadNodeKind, name), Input_(this, Input), Mode_(Mode), Pads_(Pads), Value_(Value) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  unsigned_t getMode() const { return Mode_; }
  llvm::ArrayRef<int> getPads() const { return Pads_; }
  float getValue() const { return Value_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::PadNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const PadNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
class ConvolutionGradNode final : public Node {
  NodeHandle Input_;
  NodeHandle Filter_;
  NodeHandle Bias_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;
  std::vector<unsigned_t> Kernels_;
  std::vector<unsigned_t> Strides_;
  std::vector<unsigned_t> Pads_;
  unsigned_t Group_;
  std::vector<unsigned_t> Dilation_;
  glow::ConvolutionLayout Layout_;
  glow::FusedActivation FusedActivation_;
  std::vector<float> FusedActivationArgs_;

 public:
  enum InputIndices {
    InputIdx = 0,
    FilterIdx = 1,
    BiasIdx = 2,
    OriginalOutputForResultIdx = 3,
    GradOfOriginalOutputNamedResultIdx = 4,
  };

  enum ResultIndices {
    GradOfInputNamedInputIdx = 0,
    GradOfInputNamedFilterIdx = 1,
    GradOfInputNamedBiasIdx = 2,
  };

  ConvolutionGradNode(llvm::StringRef name, NodeValue Input, NodeValue Filter, NodeValue Bias, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult, std::vector<unsigned_t> Kernels, std::vector<unsigned_t> Strides, std::vector<unsigned_t> Pads, unsigned_t Group, std::vector<unsigned_t> Dilation, glow::ConvolutionLayout Layout, glow::FusedActivation FusedActivation, std::vector<float> FusedActivationArgs)
      : Node(Kinded::Kind::ConvolutionGradNodeKind, name), Input_(this, Input), Filter_(this, Filter), Bias_(this, Bias), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult), Kernels_(Kernels), Strides_(Strides), Pads_(Pads), Group_(Group), Dilation_(Dilation), Layout_(Layout), FusedActivation_(FusedActivation), FusedActivationArgs_(FusedActivationArgs) {
    addResult(Input.getType());
    addResult(Filter.getType());
    addResult(Bias.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getFilter() const { return Filter_; }
  const NodeValue getBias() const { return Bias_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedInput() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedInput() const { return getNthResult(0); }
  NodeValue getGradOfInputNamedFilter() { return getNthResult(1); }
  const NodeValue getGradOfInputNamedFilter() const { return getNthResult(1); }
  NodeValue getGradOfInputNamedBias() { return getNthResult(2); }
  const NodeValue getGradOfInputNamedBias() const { return getNthResult(2); }
  llvm::ArrayRef<unsigned_t> getKernels() const { return Kernels_; }
  llvm::ArrayRef<unsigned_t> getStrides() const { return Strides_; }
  llvm::ArrayRef<unsigned_t> getPads() const { return Pads_; }
  void setPads(llvm::ArrayRef<unsigned_t> a) {Pads_ = a; }
  unsigned_t getGroup() const { return Group_; }
  void setGroup(unsigned_t a) {Group_ = a; }
  llvm::ArrayRef<unsigned_t> getDilation() const { return Dilation_; }
  glow::ConvolutionLayout getLayout() const { return Layout_; }
  glow::FusedActivation getFusedActivation() const { return FusedActivation_; }
  void setFusedActivation(glow::FusedActivation a) {FusedActivation_ = a; }
  llvm::ArrayRef<float> getFusedActivationArgs() const { return FusedActivationArgs_; }
  void setFusedActivationArgs(llvm::ArrayRef<float> a) {FusedActivationArgs_ = a; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ConvolutionGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const ConvolutionGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs 2D Convolution using a given Input, Filter, and Bias tensors, as well as provided Kernels, Strides, Pads, Group and Dilation. Supported Layouts are defined in the ConvolutionLayout enum: NHWC and NCHW. Supported FusedActivations are defined in the FusedActivation enum.
class ConvolutionNode final : public Node {
  NodeHandle Input_;
  NodeHandle Filter_;
  NodeHandle Bias_;
  std::vector<unsigned_t> Kernels_;
  std::vector<unsigned_t> Strides_;
  std::vector<unsigned_t> Pads_;
  unsigned_t Group_;
  std::vector<unsigned_t> Dilation_;
  glow::ConvolutionLayout Layout_;
  glow::FusedActivation FusedActivation_;
  std::vector<float> FusedActivationArgs_;

 public:
  enum InputIndices {
    InputIdx = 0,
    FilterIdx = 1,
    BiasIdx = 2,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ConvolutionNode(llvm::StringRef name, TypeRef Result , NodeValue Input, NodeValue Filter, NodeValue Bias, std::vector<unsigned_t> Kernels, std::vector<unsigned_t> Strides, std::vector<unsigned_t> Pads, unsigned_t Group, std::vector<unsigned_t> Dilation, glow::ConvolutionLayout Layout, glow::FusedActivation FusedActivation, std::vector<float> FusedActivationArgs)
      : Node(Kinded::Kind::ConvolutionNodeKind, name), Input_(this, Input), Filter_(this, Filter), Bias_(this, Bias), Kernels_(Kernels), Strides_(Strides), Pads_(Pads), Group_(Group), Dilation_(Dilation), Layout_(Layout), FusedActivation_(FusedActivation), FusedActivationArgs_(FusedActivationArgs) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getFilter() const { return Filter_; }
  const NodeValue getBias() const { return Bias_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<unsigned_t> getKernels() const { return Kernels_; }
  llvm::ArrayRef<unsigned_t> getStrides() const { return Strides_; }
  llvm::ArrayRef<unsigned_t> getPads() const { return Pads_; }
  void setPads(llvm::ArrayRef<unsigned_t> a) {Pads_ = a; }
  unsigned_t getGroup() const { return Group_; }
  void setGroup(unsigned_t a) {Group_ = a; }
  llvm::ArrayRef<unsigned_t> getDilation() const { return Dilation_; }
  glow::ConvolutionLayout getLayout() const { return Layout_; }
  glow::FusedActivation getFusedActivation() const { return FusedActivation_; }
  void setFusedActivation(glow::FusedActivation a) {FusedActivation_ = a; }
  llvm::ArrayRef<float> getFusedActivationArgs() const { return FusedActivationArgs_; }
  void setFusedActivationArgs(llvm::ArrayRef<float> a) {FusedActivationArgs_ = a; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ConvolutionNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const ConvolutionNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  bool hasFusedActivation() const;  ConvolutionGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
/// Performs 2D Convolution using a given Input, Filter, and Bias tensors, as well as provided Kernels, Strides, Pads, and Group. The filter channel wise quantization parameters are provided by FilterScales and FilterOffsets while the bias channel wise quantization parameters are provided by BiasScales and BiasOffsets.
class ChannelwiseQuantizedConvolutionNode final : public Node {
  NodeHandle Input_;
  NodeHandle Filter_;
  NodeHandle Bias_;
  NodeHandle FilterScales_;
  NodeHandle FilterOffsets_;
  NodeHandle BiasScales_;
  NodeHandle BiasOffsets_;
  std::vector<unsigned_t> Kernels_;
  std::vector<unsigned_t> Strides_;
  std::vector<unsigned_t> Pads_;
  unsigned_t Group_;
  std::vector<unsigned_t> Dilation_;
  glow::FusedActivation FusedActivation_;
  std::vector<float> FusedActivationArgs_;

 public:
  enum InputIndices {
    InputIdx = 0,
    FilterIdx = 1,
    BiasIdx = 2,
    FilterScalesIdx = 3,
    FilterOffsetsIdx = 4,
    BiasScalesIdx = 5,
    BiasOffsetsIdx = 6,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ChannelwiseQuantizedConvolutionNode(llvm::StringRef name, TypeRef Result , NodeValue Input, NodeValue Filter, NodeValue Bias, NodeValue FilterScales, NodeValue FilterOffsets, NodeValue BiasScales, NodeValue BiasOffsets, std::vector<unsigned_t> Kernels, std::vector<unsigned_t> Strides, std::vector<unsigned_t> Pads, unsigned_t Group, std::vector<unsigned_t> Dilation, glow::FusedActivation FusedActivation, std::vector<float> FusedActivationArgs)
      : Node(Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind, name), Input_(this, Input), Filter_(this, Filter), Bias_(this, Bias), FilterScales_(this, FilterScales), FilterOffsets_(this, FilterOffsets), BiasScales_(this, BiasScales), BiasOffsets_(this, BiasOffsets), Kernels_(Kernels), Strides_(Strides), Pads_(Pads), Group_(Group), Dilation_(Dilation), FusedActivation_(FusedActivation), FusedActivationArgs_(FusedActivationArgs) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getFilter() const { return Filter_; }
  const NodeValue getBias() const { return Bias_; }
  const NodeValue getFilterScales() const { return FilterScales_; }
  const NodeValue getFilterOffsets() const { return FilterOffsets_; }
  const NodeValue getBiasScales() const { return BiasScales_; }
  const NodeValue getBiasOffsets() const { return BiasOffsets_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<unsigned_t> getKernels() const { return Kernels_; }
  void setKernels(llvm::ArrayRef<unsigned_t> a) {Kernels_ = a; }
  llvm::ArrayRef<unsigned_t> getStrides() const { return Strides_; }
  llvm::ArrayRef<unsigned_t> getPads() const { return Pads_; }
  void setPads(llvm::ArrayRef<unsigned_t> a) {Pads_ = a; }
  unsigned_t getGroup() const { return Group_; }
  void setGroup(unsigned_t a) {Group_ = a; }
  llvm::ArrayRef<unsigned_t> getDilation() const { return Dilation_; }
  glow::FusedActivation getFusedActivation() const { return FusedActivation_; }
  void setFusedActivation(glow::FusedActivation a) {FusedActivation_ = a; }
  llvm::ArrayRef<float> getFusedActivationArgs() const { return FusedActivationArgs_; }
  void setFusedActivationArgs(llvm::ArrayRef<float> a) {FusedActivationArgs_ = a; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const ChannelwiseQuantizedConvolutionNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  bool hasFusedActivation() const;};
} // namespace glow


namespace glow {
/// Performs 2D Transposed Convolution using a given Input,Filter, and Bias tensors, as well as provided Kernels,Strides, Pads, and Group.
class ConvTransposeNode final : public Node {
  NodeHandle Input_;
  NodeHandle Filter_;
  NodeHandle Bias_;
  std::vector<unsigned_t> Kernels_;
  std::vector<unsigned_t> Strides_;
  std::vector<unsigned_t> Pads_;
  unsigned_t Group_;
  std::vector<unsigned_t> Dilation_;

 public:
  enum InputIndices {
    InputIdx = 0,
    FilterIdx = 1,
    BiasIdx = 2,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ConvTransposeNode(llvm::StringRef name, TypeRef Result , NodeValue Input, NodeValue Filter, NodeValue Bias, std::vector<unsigned_t> Kernels, std::vector<unsigned_t> Strides, std::vector<unsigned_t> Pads, unsigned_t Group, std::vector<unsigned_t> Dilation)
      : Node(Kinded::Kind::ConvTransposeNodeKind, name), Input_(this, Input), Filter_(this, Filter), Bias_(this, Bias), Kernels_(Kernels), Strides_(Strides), Pads_(Pads), Group_(Group), Dilation_(Dilation) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getFilter() const { return Filter_; }
  const NodeValue getBias() const { return Bias_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<unsigned_t> getKernels() const { return Kernels_; }
  llvm::ArrayRef<unsigned_t> getStrides() const { return Strides_; }
  llvm::ArrayRef<unsigned_t> getPads() const { return Pads_; }
  unsigned_t getGroup() const { return Group_; }
  llvm::ArrayRef<unsigned_t> getDilation() const { return Dilation_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ConvTransposeNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const ConvTransposeNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
class Convolution3DGradNode final : public Node {
  NodeHandle Input_;
  NodeHandle Filter_;
  NodeHandle Bias_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;
  std::vector<unsigned_t> Kernels_;
  std::vector<unsigned_t> Strides_;
  std::vector<unsigned_t> Pads_;
  unsigned_t Group_;

 public:
  enum InputIndices {
    InputIdx = 0,
    FilterIdx = 1,
    BiasIdx = 2,
    OriginalOutputForResultIdx = 3,
    GradOfOriginalOutputNamedResultIdx = 4,
  };

  enum ResultIndices {
    GradOfInputNamedInputIdx = 0,
    GradOfInputNamedFilterIdx = 1,
    GradOfInputNamedBiasIdx = 2,
  };

  Convolution3DGradNode(llvm::StringRef name, NodeValue Input, NodeValue Filter, NodeValue Bias, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult, std::vector<unsigned_t> Kernels, std::vector<unsigned_t> Strides, std::vector<unsigned_t> Pads, unsigned_t Group)
      : Node(Kinded::Kind::Convolution3DGradNodeKind, name), Input_(this, Input), Filter_(this, Filter), Bias_(this, Bias), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult), Kernels_(Kernels), Strides_(Strides), Pads_(Pads), Group_(Group) {
    addResult(Input.getType());
    addResult(Filter.getType());
    addResult(Bias.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getFilter() const { return Filter_; }
  const NodeValue getBias() const { return Bias_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedInput() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedInput() const { return getNthResult(0); }
  NodeValue getGradOfInputNamedFilter() { return getNthResult(1); }
  const NodeValue getGradOfInputNamedFilter() const { return getNthResult(1); }
  NodeValue getGradOfInputNamedBias() { return getNthResult(2); }
  const NodeValue getGradOfInputNamedBias() const { return getNthResult(2); }
  llvm::ArrayRef<unsigned_t> getKernels() const { return Kernels_; }
  llvm::ArrayRef<unsigned_t> getStrides() const { return Strides_; }
  llvm::ArrayRef<unsigned_t> getPads() const { return Pads_; }
  unsigned_t getGroup() const { return Group_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::Convolution3DGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const Convolution3DGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs 3D Convolution using a given Input, Filter, and Bias tensors, as well as provided Kernels, Strides, Pads, and Group.
class Convolution3DNode final : public Node {
  NodeHandle Input_;
  NodeHandle Filter_;
  NodeHandle Bias_;
  std::vector<unsigned_t> Kernels_;
  std::vector<unsigned_t> Strides_;
  std::vector<unsigned_t> Pads_;
  unsigned_t Group_;

 public:
  enum InputIndices {
    InputIdx = 0,
    FilterIdx = 1,
    BiasIdx = 2,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  Convolution3DNode(llvm::StringRef name, TypeRef Result , NodeValue Input, NodeValue Filter, NodeValue Bias, std::vector<unsigned_t> Kernels, std::vector<unsigned_t> Strides, std::vector<unsigned_t> Pads, unsigned_t Group)
      : Node(Kinded::Kind::Convolution3DNodeKind, name), Input_(this, Input), Filter_(this, Filter), Bias_(this, Bias), Kernels_(Kernels), Strides_(Strides), Pads_(Pads), Group_(Group) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getFilter() const { return Filter_; }
  const NodeValue getBias() const { return Bias_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<unsigned_t> getKernels() const { return Kernels_; }
  llvm::ArrayRef<unsigned_t> getStrides() const { return Strides_; }
  llvm::ArrayRef<unsigned_t> getPads() const { return Pads_; }
  unsigned_t getGroup() const { return Group_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::Convolution3DNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const Convolution3DNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  Convolution3DGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
class MaxPoolGradNode final : public Node {
  NodeHandle Input_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;
  NodeHandle OriginalOutputForArgmax_;
  NodeHandle GradOfOriginalOutputNamedArgmax_;
  std::vector<unsigned_t> Kernels_;
  std::vector<unsigned_t> Strides_;
  std::vector<unsigned_t> Pads_;
  unsigned_t Layout_;

 public:
  enum InputIndices {
    InputIdx = 0,
    OriginalOutputForResultIdx = 1,
    GradOfOriginalOutputNamedResultIdx = 2,
    OriginalOutputForArgmaxIdx = 3,
    GradOfOriginalOutputNamedArgmaxIdx = 4,
  };

  enum ResultIndices {
    GradOfInputNamedInputIdx = 0,
  };

  MaxPoolGradNode(llvm::StringRef name, NodeValue Input, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult, NodeValue OriginalOutputForArgmax, NodeValue GradOfOriginalOutputNamedArgmax, std::vector<unsigned_t> Kernels, std::vector<unsigned_t> Strides, std::vector<unsigned_t> Pads, unsigned_t Layout)
      : Node(Kinded::Kind::MaxPoolGradNodeKind, name), Input_(this, Input), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult), OriginalOutputForArgmax_(this, OriginalOutputForArgmax), GradOfOriginalOutputNamedArgmax_(this, GradOfOriginalOutputNamedArgmax), Kernels_(Kernels), Strides_(Strides), Pads_(Pads), Layout_(Layout) {
    addResult(Input.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  const NodeValue getOriginalOutputForArgmax() const { return OriginalOutputForArgmax_; }
  const NodeValue getGradOfOriginalOutputNamedArgmax() const { return GradOfOriginalOutputNamedArgmax_; }
  NodeValue getGradOfInputNamedInput() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedInput() const { return getNthResult(0); }
  llvm::ArrayRef<unsigned_t> getKernels() const { return Kernels_; }
  llvm::ArrayRef<unsigned_t> getStrides() const { return Strides_; }
  llvm::ArrayRef<unsigned_t> getPads() const { return Pads_; }
  void setPads(llvm::ArrayRef<unsigned_t> a) {Pads_ = a; }
  unsigned_t getLayout() const { return Layout_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::MaxPoolGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const MaxPoolGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs a Max Pool with Argmax operation on the Input given provided Kernels, Strides, and Pads. Argmax is a flattened index corresponding to respective max element. Supported layouts are defined in the ConvolutionLayout enum: NHWC and NCHW.
class MaxPoolNode final : public Node {
  NodeHandle Input_;
  std::vector<unsigned_t> Kernels_;
  std::vector<unsigned_t> Strides_;
  std::vector<unsigned_t> Pads_;
  unsigned_t Layout_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
    ArgmaxIdx = 1,
  };

  MaxPoolNode(llvm::StringRef name, TypeRef Result , TypeRef Argmax , NodeValue Input, std::vector<unsigned_t> Kernels, std::vector<unsigned_t> Strides, std::vector<unsigned_t> Pads, unsigned_t Layout)
      : Node(Kinded::Kind::MaxPoolNodeKind, name), Input_(this, Input), Kernels_(Kernels), Strides_(Strides), Pads_(Pads), Layout_(Layout) {
    addResult(Result);
    addResult(Argmax);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  NodeValue getArgmax() { return getNthResult(1); }
  const NodeValue getArgmax() const { return getNthResult(1); }
  llvm::ArrayRef<unsigned_t> getKernels() const { return Kernels_; }
  llvm::ArrayRef<unsigned_t> getStrides() const { return Strides_; }
  llvm::ArrayRef<unsigned_t> getPads() const { return Pads_; }
  void setPads(llvm::ArrayRef<unsigned_t> a) {Pads_ = a; }
  unsigned_t getLayout() const { return Layout_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::MaxPoolNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const MaxPoolNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  MaxPoolGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
/// Finds index of a maximum element along Axis. If KeepDims is not true, the axis is removed from output
class ArgMaxNode final : public Node {
  NodeHandle Input_;
  unsigned_t Axis_;
  bool KeepDims_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ArgMaxNode(llvm::StringRef name, TypeRef Result , NodeValue Input, unsigned_t Axis, bool KeepDims)
      : Node(Kinded::Kind::ArgMaxNodeKind, name), Input_(this, Input), Axis_(Axis), KeepDims_(KeepDims) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  unsigned_t getAxis() const { return Axis_; }
  bool getKeepDims() const { return KeepDims_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ArgMaxNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const ArgMaxNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Finds index of a minimum element along Axis. If KeepDims is not true, the axis is removed from output
class ArgMinNode final : public Node {
  NodeHandle Input_;
  unsigned_t Axis_;
  bool KeepDims_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ArgMinNode(llvm::StringRef name, TypeRef Result , NodeValue Input, unsigned_t Axis, bool KeepDims)
      : Node(Kinded::Kind::ArgMinNodeKind, name), Input_(this, Input), Axis_(Axis), KeepDims_(KeepDims) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  unsigned_t getAxis() const { return Axis_; }
  bool getKeepDims() const { return KeepDims_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ArgMinNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const ArgMinNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
class AvgPoolGradNode final : public Node {
  NodeHandle Input_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;
  std::vector<unsigned_t> Kernels_;
  std::vector<unsigned_t> Strides_;
  std::vector<unsigned_t> Pads_;
  unsigned_t Layout_;
  bool CountIncludePads_;

 public:
  enum InputIndices {
    InputIdx = 0,
    OriginalOutputForResultIdx = 1,
    GradOfOriginalOutputNamedResultIdx = 2,
  };

  enum ResultIndices {
    GradOfInputNamedInputIdx = 0,
  };

  AvgPoolGradNode(llvm::StringRef name, NodeValue Input, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult, std::vector<unsigned_t> Kernels, std::vector<unsigned_t> Strides, std::vector<unsigned_t> Pads, unsigned_t Layout, bool CountIncludePads)
      : Node(Kinded::Kind::AvgPoolGradNodeKind, name), Input_(this, Input), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult), Kernels_(Kernels), Strides_(Strides), Pads_(Pads), Layout_(Layout), CountIncludePads_(CountIncludePads) {
    addResult(Input.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedInput() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedInput() const { return getNthResult(0); }
  llvm::ArrayRef<unsigned_t> getKernels() const { return Kernels_; }
  llvm::ArrayRef<unsigned_t> getStrides() const { return Strides_; }
  llvm::ArrayRef<unsigned_t> getPads() const { return Pads_; }
  void setPads(llvm::ArrayRef<unsigned_t> a) {Pads_ = a; }
  unsigned_t getLayout() const { return Layout_; }
  bool getCountIncludePads() const { return CountIncludePads_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::AvgPoolGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const AvgPoolGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an Average Pool operation on the Input given provided Kernels, Strides, and Pads. Supported layouts are defined in the ConvolutionLayout enum: NHWC, NCHW, NTHWC and NCTHW.
class AvgPoolNode final : public Node {
  NodeHandle Input_;
  std::vector<unsigned_t> Kernels_;
  std::vector<unsigned_t> Strides_;
  std::vector<unsigned_t> Pads_;
  unsigned_t Layout_;
  bool CountIncludePads_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  AvgPoolNode(llvm::StringRef name, TypeRef Result , NodeValue Input, std::vector<unsigned_t> Kernels, std::vector<unsigned_t> Strides, std::vector<unsigned_t> Pads, unsigned_t Layout, bool CountIncludePads)
      : Node(Kinded::Kind::AvgPoolNodeKind, name), Input_(this, Input), Kernels_(Kernels), Strides_(Strides), Pads_(Pads), Layout_(Layout), CountIncludePads_(CountIncludePads) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<unsigned_t> getKernels() const { return Kernels_; }
  llvm::ArrayRef<unsigned_t> getStrides() const { return Strides_; }
  llvm::ArrayRef<unsigned_t> getPads() const { return Pads_; }
  void setPads(llvm::ArrayRef<unsigned_t> a) {Pads_ = a; }
  unsigned_t getLayout() const { return Layout_; }
  bool getCountIncludePads() const { return CountIncludePads_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::AvgPoolNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const AvgPoolNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  AvgPoolGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
class AdaptiveAvgPoolGradNode final : public Node {
  NodeHandle Input_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;

 public:
  enum InputIndices {
    InputIdx = 0,
    OriginalOutputForResultIdx = 1,
    GradOfOriginalOutputNamedResultIdx = 2,
  };

  enum ResultIndices {
    GradOfInputNamedInputIdx = 0,
  };

  AdaptiveAvgPoolGradNode(llvm::StringRef name, NodeValue Input, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult)
      : Node(Kinded::Kind::AdaptiveAvgPoolGradNodeKind, name), Input_(this, Input), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult) {
    addResult(Input.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedInput() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedInput() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::AdaptiveAvgPoolGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const AdaptiveAvgPoolGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an Adaptive Average Pool operation on the Input given
class AdaptiveAvgPoolNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  AdaptiveAvgPoolNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::AdaptiveAvgPoolNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::AdaptiveAvgPoolNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const AdaptiveAvgPoolNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  AdaptiveAvgPoolGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
/// Computes Y = Alpha * A * B + Beta * C where Alpha, Beta are scalars and A, B, C are matrices. If TransposeA or TransposeB is used then A or B is additionally transposed.
class GemmNode final : public Node {
  NodeHandle A_;
  NodeHandle B_;
  NodeHandle C_;
  float Alpha_;
  float Beta_;
  bool TransposeA_;
  bool TransposeB_;

 public:
  enum InputIndices {
    AIdx = 0,
    BIdx = 1,
    CIdx = 2,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  GemmNode(llvm::StringRef name, TypeRef Result , NodeValue A, NodeValue B, NodeValue C, float Alpha, float Beta, bool TransposeA, bool TransposeB)
      : Node(Kinded::Kind::GemmNodeKind, name), A_(this, A), B_(this, B), C_(this, C), Alpha_(Alpha), Beta_(Beta), TransposeA_(TransposeA), TransposeB_(TransposeB) {
    addResult(Result);
  }
  const NodeValue getA() const { return A_; }
  const NodeValue getB() const { return B_; }
  const NodeValue getC() const { return C_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  float getAlpha() const { return Alpha_; }
  float getBeta() const { return Beta_; }
  bool getTransposeA() const { return TransposeA_; }
  bool getTransposeB() const { return TransposeB_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::GemmNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const GemmNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
class FullyConnectedGradNode final : public Node {
  NodeHandle Input_;
  NodeHandle Weights_;
  NodeHandle Bias_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;

 public:
  enum InputIndices {
    InputIdx = 0,
    WeightsIdx = 1,
    BiasIdx = 2,
    OriginalOutputForResultIdx = 3,
    GradOfOriginalOutputNamedResultIdx = 4,
  };

  enum ResultIndices {
    GradOfInputNamedInputIdx = 0,
    GradOfInputNamedWeightsIdx = 1,
    GradOfInputNamedBiasIdx = 2,
  };

  FullyConnectedGradNode(llvm::StringRef name, NodeValue Input, NodeValue Weights, NodeValue Bias, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult)
      : Node(Kinded::Kind::FullyConnectedGradNodeKind, name), Input_(this, Input), Weights_(this, Weights), Bias_(this, Bias), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult) {
    addResult(Input.getType());
    addResult(Weights.getType());
    addResult(Bias.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getWeights() const { return Weights_; }
  const NodeValue getBias() const { return Bias_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedInput() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedInput() const { return getNthResult(0); }
  NodeValue getGradOfInputNamedWeights() { return getNthResult(1); }
  const NodeValue getGradOfInputNamedWeights() const { return getNthResult(1); }
  NodeValue getGradOfInputNamedBias() { return getNthResult(2); }
  const NodeValue getGradOfInputNamedBias() const { return getNthResult(2); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::FullyConnectedGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const FullyConnectedGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Creates a FullyConnected node where the Input tensor and Weights tensor are multiplied, and then the Bias tensor is added to it, producing the Output.
class FullyConnectedNode final : public Node {
  NodeHandle Input_;
  NodeHandle Weights_;
  NodeHandle Bias_;

 public:
  enum InputIndices {
    InputIdx = 0,
    WeightsIdx = 1,
    BiasIdx = 2,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  FullyConnectedNode(llvm::StringRef name, TypeRef Result , NodeValue Input, NodeValue Weights, NodeValue Bias)
      : Node(Kinded::Kind::FullyConnectedNodeKind, name), Input_(this, Input), Weights_(this, Weights), Bias_(this, Bias) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getWeights() const { return Weights_; }
  const NodeValue getBias() const { return Bias_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::FullyConnectedNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const FullyConnectedNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  FullyConnectedGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
/// Creates a RowwiseQuantizedFullyConnected node where the Input matrix and the transpose of Weights matrix are multiplied, and then the Bias vector is broadcast-added to the result. Input, Bias and Result are regularly quantized, while Weights use row-wisequantization.
class RowwiseQuantizedFullyConnectedNode final : public Node {
  NodeHandle Input_;
  NodeHandle Weights_;
  NodeHandle Scales_;
  NodeHandle Offsets_;
  NodeHandle Bias_;

 public:
  enum InputIndices {
    InputIdx = 0,
    WeightsIdx = 1,
    ScalesIdx = 2,
    OffsetsIdx = 3,
    BiasIdx = 4,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  RowwiseQuantizedFullyConnectedNode(llvm::StringRef name, TypeRef Result , NodeValue Input, NodeValue Weights, NodeValue Scales, NodeValue Offsets, NodeValue Bias)
      : Node(Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind, name), Input_(this, Input), Weights_(this, Weights), Scales_(this, Scales), Offsets_(this, Offsets), Bias_(this, Bias) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getWeights() const { return Weights_; }
  const NodeValue getScales() const { return Scales_; }
  const NodeValue getOffsets() const { return Offsets_; }
  const NodeValue getBias() const { return Bias_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const RowwiseQuantizedFullyConnectedNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Creates a DynamicQuantizedFullyConnectedNode which implement the functionality of dynamic_quantization => quantized_fc => dequantize, which support symmteric/asymmetric quantization. Quantize parameters are automatically selected from range of input, while weights are pre-quantized to int8 and bias are whether float or int32
class DynamicQuantizedFullyConnectedNode final : public Node {
  NodeHandle Input_;
  NodeHandle Weights_;
  NodeHandle Bias_;
  bool IsSymmetric_;
  bool IsPerBatchElement_;

 public:
  enum InputIndices {
    InputIdx = 0,
    WeightsIdx = 1,
    BiasIdx = 2,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  DynamicQuantizedFullyConnectedNode(llvm::StringRef name, TypeRef Result , NodeValue Input, NodeValue Weights, NodeValue Bias, bool IsSymmetric, bool IsPerBatchElement)
      : Node(Kinded::Kind::DynamicQuantizedFullyConnectedNodeKind, name), Input_(this, Input), Weights_(this, Weights), Bias_(this, Bias), IsSymmetric_(IsSymmetric), IsPerBatchElement_(IsPerBatchElement) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getWeights() const { return Weights_; }
  const NodeValue getBias() const { return Bias_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  bool getIsSymmetric() const { return IsSymmetric_; }
  bool getIsPerBatchElement() const { return IsPerBatchElement_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::DynamicQuantizedFullyConnectedNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const DynamicQuantizedFullyConnectedNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Creates a DynamicRowwiseQuantizedFullyConnectedNode which implement the functionality of dynamic_quantization => quantized_fc => dequantize, which support symmteric/asymmetric quantization. Quantize parameters are automatically selected from range of input, while weights are pre-rowwise-quantized to int8, whose rowwise params stored in Scales and Offsets, and bias are whether float or int32
class DynamicRowwiseQuantizedFullyConnectedNode final : public Node {
  NodeHandle Input_;
  NodeHandle Weights_;
  NodeHandle Bias_;
  NodeHandle Scales_;
  NodeHandle Offsets_;
  bool IsSymmetric_;
  bool IsPerBatchElement_;

 public:
  enum InputIndices {
    InputIdx = 0,
    WeightsIdx = 1,
    BiasIdx = 2,
    ScalesIdx = 3,
    OffsetsIdx = 4,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  DynamicRowwiseQuantizedFullyConnectedNode(llvm::StringRef name, TypeRef Result , NodeValue Input, NodeValue Weights, NodeValue Bias, NodeValue Scales, NodeValue Offsets, bool IsSymmetric, bool IsPerBatchElement)
      : Node(Kinded::Kind::DynamicRowwiseQuantizedFullyConnectedNodeKind, name), Input_(this, Input), Weights_(this, Weights), Bias_(this, Bias), Scales_(this, Scales), Offsets_(this, Offsets), IsSymmetric_(IsSymmetric), IsPerBatchElement_(IsPerBatchElement) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getWeights() const { return Weights_; }
  const NodeValue getBias() const { return Bias_; }
  const NodeValue getScales() const { return Scales_; }
  const NodeValue getOffsets() const { return Offsets_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  bool getIsSymmetric() const { return IsSymmetric_; }
  bool getIsPerBatchElement() const { return IsPerBatchElement_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::DynamicRowwiseQuantizedFullyConnectedNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const DynamicRowwiseQuantizedFullyConnectedNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
class BatchNormalizationGradNode final : public Node {
  NodeHandle Input_;
  NodeHandle Scale_;
  NodeHandle Bias_;
  NodeHandle Mean_;
  NodeHandle Var_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;
  unsigned_t ChannelIdx_;
  float Epsilon_;
  float Momentum_;

 public:
  enum InputIndices {
    InputIdx = 0,
    ScaleIdx = 1,
    BiasIdx = 2,
    MeanIdx = 3,
    VarIdx = 4,
    OriginalOutputForResultIdx = 5,
    GradOfOriginalOutputNamedResultIdx = 6,
  };

  enum ResultIndices {
    GradOfInputNamedInputIdx = 0,
    GradOfInputNamedScaleIdx = 1,
    GradOfInputNamedBiasIdx = 2,
    GradOfInputNamedMeanIdx = 3,
    GradOfInputNamedVarIdx = 4,
  };

  BatchNormalizationGradNode(llvm::StringRef name, NodeValue Input, NodeValue Scale, NodeValue Bias, NodeValue Mean, NodeValue Var, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult, unsigned_t ChannelIdx, float Epsilon, float Momentum)
      : Node(Kinded::Kind::BatchNormalizationGradNodeKind, name), Input_(this, Input), Scale_(this, Scale), Bias_(this, Bias), Mean_(this, Mean), Var_(this, Var), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult), ChannelIdx_(ChannelIdx), Epsilon_(Epsilon), Momentum_(Momentum) {
    addResult(Input.getType());
    addResult(Scale.getType());
    addResult(Bias.getType());
    addResult(Mean.getType());
    addResult(Var.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getScale() const { return Scale_; }
  const NodeValue getBias() const { return Bias_; }
  const NodeValue getMean() const { return Mean_; }
  const NodeValue getVar() const { return Var_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedInput() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedInput() const { return getNthResult(0); }
  NodeValue getGradOfInputNamedScale() { return getNthResult(1); }
  const NodeValue getGradOfInputNamedScale() const { return getNthResult(1); }
  NodeValue getGradOfInputNamedBias() { return getNthResult(2); }
  const NodeValue getGradOfInputNamedBias() const { return getNthResult(2); }
  NodeValue getGradOfInputNamedMean() { return getNthResult(3); }
  const NodeValue getGradOfInputNamedMean() const { return getNthResult(3); }
  NodeValue getGradOfInputNamedVar() { return getNthResult(4); }
  const NodeValue getGradOfInputNamedVar() const { return getNthResult(4); }
  unsigned_t getChannelIdx() const { return ChannelIdx_; }
  float getEpsilon() const { return Epsilon_; }
  float getMomentum() const { return Momentum_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchNormalizationGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BatchNormalizationGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs batch normalization on the Input tensor with the provided Scale, Bias, Mean, Var, ChannelIdx, Epsilon, and Momentum. Similar to Caffe2 SpatialBN, and ONNX BatchNormalization operator.
class BatchNormalizationNode final : public Node {
  NodeHandle Input_;
  NodeHandle Scale_;
  NodeHandle Bias_;
  NodeHandle Mean_;
  NodeHandle Var_;
  unsigned_t ChannelIdx_;
  float Epsilon_;
  float Momentum_;

 public:
  enum InputIndices {
    InputIdx = 0,
    ScaleIdx = 1,
    BiasIdx = 2,
    MeanIdx = 3,
    VarIdx = 4,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BatchNormalizationNode(llvm::StringRef name, TypeRef Result , NodeValue Input, NodeValue Scale, NodeValue Bias, NodeValue Mean, NodeValue Var, unsigned_t ChannelIdx, float Epsilon, float Momentum)
      : Node(Kinded::Kind::BatchNormalizationNodeKind, name), Input_(this, Input), Scale_(this, Scale), Bias_(this, Bias), Mean_(this, Mean), Var_(this, Var), ChannelIdx_(ChannelIdx), Epsilon_(Epsilon), Momentum_(Momentum) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getScale() const { return Scale_; }
  const NodeValue getBias() const { return Bias_; }
  const NodeValue getMean() const { return Mean_; }
  const NodeValue getVar() const { return Var_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  unsigned_t getChannelIdx() const { return ChannelIdx_; }
  float getEpsilon() const { return Epsilon_; }
  float getMomentum() const { return Momentum_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchNormalizationNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BatchNormalizationNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  BatchNormalizationGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
/// Calculates new normalized mean and variance based on the input mean, variance, and input.
class MeanVarNormalizationNode final : public Node {
  NodeHandle Input_;
  NodeHandle Mean_;
  NodeHandle Var_;
  unsigned_t ChannelIdx_;
  float Momentum_;

 public:
  enum InputIndices {
    InputIdx = 0,
    MeanIdx = 1,
    VarIdx = 2,
  };

  enum ResultIndices {
    NewMeanIdx = 0,
    NewVarIdx = 1,
  };

  MeanVarNormalizationNode(llvm::StringRef name, NodeValue Input, NodeValue Mean, NodeValue Var, unsigned_t ChannelIdx, float Momentum)
      : Node(Kinded::Kind::MeanVarNormalizationNodeKind, name), Input_(this, Input), Mean_(this, Mean), Var_(this, Var), ChannelIdx_(ChannelIdx), Momentum_(Momentum) {
    addResult(Mean.getType());
    addResult(Var.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getMean() const { return Mean_; }
  const NodeValue getVar() const { return Var_; }
  NodeValue getNewMean() { return getNthResult(0); }
  const NodeValue getNewMean() const { return getNthResult(0); }
  NodeValue getNewVar() { return getNthResult(1); }
  const NodeValue getNewVar() const { return getNthResult(1); }
  unsigned_t getChannelIdx() const { return ChannelIdx_; }
  float getMomentum() const { return Momentum_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::MeanVarNormalizationNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const MeanVarNormalizationNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
class LocalResponseNormalizationGradNode final : public Node {
  NodeHandle Input_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;
  unsigned_t HalfWindowSize_;
  float Alpha_;
  float Beta_;
  float K_;

 public:
  enum InputIndices {
    InputIdx = 0,
    OriginalOutputForResultIdx = 1,
    GradOfOriginalOutputNamedResultIdx = 2,
  };

  enum ResultIndices {
    GradOfInputNamedInputIdx = 0,
  };

  LocalResponseNormalizationGradNode(llvm::StringRef name, NodeValue Input, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult, unsigned_t HalfWindowSize, float Alpha, float Beta, float K)
      : Node(Kinded::Kind::LocalResponseNormalizationGradNodeKind, name), Input_(this, Input), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult), HalfWindowSize_(HalfWindowSize), Alpha_(Alpha), Beta_(Beta), K_(K) {
    addResult(Input.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedInput() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedInput() const { return getNthResult(0); }
  unsigned_t getHalfWindowSize() const { return HalfWindowSize_; }
  float getAlpha() const { return Alpha_; }
  float getBeta() const { return Beta_; }
  float getK() const { return K_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::LocalResponseNormalizationGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const LocalResponseNormalizationGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs local response normalization on the Input tensor with the provided Scale, Bias, Mean, Var, ChannelIdx, Epsilon, and Momentum. Similar to Caffe2 and ONNX LRN.
class LocalResponseNormalizationNode final : public Node {
  NodeHandle Input_;
  unsigned_t HalfWindowSize_;
  float Alpha_;
  float Beta_;
  float K_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  LocalResponseNormalizationNode(llvm::StringRef name, NodeValue Input, unsigned_t HalfWindowSize, float Alpha, float Beta, float K)
      : Node(Kinded::Kind::LocalResponseNormalizationNodeKind, name), Input_(this, Input), HalfWindowSize_(HalfWindowSize), Alpha_(Alpha), Beta_(Beta), K_(K) {
    addResult(Input.getType());
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  unsigned_t getHalfWindowSize() const { return HalfWindowSize_; }
  float getAlpha() const { return Alpha_; }
  float getBeta() const { return Beta_; }
  float getK() const { return K_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::LocalResponseNormalizationNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const LocalResponseNormalizationNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  LocalResponseNormalizationGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
/// Performs layer normalization on the Input tensor with the provided Scale, Bias, and Epsilon. Layer sizes are determined by the dimensions of Scale and Bias. Similar to PyTorch layer_norm.
class LayerNormalizationNode final : public Node {
  NodeHandle Input_;
  NodeHandle Scale_;
  NodeHandle Bias_;
  float Epsilon_;

 public:
  enum InputIndices {
    InputIdx = 0,
    ScaleIdx = 1,
    BiasIdx = 2,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  LayerNormalizationNode(llvm::StringRef name, TypeRef Result , NodeValue Input, NodeValue Scale, NodeValue Bias, float Epsilon)
      : Node(Kinded::Kind::LayerNormalizationNodeKind, name), Input_(this, Input), Scale_(this, Scale), Bias_(this, Bias), Epsilon_(Epsilon) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getScale() const { return Scale_; }
  const NodeValue getBias() const { return Bias_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  float getEpsilon() const { return Epsilon_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::LayerNormalizationNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const LayerNormalizationNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Apply box-cox transform for each column for each column in NxD input tensor
class BatchBoxCoxNode final : public Node {
  NodeHandle Input_;
  NodeHandle Lambda1_;
  NodeHandle Lambda2_;
  float Epsilon_;

 public:
  enum InputIndices {
    InputIdx = 0,
    Lambda1Idx = 1,
    Lambda2Idx = 2,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BatchBoxCoxNode(llvm::StringRef name, NodeValue Input, NodeValue Lambda1, NodeValue Lambda2, float Epsilon)
      : Node(Kinded::Kind::BatchBoxCoxNodeKind, name), Input_(this, Input), Lambda1_(this, Lambda1), Lambda2_(this, Lambda2), Epsilon_(Epsilon) {
    addResult(Input.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getLambda1() const { return Lambda1_; }
  const NodeValue getLambda2() const { return Lambda2_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  float getEpsilon() const { return Epsilon_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchBoxCoxNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BatchBoxCoxNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs L2 norm of the Input operand based on Axis.
class VectorNormNode final : public Node {
  NodeHandle Input_;
  unsigned_t Axis_;
  unsigned_t P_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  VectorNormNode(llvm::StringRef name, TypeRef Result , NodeValue Input, unsigned_t Axis, unsigned_t P)
      : Node(Kinded::Kind::VectorNormNodeKind, name), Input_(this, Input), Axis_(Axis), P_(P) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  unsigned_t getAxis() const { return Axis_; }
  unsigned_t getP() const { return P_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::VectorNormNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const VectorNormNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs bucketization on the input given Boundaries
class BucketizeNode final : public Node {
  NodeHandle Input_;
  std::vector<float> Boundaries_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BucketizeNode(llvm::StringRef name, TypeRef Result , NodeValue Input, std::vector<float> Boundaries)
      : Node(Kinded::Kind::BucketizeNodeKind, name), Input_(this, Input), Boundaries_(Boundaries) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<float> getBoundaries() const { return Boundaries_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BucketizeNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BucketizeNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
class SoftMaxGradNode final : public Node {
  NodeHandle Input_;
  NodeHandle Selected_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;

 public:
  enum InputIndices {
    InputIdx = 0,
    SelectedIdx = 1,
    OriginalOutputForResultIdx = 2,
    GradOfOriginalOutputNamedResultIdx = 3,
  };

  enum ResultIndices {
    GradOfInputNamedInputIdx = 0,
    GradOfInputNamedSelectedIdx = 1,
  };

  SoftMaxGradNode(llvm::StringRef name, NodeValue Input, NodeValue Selected, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult)
      : Node(Kinded::Kind::SoftMaxGradNodeKind, name), Input_(this, Input), Selected_(this, Selected), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult) {
    addResult(Input.getType());
    addResult(Selected.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getSelected() const { return Selected_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedInput() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedInput() const { return getNthResult(0); }
  NodeValue getGradOfInputNamedSelected() { return getNthResult(1); }
  const NodeValue getGradOfInputNamedSelected() const { return getNthResult(1); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SoftMaxGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const SoftMaxGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs SoftMax normalization on the Input tensor.
class SoftMaxNode final : public Node {
  NodeHandle Input_;
  NodeHandle Selected_;

 public:
  enum InputIndices {
    InputIdx = 0,
    SelectedIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  SoftMaxNode(llvm::StringRef name, TypeRef Result , NodeValue Input, NodeValue Selected)
      : Node(Kinded::Kind::SoftMaxNodeKind, name), Input_(this, Input), Selected_(this, Selected) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getSelected() const { return Selected_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SoftMaxNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const SoftMaxNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  SoftMaxGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
class LogSoftMaxGradNode final : public Node {
  NodeHandle Input_;
  NodeHandle Selected_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;

 public:
  enum InputIndices {
    InputIdx = 0,
    SelectedIdx = 1,
    OriginalOutputForResultIdx = 2,
    GradOfOriginalOutputNamedResultIdx = 3,
  };

  enum ResultIndices {
    GradOfInputNamedInputIdx = 0,
    GradOfInputNamedSelectedIdx = 1,
  };

  LogSoftMaxGradNode(llvm::StringRef name, NodeValue Input, NodeValue Selected, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult)
      : Node(Kinded::Kind::LogSoftMaxGradNodeKind, name), Input_(this, Input), Selected_(this, Selected), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult) {
    addResult(Input.getType());
    addResult(Selected.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getSelected() const { return Selected_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedInput() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedInput() const { return getNthResult(0); }
  NodeValue getGradOfInputNamedSelected() { return getNthResult(1); }
  const NodeValue getGradOfInputNamedSelected() const { return getNthResult(1); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::LogSoftMaxGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const LogSoftMaxGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs LogSoftMax normalization on the Input tensor.
class LogSoftMaxNode final : public Node {
  NodeHandle Input_;
  NodeHandle Selected_;

 public:
  enum InputIndices {
    InputIdx = 0,
    SelectedIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  LogSoftMaxNode(llvm::StringRef name, TypeRef Result , NodeValue Input, NodeValue Selected)
      : Node(Kinded::Kind::LogSoftMaxNodeKind, name), Input_(this, Input), Selected_(this, Selected) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getSelected() const { return Selected_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::LogSoftMaxNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const LogSoftMaxNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  LogSoftMaxGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
class CrossEntropyLossGradNode final : public Node {
  NodeHandle P_;
  NodeHandle Labels_;
  NodeHandle OriginalOutputForCE_;
  NodeHandle GradOfOriginalOutputNamedCE_;

 public:
  enum InputIndices {
    PIdx = 0,
    LabelsIdx = 1,
    OriginalOutputForCEIdx = 2,
    GradOfOriginalOutputNamedCEIdx = 3,
  };

  enum ResultIndices {
    GradOfInputNamedPIdx = 0,
    GradOfInputNamedLabelsIdx = 1,
  };

  CrossEntropyLossGradNode(llvm::StringRef name, NodeValue P, NodeValue Labels, NodeValue OriginalOutputForCE, NodeValue GradOfOriginalOutputNamedCE)
      : Node(Kinded::Kind::CrossEntropyLossGradNodeKind, name), P_(this, P), Labels_(this, Labels), OriginalOutputForCE_(this, OriginalOutputForCE), GradOfOriginalOutputNamedCE_(this, GradOfOriginalOutputNamedCE) {
    addResult(P.getType());
    addResult(Labels.getType());
  }
  const NodeValue getP() const { return P_; }
  const NodeValue getLabels() const { return Labels_; }
  const NodeValue getOriginalOutputForCE() const { return OriginalOutputForCE_; }
  const NodeValue getGradOfOriginalOutputNamedCE() const { return GradOfOriginalOutputNamedCE_; }
  NodeValue getGradOfInputNamedP() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedP() const { return getNthResult(0); }
  NodeValue getGradOfInputNamedLabels() { return getNthResult(1); }
  const NodeValue getGradOfInputNamedLabels() const { return getNthResult(1); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::CrossEntropyLossGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const CrossEntropyLossGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Computes the average cross entropy loss of the input.
class CrossEntropyLossNode final : public Node {
  NodeHandle P_;
  NodeHandle Labels_;

 public:
  enum InputIndices {
    PIdx = 0,
    LabelsIdx = 1,
  };

  enum ResultIndices {
    CEIdx = 0,
  };

  CrossEntropyLossNode(llvm::StringRef name, TypeRef CE , NodeValue P, NodeValue Labels)
      : Node(Kinded::Kind::CrossEntropyLossNodeKind, name), P_(this, P), Labels_(this, Labels) {
    addResult(CE);
  }
  const NodeValue getP() const { return P_; }
  const NodeValue getLabels() const { return Labels_; }
  NodeValue getCE() { return getNthResult(0); }
  const NodeValue getCE() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::CrossEntropyLossNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const CrossEntropyLossNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  CrossEntropyLossGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
class RegressionGradNode final : public Node {
  NodeHandle Input_;
  NodeHandle Expected_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;

 public:
  enum InputIndices {
    InputIdx = 0,
    ExpectedIdx = 1,
    OriginalOutputForResultIdx = 2,
    GradOfOriginalOutputNamedResultIdx = 3,
  };

  enum ResultIndices {
    GradOfInputNamedInputIdx = 0,
    GradOfInputNamedExpectedIdx = 1,
  };

  RegressionGradNode(llvm::StringRef name, NodeValue Input, NodeValue Expected, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult)
      : Node(Kinded::Kind::RegressionGradNodeKind, name), Input_(this, Input), Expected_(this, Expected), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult) {
    addResult(Input.getType());
    addResult(Expected.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getExpected() const { return Expected_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedInput() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedInput() const { return getNthResult(0); }
  NodeValue getGradOfInputNamedExpected() { return getNthResult(1); }
  const NodeValue getGradOfInputNamedExpected() const { return getNthResult(1); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::RegressionGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const RegressionGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Takes an Input tensor and creates a regression output layer.
class RegressionNode final : public Node {
  NodeHandle Input_;
  NodeHandle Expected_;

 public:
  enum InputIndices {
    InputIdx = 0,
    ExpectedIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  RegressionNode(llvm::StringRef name, NodeValue Input, NodeValue Expected)
      : Node(Kinded::Kind::RegressionNodeKind, name), Input_(this, Input), Expected_(this, Expected) {
    addResult(Input.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getExpected() const { return Expected_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::RegressionNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const RegressionNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  RegressionGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
/// Computes the sigmoid cross entropy between two inputs.
class SigmoidCrossEntropyWithLogitsNode final : public Node {
  NodeHandle Logits_;
  NodeHandle Targets_;

 public:
  enum InputIndices {
    LogitsIdx = 0,
    TargetsIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  SigmoidCrossEntropyWithLogitsNode(llvm::StringRef name, TypeRef Result , NodeValue Logits, NodeValue Targets)
      : Node(Kinded::Kind::SigmoidCrossEntropyWithLogitsNodeKind, name), Logits_(this, Logits), Targets_(this, Targets) {
    addResult(Result);
  }
  const NodeValue getLogits() const { return Logits_; }
  const NodeValue getTargets() const { return Targets_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SigmoidCrossEntropyWithLogitsNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const SigmoidCrossEntropyWithLogitsNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
class AddGradNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
    OriginalOutputForResultIdx = 2,
    GradOfOriginalOutputNamedResultIdx = 3,
  };

  enum ResultIndices {
    GradOfInputNamedLHSIdx = 0,
    GradOfInputNamedRHSIdx = 1,
  };

  AddGradNode(llvm::StringRef name, NodeValue LHS, NodeValue RHS, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult)
      : Node(Kinded::Kind::AddGradNodeKind, name), LHS_(this, LHS), RHS_(this, RHS), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult) {
    addResult(LHS.getType());
    addResult(RHS.getType());
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedLHS() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedLHS() const { return getNthResult(0); }
  NodeValue getGradOfInputNamedRHS() { return getNthResult(1); }
  const NodeValue getGradOfInputNamedRHS() const { return getNthResult(1); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::AddGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const AddGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs Add on the LHS and RHS operands.
class AddNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  AddNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::AddNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::AddNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const AddNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  AddGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
class MulGradNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
    OriginalOutputForResultIdx = 2,
    GradOfOriginalOutputNamedResultIdx = 3,
  };

  enum ResultIndices {
    GradOfInputNamedLHSIdx = 0,
    GradOfInputNamedRHSIdx = 1,
  };

  MulGradNode(llvm::StringRef name, NodeValue LHS, NodeValue RHS, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult)
      : Node(Kinded::Kind::MulGradNodeKind, name), LHS_(this, LHS), RHS_(this, RHS), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult) {
    addResult(LHS.getType());
    addResult(RHS.getType());
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedLHS() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedLHS() const { return getNthResult(0); }
  NodeValue getGradOfInputNamedRHS() { return getNthResult(1); }
  const NodeValue getGradOfInputNamedRHS() const { return getNthResult(1); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::MulGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const MulGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs Mul on the LHS and RHS operands.
class MulNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  MulNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::MulNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::MulNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const MulNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  MulGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
class SubGradNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
    OriginalOutputForResultIdx = 2,
    GradOfOriginalOutputNamedResultIdx = 3,
  };

  enum ResultIndices {
    GradOfInputNamedLHSIdx = 0,
    GradOfInputNamedRHSIdx = 1,
  };

  SubGradNode(llvm::StringRef name, NodeValue LHS, NodeValue RHS, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult)
      : Node(Kinded::Kind::SubGradNodeKind, name), LHS_(this, LHS), RHS_(this, RHS), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult) {
    addResult(LHS.getType());
    addResult(RHS.getType());
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedLHS() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedLHS() const { return getNthResult(0); }
  NodeValue getGradOfInputNamedRHS() { return getNthResult(1); }
  const NodeValue getGradOfInputNamedRHS() const { return getNthResult(1); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SubGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const SubGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs Sub on the LHS and RHS operands.
class SubNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  SubNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::SubNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SubNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const SubNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  SubGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
class DivGradNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
    OriginalOutputForResultIdx = 2,
    GradOfOriginalOutputNamedResultIdx = 3,
  };

  enum ResultIndices {
    GradOfInputNamedLHSIdx = 0,
    GradOfInputNamedRHSIdx = 1,
  };

  DivGradNode(llvm::StringRef name, NodeValue LHS, NodeValue RHS, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult)
      : Node(Kinded::Kind::DivGradNodeKind, name), LHS_(this, LHS), RHS_(this, RHS), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult) {
    addResult(LHS.getType());
    addResult(RHS.getType());
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedLHS() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedLHS() const { return getNthResult(0); }
  NodeValue getGradOfInputNamedRHS() { return getNthResult(1); }
  const NodeValue getGradOfInputNamedRHS() const { return getNthResult(1); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::DivGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const DivGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs Div on the LHS and RHS operands.
class DivNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  DivNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::DivNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::DivNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const DivNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  DivGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
/// Performs Div on the LHS and RHS operands, then Floor. If Truncate is set to true then truncate the quotient to zero instead.
class FloorDivNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;
  bool Truncate_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  FloorDivNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS, bool Truncate)
      : Node(Kinded::Kind::FloorDivNodeKind, name), LHS_(this, LHS), RHS_(this, RHS), Truncate_(Truncate) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  bool getTruncate() const { return Truncate_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::FloorDivNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const FloorDivNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Computes the element-wise remainder of division.
class FmodNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  FmodNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::FmodNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::FmodNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const FmodNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs Max on the LHS and RHS operands.
class MaxNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  MaxNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::MaxNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::MaxNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const MaxNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs Min on the LHS and RHS operands.
class MinNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  MinNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::MinNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::MinNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const MinNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise EQUAL comparison between the LHS and RHS operands.
class CmpEQNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  CmpEQNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::CmpEQNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::CmpEQNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const CmpEQNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise NOT EQUAL comparison between the LHS and RHS operands.
class CmpNEQNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  CmpNEQNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::CmpNEQNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::CmpNEQNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const CmpNEQNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise LESS THAN comparison between the LHS and RHS operands.
class CmpLTNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  CmpLTNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::CmpLTNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::CmpLTNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const CmpLTNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise LESS THAN OR EQUAL comparison between the LHS and RHS operands.
class CmpLTENode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  CmpLTENode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::CmpLTENodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::CmpLTENodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const CmpLTENode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs elementwise pow(LHS, RHS).
class PowNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  PowNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::PowNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::PowNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const PowNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise logical AND between the LHS and RHS operands.
class AndNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  AndNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::AndNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::AndNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const AndNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise bitwise AND between the LHS and RHS operands.
class BitwiseAndNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BitwiseAndNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::BitwiseAndNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BitwiseAndNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const BitwiseAndNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise logical OR between the LHS and RHS operands.
class OrNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  OrNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::OrNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::OrNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const OrNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise bitwise OR between the LHS and RHS operands.
class BitwiseOrNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BitwiseOrNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::BitwiseOrNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BitwiseOrNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const BitwiseOrNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise logical XOR between the LHS and RHS operands.
class XorNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  XorNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::XorNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::XorNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const XorNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise bitwise XOR between the LHS and RHS operands.
class BitwiseXorNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BitwiseXorNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::BitwiseXorNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BitwiseXorNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const BitwiseXorNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise logical NOT of the Input operand.
class NotNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  NotNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::NotNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::NotNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const NotNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise bitwise NOT of the Input operand.
class BitwiseNotNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BitwiseNotNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::BitwiseNotNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BitwiseNotNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const BitwiseNotNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise negation (sign flip) of the Input operand.
class NegNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  NegNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::NegNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::NegNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const NegNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise ABS(x) of the Input operand.
class AbsNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  AbsNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::AbsNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::AbsNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const AbsNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise FLOOR(x) of the Input operand.
class FloorNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  FloorNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::FloorNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::FloorNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const FloorNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise Sign(x) of the Input operand
class SignNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  SignNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::SignNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SignNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const SignNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise CEIL(x) of the Input operand.
class CeilNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  CeilNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::CeilNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::CeilNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const CeilNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise ROUND(x) of the Input operand.
class RoundNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  RoundNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::RoundNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::RoundNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const RoundNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise TRUNCATE(x) of the Input operand.
class TruncateNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  TruncateNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::TruncateNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::TruncateNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const TruncateNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise SQRT(x) of the Input operand.
class SqrtNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  SqrtNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::SqrtNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SqrtNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const SqrtNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise RSQRT(x) = 1 / SQRT(x) of the Input operand.
class RsqrtNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  RsqrtNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::RsqrtNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::RsqrtNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const RsqrtNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise RECIPROCAL(x) = 1 / x of the Input operand.
class ReciprocalNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ReciprocalNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::ReciprocalNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ReciprocalNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const ReciprocalNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise SIN(x) of the Input operand.
class SinNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  SinNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::SinNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SinNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const SinNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise COS(x) of the Input operand.
class CosNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  CosNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::CosNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::CosNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const CosNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs element-wise natural log to the Input.
class LogNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  LogNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::LogNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::LogNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const LogNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise Arccosine(x) of the Input operand.
class AcosNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  AcosNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::AcosNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::AcosNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const AcosNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise Arcsine(x) of the Input operand.
class AsinNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  AsinNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::AsinNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::AsinNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const AsinNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise Arctan(x) of the Input operand.
class AtanNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  AtanNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::AtanNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::AtanNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const AtanNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs an element-wise Erf(x) of the Input operand.
class ErfNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ErfNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::ErfNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ErfNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const ErfNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs element-wise exponential to the Input.
class ExpNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ExpNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::ExpNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ExpNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const ExpNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Computes elementwise: result = log(input / (1 - input)).
class LogitNode final : public Node {
  NodeHandle Input_;
  float Epsilon_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  LogitNode(llvm::StringRef name, TypeRef Result , NodeValue Input, float Epsilon)
      : Node(Kinded::Kind::LogitNodeKind, name), Input_(this, Input), Epsilon_(Epsilon) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  float getEpsilon() const { return Epsilon_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::LogitNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const LogitNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Selects between values on the LHS or RHS, depending on the value of Cond. Cond is generated by the compare instruction, and is target- and type-specific.
class SelectNode final : public Node {
  NodeHandle Cond_;
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    CondIdx = 0,
    LHSIdx = 1,
    RHSIdx = 2,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  SelectNode(llvm::StringRef name, TypeRef Result , NodeValue Cond, NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::SelectNodeKind, name), Cond_(this, Cond), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getCond() const { return Cond_; }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SelectNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const SelectNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Adds the 'Slice' operand to each one of the slices in the batch.
class BatchedAddNode final : public Node {
  NodeHandle Batch_;
  NodeHandle Slice_;

 public:
  enum InputIndices {
    BatchIdx = 0,
    SliceIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BatchedAddNode(llvm::StringRef name, TypeRef Result , NodeValue Batch, NodeValue Slice)
      : Node(Kinded::Kind::BatchedAddNodeKind, name), Batch_(this, Batch), Slice_(this, Slice) {
    addResult(Result);
  }
  const NodeValue getBatch() const { return Batch_; }
  const NodeValue getSlice() const { return Slice_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchedAddNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BatchedAddNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Multiplies the 'Slice' operand to each one of the slices in the batch.
class BatchedMulNode final : public Node {
  NodeHandle Batch_;
  NodeHandle Slice_;

 public:
  enum InputIndices {
    BatchIdx = 0,
    SliceIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BatchedMulNode(llvm::StringRef name, TypeRef Result , NodeValue Batch, NodeValue Slice)
      : Node(Kinded::Kind::BatchedMulNodeKind, name), Batch_(this, Batch), Slice_(this, Slice) {
    addResult(Result);
  }
  const NodeValue getBatch() const { return Batch_; }
  const NodeValue getSlice() const { return Slice_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchedMulNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BatchedMulNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs matrix multiplication between the LHS and RHS.Example: (A, Z) x (Z, B) => (A, B)
class MatMulNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  MatMulNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::MatMulNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::MatMulNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const MatMulNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs batch matrix multiplication between the LHS and RHS. The operands are a stack of two dimensional matrices. Example: (N, A, Z) x (N, Z, B) => (N, A, B)
class BatchMatMulNode final : public Node {
  NodeHandle LHS_;
  NodeHandle RHS_;

 public:
  enum InputIndices {
    LHSIdx = 0,
    RHSIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BatchMatMulNode(llvm::StringRef name, TypeRef Result , NodeValue LHS, NodeValue RHS)
      : Node(Kinded::Kind::BatchMatMulNodeKind, name), LHS_(this, LHS), RHS_(this, RHS) {
    addResult(Result);
  }
  const NodeValue getLHS() const { return LHS_; }
  const NodeValue getRHS() const { return RHS_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchMatMulNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BatchMatMulNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Accumulates all of the layers in the batch and produce a tensor that has the same dimensions as the input tensor without the first dimension.
class BatchedReduceAddNode final : public Node {
  NodeHandle Batch_;
  unsigned_t Axis_;

 public:
  enum InputIndices {
    BatchIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BatchedReduceAddNode(llvm::StringRef name, TypeRef Result , NodeValue Batch, unsigned_t Axis)
      : Node(Kinded::Kind::BatchedReduceAddNodeKind, name), Batch_(this, Batch), Axis_(Axis) {
    addResult(Result);
  }
  const NodeValue getBatch() const { return Batch_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  unsigned_t getAxis() const { return Axis_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchedReduceAddNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BatchedReduceAddNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Accumulates squares of all of the layers in the batch and produce a tensor that has the same dimensions as the input tensor without the first dimension.
class BatchedReduceSumSquareNode final : public Node {
  NodeHandle Batch_;
  unsigned_t Axis_;

 public:
  enum InputIndices {
    BatchIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BatchedReduceSumSquareNode(llvm::StringRef name, TypeRef Result , NodeValue Batch, unsigned_t Axis)
      : Node(Kinded::Kind::BatchedReduceSumSquareNodeKind, name), Batch_(this, Batch), Axis_(Axis) {
    addResult(Result);
  }
  const NodeValue getBatch() const { return Batch_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  unsigned_t getAxis() const { return Axis_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchedReduceSumSquareNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BatchedReduceSumSquareNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs Average Mean operation on the Input given Axes.
class BatchedReduceMeanNode final : public Node {
  NodeHandle Batch_;
  std::vector<unsigned_t> Axes_;

 public:
  enum InputIndices {
    BatchIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BatchedReduceMeanNode(llvm::StringRef name, TypeRef Result , NodeValue Batch, std::vector<unsigned_t> Axes)
      : Node(Kinded::Kind::BatchedReduceMeanNodeKind, name), Batch_(this, Batch), Axes_(Axes) {
    addResult(Result);
  }
  const NodeValue getBatch() const { return Batch_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<unsigned_t> getAxes() const { return Axes_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchedReduceMeanNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BatchedReduceMeanNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs Reduce Min operation on the Input given Axes.
class BatchedReduceMinNode final : public Node {
  NodeHandle Batch_;
  std::vector<unsigned_t> Axes_;

 public:
  enum InputIndices {
    BatchIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BatchedReduceMinNode(llvm::StringRef name, TypeRef Result , NodeValue Batch, std::vector<unsigned_t> Axes)
      : Node(Kinded::Kind::BatchedReduceMinNodeKind, name), Batch_(this, Batch), Axes_(Axes) {
    addResult(Result);
  }
  const NodeValue getBatch() const { return Batch_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<unsigned_t> getAxes() const { return Axes_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchedReduceMinNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BatchedReduceMinNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs Reduce Max operation on the Input given Axes.
class BatchedReduceMaxNode final : public Node {
  NodeHandle Batch_;
  std::vector<unsigned_t> Axes_;

 public:
  enum InputIndices {
    BatchIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BatchedReduceMaxNode(llvm::StringRef name, TypeRef Result , NodeValue Batch, std::vector<unsigned_t> Axes)
      : Node(Kinded::Kind::BatchedReduceMaxNodeKind, name), Batch_(this, Batch), Axes_(Axes) {
    addResult(Result);
  }
  const NodeValue getBatch() const { return Batch_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<unsigned_t> getAxes() const { return Axes_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchedReduceMaxNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BatchedReduceMaxNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Accumulates the product all of the layers in the batch  and produce a tensor that has the same dimensions as  the input tensor without the first dimension.
class BatchedReduceProdNode final : public Node {
  NodeHandle Batch_;
  unsigned_t Axis_;

 public:
  enum InputIndices {
    BatchIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BatchedReduceProdNode(llvm::StringRef name, TypeRef Result , NodeValue Batch, unsigned_t Axis)
      : Node(Kinded::Kind::BatchedReduceProdNodeKind, name), Batch_(this, Batch), Axis_(Axis) {
    addResult(Result);
  }
  const NodeValue getBatch() const { return Batch_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  unsigned_t getAxis() const { return Axis_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchedReduceProdNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BatchedReduceProdNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs Channel shuffle.
class ChannelShuffleNode final : public Node {
  NodeHandle Input_;
  unsigned_t Group_;
  unsigned_t Kernel_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ChannelShuffleNode(llvm::StringRef name, TypeRef Result , NodeValue Input, unsigned_t Group, unsigned_t Kernel)
      : Node(Kinded::Kind::ChannelShuffleNodeKind, name), Input_(this, Input), Group_(Group), Kernel_(Kernel) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  unsigned_t getGroup() const { return Group_; }
  unsigned_t getKernel() const { return Kernel_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ChannelShuffleNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const ChannelShuffleNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs a Cumulative Sum operation over a 1D vector with flags for working in exclusive mode and in reverse. In each case the output size is the same as in input size.e.g (default) [1, 2, 3, 4] -> [1, 3, 6, 10]. (exclusive) [1, 2, 3, 4] -> [0, 1, 3, 6]. (reverse) [1, 2, 3, 4] -> [10, 9, 7, 4]. 
class CumSumNode final : public Node {
  NodeHandle Input_;
  int64_t Dim_;
  unsigned_t Exclusive_;
  unsigned_t Reverse_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  CumSumNode(llvm::StringRef name, TypeRef Result , NodeValue Input, int64_t Dim, unsigned_t Exclusive, unsigned_t Reverse)
      : Node(Kinded::Kind::CumSumNodeKind, name), Input_(this, Input), Dim_(Dim), Exclusive_(Exclusive), Reverse_(Reverse) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  int64_t getDim() const { return Dim_; }
  unsigned_t getExclusive() const { return Exclusive_; }
  unsigned_t getReverse() const { return Reverse_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::CumSumNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const CumSumNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Sums slices of the outermost dimension of Data in groups defined by Lengths. The first Lengths[0] slices are added together and stored in Result[0], the subsequent Lengths[1] slices are added together and stored in Result[1], etc.
class LengthsSumNode final : public Node {
  NodeHandle Data_;
  NodeHandle Lengths_;

 public:
  enum InputIndices {
    DataIdx = 0,
    LengthsIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  LengthsSumNode(llvm::StringRef name, TypeRef Result , NodeValue Data, NodeValue Lengths)
      : Node(Kinded::Kind::LengthsSumNodeKind, name), Data_(this, Data), Lengths_(this, Lengths) {
    addResult(Result);
  }
  const NodeValue getData() const { return Data_; }
  const NodeValue getLengths() const { return Lengths_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::LengthsSumNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const LengthsSumNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
class SparseLengthsSumGradNode final : public Node {
  NodeHandle Data_;
  NodeHandle Indices_;
  NodeHandle Lengths_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;
  glow::LengthsMode LengthsMode_;
  float AvgLength_;

 public:
  enum InputIndices {
    DataIdx = 0,
    IndicesIdx = 1,
    LengthsIdx = 2,
    OriginalOutputForResultIdx = 3,
    GradOfOriginalOutputNamedResultIdx = 4,
  };

  enum ResultIndices {
    GradOfInputNamedDataIdx = 0,
    GradOfInputNamedIndicesIdx = 1,
    GradOfInputNamedLengthsIdx = 2,
  };

  SparseLengthsSumGradNode(llvm::StringRef name, NodeValue Data, NodeValue Indices, NodeValue Lengths, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult, glow::LengthsMode LengthsMode, float AvgLength)
      : Node(Kinded::Kind::SparseLengthsSumGradNodeKind, name), Data_(this, Data), Indices_(this, Indices), Lengths_(this, Lengths), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult), LengthsMode_(LengthsMode), AvgLength_(AvgLength) {
    addResult(Data.getType());
    addResult(Indices.getType());
    addResult(Lengths.getType());
  }
  const NodeValue getData() const { return Data_; }
  const NodeValue getIndices() const { return Indices_; }
  const NodeValue getLengths() const { return Lengths_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedData() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedData() const { return getNthResult(0); }
  NodeValue getGradOfInputNamedIndices() { return getNthResult(1); }
  const NodeValue getGradOfInputNamedIndices() const { return getNthResult(1); }
  NodeValue getGradOfInputNamedLengths() { return getNthResult(2); }
  const NodeValue getGradOfInputNamedLengths() const { return getNthResult(2); }
  glow::LengthsMode getLengthsMode() const { return LengthsMode_; }
  float getAvgLength() const { return AvgLength_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SparseLengthsSumGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const SparseLengthsSumGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Gathers slices of the outer-most dimension of Data indexed by Indices vector, and then accumulates them into len(Lengths) entries: first Lengths[0] slices are aggregated to Result[0], next Lengths[1] slices are aggregated to Result[1], etc. I.e. sum(Lengths) must be equal to len(Indices).
class SparseLengthsSumNode final : public Node {
  NodeHandle Data_;
  NodeHandle Indices_;
  NodeHandle Lengths_;
  glow::LengthsMode LengthsMode_;
  float AvgLength_;

 public:
  enum InputIndices {
    DataIdx = 0,
    IndicesIdx = 1,
    LengthsIdx = 2,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  SparseLengthsSumNode(llvm::StringRef name, TypeRef Result , NodeValue Data, NodeValue Indices, NodeValue Lengths, glow::LengthsMode LengthsMode, float AvgLength)
      : Node(Kinded::Kind::SparseLengthsSumNodeKind, name), Data_(this, Data), Indices_(this, Indices), Lengths_(this, Lengths), LengthsMode_(LengthsMode), AvgLength_(AvgLength) {
    addResult(Result);
  }
  const NodeValue getData() const { return Data_; }
  const NodeValue getIndices() const { return Indices_; }
  const NodeValue getLengths() const { return Lengths_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  glow::LengthsMode getLengthsMode() const { return LengthsMode_; }
  float getAvgLength() const { return AvgLength_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SparseLengthsSumNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const SparseLengthsSumNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  SparseLengthsSumGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
class SparseLengthsWeightedSumGradNode final : public Node {
  NodeHandle Data_;
  NodeHandle Weights_;
  NodeHandle Indices_;
  NodeHandle Lengths_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;
  glow::LengthsMode LengthsMode_;
  float AvgLength_;

 public:
  enum InputIndices {
    DataIdx = 0,
    WeightsIdx = 1,
    IndicesIdx = 2,
    LengthsIdx = 3,
    OriginalOutputForResultIdx = 4,
    GradOfOriginalOutputNamedResultIdx = 5,
  };

  enum ResultIndices {
    GradOfInputNamedDataIdx = 0,
    GradOfInputNamedWeightsIdx = 1,
    GradOfInputNamedIndicesIdx = 2,
    GradOfInputNamedLengthsIdx = 3,
  };

  SparseLengthsWeightedSumGradNode(llvm::StringRef name, NodeValue Data, NodeValue Weights, NodeValue Indices, NodeValue Lengths, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult, glow::LengthsMode LengthsMode, float AvgLength)
      : Node(Kinded::Kind::SparseLengthsWeightedSumGradNodeKind, name), Data_(this, Data), Weights_(this, Weights), Indices_(this, Indices), Lengths_(this, Lengths), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult), LengthsMode_(LengthsMode), AvgLength_(AvgLength) {
    addResult(Data.getType());
    addResult(Weights.getType());
    addResult(Indices.getType());
    addResult(Lengths.getType());
  }
  const NodeValue getData() const { return Data_; }
  const NodeValue getWeights() const { return Weights_; }
  const NodeValue getIndices() const { return Indices_; }
  const NodeValue getLengths() const { return Lengths_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedData() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedData() const { return getNthResult(0); }
  NodeValue getGradOfInputNamedWeights() { return getNthResult(1); }
  const NodeValue getGradOfInputNamedWeights() const { return getNthResult(1); }
  NodeValue getGradOfInputNamedIndices() { return getNthResult(2); }
  const NodeValue getGradOfInputNamedIndices() const { return getNthResult(2); }
  NodeValue getGradOfInputNamedLengths() { return getNthResult(3); }
  const NodeValue getGradOfInputNamedLengths() const { return getNthResult(3); }
  glow::LengthsMode getLengthsMode() const { return LengthsMode_; }
  float getAvgLength() const { return AvgLength_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SparseLengthsWeightedSumGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const SparseLengthsWeightedSumGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Gathers slices of the outer-most dimension of Data indexed by Indices vector, and then accumulates them into len(Lengths) entries: first Lengths[0] slices are aggregated to Result[0], next Lengths[1] slices are aggregated to Result[1], etc. I.e. sum(Lengths) must be equal to len(Indices). Before doing aggregation, each individual slice is scaled by its weight: Result[0] = Weights[0] * Slice(0) + Weights[1] * Slice(1) + ... It implies that len(Weights) == len(Indices).
class SparseLengthsWeightedSumNode final : public Node {
  NodeHandle Data_;
  NodeHandle Weights_;
  NodeHandle Indices_;
  NodeHandle Lengths_;
  glow::LengthsMode LengthsMode_;
  float AvgLength_;

 public:
  enum InputIndices {
    DataIdx = 0,
    WeightsIdx = 1,
    IndicesIdx = 2,
    LengthsIdx = 3,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  SparseLengthsWeightedSumNode(llvm::StringRef name, TypeRef Result , NodeValue Data, NodeValue Weights, NodeValue Indices, NodeValue Lengths, glow::LengthsMode LengthsMode, float AvgLength)
      : Node(Kinded::Kind::SparseLengthsWeightedSumNodeKind, name), Data_(this, Data), Weights_(this, Weights), Indices_(this, Indices), Lengths_(this, Lengths), LengthsMode_(LengthsMode), AvgLength_(AvgLength) {
    addResult(Result);
  }
  const NodeValue getData() const { return Data_; }
  const NodeValue getWeights() const { return Weights_; }
  const NodeValue getIndices() const { return Indices_; }
  const NodeValue getLengths() const { return Lengths_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  glow::LengthsMode getLengthsMode() const { return LengthsMode_; }
  float getAvgLength() const { return AvgLength_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SparseLengthsWeightedSumNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const SparseLengthsWeightedSumNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  SparseLengthsWeightedSumGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
/// Gathers slices of the outer-most dimension of Weights indexed by Indices tensor.
class EmbeddingNode final : public Node {
  NodeHandle Weights_;
  NodeHandle Indices_;
  int64_t PadIdx_;
  bool Scale_;
  bool Sparse_;

 public:
  enum InputIndices {
    WeightsIdx = 0,
    IndicesIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  EmbeddingNode(llvm::StringRef name, TypeRef Result , NodeValue Weights, NodeValue Indices, int64_t PadIdx, bool Scale, bool Sparse)
      : Node(Kinded::Kind::EmbeddingNodeKind, name), Weights_(this, Weights), Indices_(this, Indices), PadIdx_(PadIdx), Scale_(Scale), Sparse_(Sparse) {
    addResult(Result);
  }
  const NodeValue getWeights() const { return Weights_; }
  const NodeValue getIndices() const { return Indices_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  int64_t getPadIdx() const { return PadIdx_; }
  bool getScale() const { return Scale_; }
  bool getSparse() const { return Sparse_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::EmbeddingNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const EmbeddingNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Gathers slices of the outer-most dimension of Data indexed by Indices vector, and then accumulates them into len(Offsets) entries: first slice between Offsets[0] and Offsets[1] (or total length if there's only one elem in Offsets) are aggregated to Result[0], etc. I.e. largest offset must be less than or equal to len(Indices). Before doing aggregation, each individual slice is scaled by its weight: Result[0] = Weights[0] * Slice(0) + Weights[1] * Slice(1) + ... It implies that len(Weights) == len(Indices).
class EmbeddingBagNode final : public Node {
  NodeHandle Data_;
  NodeHandle Weights_;
  NodeHandle Indices_;
  NodeHandle Offsets_;
  bool HasEndOffset_;
  glow::LengthsMode LengthsMode_;
  float AvgLength_;

 public:
  enum InputIndices {
    DataIdx = 0,
    WeightsIdx = 1,
    IndicesIdx = 2,
    OffsetsIdx = 3,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  EmbeddingBagNode(llvm::StringRef name, TypeRef Result , NodeValue Data, NodeValue Weights, NodeValue Indices, NodeValue Offsets, bool HasEndOffset, glow::LengthsMode LengthsMode, float AvgLength)
      : Node(Kinded::Kind::EmbeddingBagNodeKind, name), Data_(this, Data), Weights_(this, Weights), Indices_(this, Indices), Offsets_(this, Offsets), HasEndOffset_(HasEndOffset), LengthsMode_(LengthsMode), AvgLength_(AvgLength) {
    addResult(Result);
  }
  const NodeValue getData() const { return Data_; }
  const NodeValue getWeights() const { return Weights_; }
  const NodeValue getIndices() const { return Indices_; }
  const NodeValue getOffsets() const { return Offsets_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  bool getHasEndOffset() const { return HasEndOffset_; }
  glow::LengthsMode getLengthsMode() const { return LengthsMode_; }
  float getAvgLength() const { return AvgLength_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::EmbeddingBagNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const EmbeddingBagNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Same as FusedRowwiseQuantizedSparseLengthsWeightedSum but using offsets instead of lengths.
class EmbeddingBagByteRowwiseOffsetsNode final : public Node {
  NodeHandle Data_;
  NodeHandle Weights_;
  NodeHandle Indices_;
  NodeHandle Offsets_;
  bool UseFP16Accumulation_;
  bool HasEndOffset_;
  glow::LengthsMode LengthsMode_;
  float AvgLength_;

 public:
  enum InputIndices {
    DataIdx = 0,
    WeightsIdx = 1,
    IndicesIdx = 2,
    OffsetsIdx = 3,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  EmbeddingBagByteRowwiseOffsetsNode(llvm::StringRef name, TypeRef Result , NodeValue Data, NodeValue Weights, NodeValue Indices, NodeValue Offsets, bool UseFP16Accumulation, bool HasEndOffset, glow::LengthsMode LengthsMode, float AvgLength)
      : Node(Kinded::Kind::EmbeddingBagByteRowwiseOffsetsNodeKind, name), Data_(this, Data), Weights_(this, Weights), Indices_(this, Indices), Offsets_(this, Offsets), UseFP16Accumulation_(UseFP16Accumulation), HasEndOffset_(HasEndOffset), LengthsMode_(LengthsMode), AvgLength_(AvgLength) {
    addResult(Result);
  }
  const NodeValue getData() const { return Data_; }
  const NodeValue getWeights() const { return Weights_; }
  const NodeValue getIndices() const { return Indices_; }
  const NodeValue getOffsets() const { return Offsets_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  bool getUseFP16Accumulation() const { return UseFP16Accumulation_; }
  void setUseFP16Accumulation(bool a) {UseFP16Accumulation_ = a; }
  bool getHasEndOffset() const { return HasEndOffset_; }
  glow::LengthsMode getLengthsMode() const { return LengthsMode_; }
  float getAvgLength() const { return AvgLength_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::EmbeddingBagByteRowwiseOffsetsNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const EmbeddingBagByteRowwiseOffsetsNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Gathers slices of the outer-most dimension of Data indexed by Indices vector, and then accumulates them into len(Lengths) entries: first Lengths[0] slices are aggregated to Result[0], next Lengths[1] slices are aggregated to Result[1], etc. I.e. sum(Lengths) must be equal to len(Indices). Before doing aggregation, each individual slice is scaled by its weight: Result[0] = Weights[0] * Slice(0) + Weights[1] * Slice(1) + ... It implies that len(Weights) == len(Indices). The input data is rowwise-quantized, where the Scales and Offsets are 1D tensors of length equal to the first dim of Data.
class RowwiseQuantizedSparseLengthsWeightedSumNode final : public Node {
  NodeHandle Data_;
  NodeHandle Scales_;
  NodeHandle Offsets_;
  NodeHandle Weights_;
  NodeHandle Indices_;
  NodeHandle Lengths_;
  bool UseFP16Accumulation_;
  glow::LengthsMode LengthsMode_;
  float AvgLength_;

 public:
  enum InputIndices {
    DataIdx = 0,
    ScalesIdx = 1,
    OffsetsIdx = 2,
    WeightsIdx = 3,
    IndicesIdx = 4,
    LengthsIdx = 5,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  RowwiseQuantizedSparseLengthsWeightedSumNode(llvm::StringRef name, TypeRef Result , NodeValue Data, NodeValue Scales, NodeValue Offsets, NodeValue Weights, NodeValue Indices, NodeValue Lengths, bool UseFP16Accumulation, glow::LengthsMode LengthsMode, float AvgLength)
      : Node(Kinded::Kind::RowwiseQuantizedSparseLengthsWeightedSumNodeKind, name), Data_(this, Data), Scales_(this, Scales), Offsets_(this, Offsets), Weights_(this, Weights), Indices_(this, Indices), Lengths_(this, Lengths), UseFP16Accumulation_(UseFP16Accumulation), LengthsMode_(LengthsMode), AvgLength_(AvgLength) {
    addResult(Result);
  }
  const NodeValue getData() const { return Data_; }
  const NodeValue getScales() const { return Scales_; }
  const NodeValue getOffsets() const { return Offsets_; }
  const NodeValue getWeights() const { return Weights_; }
  const NodeValue getIndices() const { return Indices_; }
  const NodeValue getLengths() const { return Lengths_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  bool getUseFP16Accumulation() const { return UseFP16Accumulation_; }
  void setUseFP16Accumulation(bool a) {UseFP16Accumulation_ = a; }
  glow::LengthsMode getLengthsMode() const { return LengthsMode_; }
  float getAvgLength() const { return AvgLength_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::RowwiseQuantizedSparseLengthsWeightedSumNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const RowwiseQuantizedSparseLengthsWeightedSumNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Gathers slices of the outer-most dimension of Data indexed by Indices vector, and then accumulates them into len(Lengths) entries: first Lengths[0] slices are aggregated to Result[0], next Lengths[1] slices are aggregated to Result[1], etc. I.e. sum(Lengths) must be equal to len(Indices). Before doing aggregation, each individual slice is scaled by its weight: Result[0] = Weights[0] * Slice(0) + Weights[1] * Slice(1) + ... It implies that len(Weights) == len(Indices). The input data is fused rowwise-quantized, where the Scales and Offsets are appended to the end of each row. Thus, Data must be a two-dimensional tensor.
class FusedRowwiseQuantizedSparseLengthsWeightedSumNode final : public Node {
  NodeHandle Data_;
  NodeHandle Weights_;
  NodeHandle Indices_;
  NodeHandle Lengths_;
  bool UseFP16Accumulation_;
  glow::LengthsMode LengthsMode_;
  float AvgLength_;

 public:
  enum InputIndices {
    DataIdx = 0,
    WeightsIdx = 1,
    IndicesIdx = 2,
    LengthsIdx = 3,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  FusedRowwiseQuantizedSparseLengthsWeightedSumNode(llvm::StringRef name, TypeRef Result , NodeValue Data, NodeValue Weights, NodeValue Indices, NodeValue Lengths, bool UseFP16Accumulation, glow::LengthsMode LengthsMode, float AvgLength)
      : Node(Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind, name), Data_(this, Data), Weights_(this, Weights), Indices_(this, Indices), Lengths_(this, Lengths), UseFP16Accumulation_(UseFP16Accumulation), LengthsMode_(LengthsMode), AvgLength_(AvgLength) {
    addResult(Result);
  }
  const NodeValue getData() const { return Data_; }
  const NodeValue getWeights() const { return Weights_; }
  const NodeValue getIndices() const { return Indices_; }
  const NodeValue getLengths() const { return Lengths_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  bool getUseFP16Accumulation() const { return UseFP16Accumulation_; }
  void setUseFP16Accumulation(bool a) {UseFP16Accumulation_ = a; }
  glow::LengthsMode getLengthsMode() const { return LengthsMode_; }
  float getAvgLength() const { return AvgLength_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const FusedRowwiseQuantizedSparseLengthsWeightedSumNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Gathers slices of the outer-most dimension of Data indexed by Indices vector, and then accumulates them into len(Lengths) entries: first Lengths[0] slices are aggregated to Result[0], next Lengths[1] slices are aggregated to Result[1], etc. I.e. sum(Lengths) must be equal to len(Indices). The input data is fused rowwise-quantized, where the Scales and Offsets are appended to the end of each row. Thus, Data must be a two-dimensional tensor.
class FusedRowwiseQuantizedSparseLengthsSumNode final : public Node {
  NodeHandle Data_;
  NodeHandle Indices_;
  NodeHandle Lengths_;
  bool UseFP16Accumulation_;
  glow::LengthsMode LengthsMode_;
  float AvgLength_;

 public:
  enum InputIndices {
    DataIdx = 0,
    IndicesIdx = 1,
    LengthsIdx = 2,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  FusedRowwiseQuantizedSparseLengthsSumNode(llvm::StringRef name, TypeRef Result , NodeValue Data, NodeValue Indices, NodeValue Lengths, bool UseFP16Accumulation, glow::LengthsMode LengthsMode, float AvgLength)
      : Node(Kinded::Kind::FusedRowwiseQuantizedSparseLengthsSumNodeKind, name), Data_(this, Data), Indices_(this, Indices), Lengths_(this, Lengths), UseFP16Accumulation_(UseFP16Accumulation), LengthsMode_(LengthsMode), AvgLength_(AvgLength) {
    addResult(Result);
  }
  const NodeValue getData() const { return Data_; }
  const NodeValue getIndices() const { return Indices_; }
  const NodeValue getLengths() const { return Lengths_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  bool getUseFP16Accumulation() const { return UseFP16Accumulation_; }
  void setUseFP16Accumulation(bool a) {UseFP16Accumulation_ = a; }
  glow::LengthsMode getLengthsMode() const { return LengthsMode_; }
  float getAvgLength() const { return AvgLength_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::FusedRowwiseQuantizedSparseLengthsSumNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const FusedRowwiseQuantizedSparseLengthsSumNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Given a vector of segment lengths, calculates offsets of each segment and packs them next to the lengths. For the input vector of length N the output is a Nx2 matrix with (offset, lengths) packaged for each segment.
class LengthsToRangesNode final : public Node {
  NodeHandle Lengths_;

 public:
  enum InputIndices {
    LengthsIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  LengthsToRangesNode(llvm::StringRef name, TypeRef Result , NodeValue Lengths)
      : Node(Kinded::Kind::LengthsToRangesNodeKind, name), Lengths_(this, Lengths) {
    addResult(Result);
  }
  const NodeValue getLengths() const { return Lengths_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::LengthsToRangesNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const LengthsToRangesNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Converts an input Lengths 1D vector into a range sequence.
class LengthsRangeFillNode final : public Node {
  NodeHandle Lengths_;

 public:
  enum InputIndices {
    LengthsIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  LengthsRangeFillNode(llvm::StringRef name, TypeRef Result , NodeValue Lengths)
      : Node(Kinded::Kind::LengthsRangeFillNodeKind, name), Lengths_(this, Lengths) {
    addResult(Result);
  }
  const NodeValue getLengths() const { return Lengths_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::LengthsRangeFillNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const LengthsRangeFillNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Converts the sparse representation specified by the pair (Indices, Values) into a dense one. This dense representation contains each value from Values at the corresponding index specified in Indices. Unspecified indices are filled with zeroes. Indices may contain duplicate values and in this case, all of the corresponding values in Values are added together.
class SparseToDenseNode final : public Node {
  NodeHandle Indices_;
  NodeHandle Values_;

 public:
  enum InputIndices {
    IndicesIdx = 0,
    ValuesIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  SparseToDenseNode(llvm::StringRef name, TypeRef Result , NodeValue Indices, NodeValue Values)
      : Node(Kinded::Kind::SparseToDenseNodeKind, name), Indices_(this, Indices), Values_(this, Values) {
    addResult(Result);
  }
  const NodeValue getIndices() const { return Indices_; }
  const NodeValue getValues() const { return Values_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SparseToDenseNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const SparseToDenseNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Converts the sparse representation specified by the pair (Indices, Values) into a dense one, where compacted tensor only contains IDs from given Mask. Indices cannot contain duplicate values. Lengths is used to distinguish elements from different examples of one batch. That is, first Lengths[0] index-value pairs belong to batch's example 0, next Lengths[1] pairs belong to example 1, and so on.
class SparseToDenseMaskNode final : public Node {
  NodeHandle Indices_;
  NodeHandle Values_;
  NodeHandle DefaultValue_;
  NodeHandle Lengths_;
  std::vector<dim_t> Mask_;

 public:
  enum InputIndices {
    IndicesIdx = 0,
    ValuesIdx = 1,
    DefaultValueIdx = 2,
    LengthsIdx = 3,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  SparseToDenseMaskNode(llvm::StringRef name, TypeRef Result , NodeValue Indices, NodeValue Values, NodeValue DefaultValue, NodeValue Lengths, std::vector<dim_t> Mask)
      : Node(Kinded::Kind::SparseToDenseMaskNodeKind, name), Indices_(this, Indices), Values_(this, Values), DefaultValue_(this, DefaultValue), Lengths_(this, Lengths), Mask_(Mask) {
    addResult(Result);
  }
  const NodeValue getIndices() const { return Indices_; }
  const NodeValue getValues() const { return Values_; }
  const NodeValue getDefaultValue() const { return DefaultValue_; }
  const NodeValue getLengths() const { return Lengths_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<dim_t> getMask() const { return Mask_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SparseToDenseMaskNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const SparseToDenseMaskNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Determines whether each element of the Input is NaN and generates a mask that can be consumed by a Select node.
class IsNaNNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  IsNaNNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::IsNaNNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::IsNaNNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const IsNaNNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Replaces NaNs found in Input with Value.
class ReplaceNaNNode final : public Node {
  NodeHandle Input_;
  float Value_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ReplaceNaNNode(llvm::StringRef name, TypeRef Result , NodeValue Input, float Value)
      : Node(Kinded::Kind::ReplaceNaNNodeKind, name), Input_(this, Input), Value_(Value) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  float getValue() const { return Value_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ReplaceNaNNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const ReplaceNaNNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs elementwise modulo operation on the input where each element in the output is the corresponding element in the input data modulo Divisor.
class ModuloNode final : public Node {
  NodeHandle Input_;
  int64_t Divisor_;
  bool SignFollowDivisor_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ModuloNode(llvm::StringRef name, TypeRef Result , NodeValue Input, int64_t Divisor, bool SignFollowDivisor)
      : Node(Kinded::Kind::ModuloNodeKind, name), Input_(this, Input), Divisor_(Divisor), SignFollowDivisor_(SignFollowDivisor) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  int64_t getDivisor() const { return Divisor_; }
  bool getSignFollowDivisor() const { return SignFollowDivisor_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ModuloNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const ModuloNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs batched pairwise dot products of the input vectors
class BatchedPairwiseDotProductNode final : public Node {
  std::vector<NodeHandle> Inputs_;

 public:
  enum InputIndices {
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BatchedPairwiseDotProductNode(llvm::StringRef name, TypeRef Result , std::vector<NodeValue> Inputs)
      : Node(Kinded::Kind::BatchedPairwiseDotProductNodeKind, name) {
    addResult(Result);
    Inputs_.resize(Inputs.size());
    for (size_t idx = 0, e = Inputs.size(); idx < e; ++idx) {
        Inputs_[idx] = Inputs[idx];
        Inputs_[idx].setParent(this);
    }
  }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  NodeValueArrayRef getInputs() const { return Inputs_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchedPairwiseDotProductNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BatchedPairwiseDotProductNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs the gradient operation for BatchedPairwiseDotProduct
class BatchedPairwiseDotProductGradNode final : public Node {
  NodeHandle OutputGrad_;
  std::vector<NodeHandle> OriginalInputs_;

 public:
  enum InputIndices {
    OutputGradIdx = 0,
  };

  enum ResultIndices {
  };

  BatchedPairwiseDotProductGradNode(llvm::StringRef name, NodeValue OutputGrad, std::vector<NodeValue> OriginalInputs)
      : Node(Kinded::Kind::BatchedPairwiseDotProductGradNodeKind, name), OutputGrad_(this, OutputGrad) {
    OriginalInputs_.resize(OriginalInputs.size());
    for (size_t idx = 0, e = OriginalInputs.size(); idx < e; ++idx) {
        OriginalInputs_[idx] = OriginalInputs[idx];
        OriginalInputs_[idx].setParent(this);
    }
  }
  const NodeValue getOutputGrad() const { return OutputGrad_; }
  NodeValueArrayRef getOriginalInputs() const { return OriginalInputs_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchedPairwiseDotProductGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BatchedPairwiseDotProductGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  void addExtraResult(TypeRef T) { addResult(T); }
};
} // namespace glow


namespace glow {
class ReluGradNode final : public Node {
  NodeHandle Input_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;

 public:
  enum InputIndices {
    InputIdx = 0,
    OriginalOutputForResultIdx = 1,
    GradOfOriginalOutputNamedResultIdx = 2,
  };

  enum ResultIndices {
    GradOfInputNamedInputIdx = 0,
  };

  ReluGradNode(llvm::StringRef name, NodeValue Input, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult)
      : Node(Kinded::Kind::ReluGradNodeKind, name), Input_(this, Input), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult) {
    addResult(Input.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedInput() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedInput() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ReluGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const ReluGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Applies ReLU, max(0, x), to each element in the Input tensor.
class ReluNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ReluNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::ReluNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ReluNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const ReluNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  ReluGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
/// Applies GeLU, to each element in the Input tensor.
class GeluNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  GeluNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::GeluNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::GeluNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const GeluNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Clip range of inputs to lie in [Min, Max].
class ClipNode final : public Node {
  NodeHandle Input_;
  float Min_;
  float Max_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ClipNode(llvm::StringRef name, TypeRef Result , NodeValue Input, float Min, float Max)
      : Node(Kinded::Kind::ClipNodeKind, name), Input_(this, Input), Min_(Min), Max_(Max) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  float getMin() const { return Min_; }
  float getMax() const { return Max_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ClipNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const ClipNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Applies PReLU, slope * min(0, x) + max(0, x), to each element in the Input tensor.
class PReluNode final : public Node {
  NodeHandle Input_;
  NodeHandle Slope_;

 public:
  enum InputIndices {
    InputIdx = 0,
    SlopeIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  PReluNode(llvm::StringRef name, TypeRef Result , NodeValue Input, NodeValue Slope)
      : Node(Kinded::Kind::PReluNodeKind, name), Input_(this, Input), Slope_(this, Slope) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getSlope() const { return Slope_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::PReluNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const PReluNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
class SigmoidGradNode final : public Node {
  NodeHandle Input_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;

 public:
  enum InputIndices {
    InputIdx = 0,
    OriginalOutputForResultIdx = 1,
    GradOfOriginalOutputNamedResultIdx = 2,
  };

  enum ResultIndices {
    GradOfInputNamedInputIdx = 0,
  };

  SigmoidGradNode(llvm::StringRef name, NodeValue Input, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult)
      : Node(Kinded::Kind::SigmoidGradNodeKind, name), Input_(this, Input), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult) {
    addResult(Input.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedInput() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedInput() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SigmoidGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const SigmoidGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Applies Sigmoid, 1 / (1 + exp(-x)), to each element in the Input tensor.
class SigmoidNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  SigmoidNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::SigmoidNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SigmoidNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const SigmoidNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  SigmoidGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
/// Applies Swish, X * Sigmoid(X), to each element in the Input tensor.
class SwishNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  SwishNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::SwishNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SwishNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const SwishNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
class TanhGradNode final : public Node {
  NodeHandle Input_;
  NodeHandle OriginalOutputForResult_;
  NodeHandle GradOfOriginalOutputNamedResult_;

 public:
  enum InputIndices {
    InputIdx = 0,
    OriginalOutputForResultIdx = 1,
    GradOfOriginalOutputNamedResultIdx = 2,
  };

  enum ResultIndices {
    GradOfInputNamedInputIdx = 0,
  };

  TanhGradNode(llvm::StringRef name, NodeValue Input, NodeValue OriginalOutputForResult, NodeValue GradOfOriginalOutputNamedResult)
      : Node(Kinded::Kind::TanhGradNodeKind, name), Input_(this, Input), OriginalOutputForResult_(this, OriginalOutputForResult), GradOfOriginalOutputNamedResult_(this, GradOfOriginalOutputNamedResult) {
    addResult(Input.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getOriginalOutputForResult() const { return OriginalOutputForResult_; }
  const NodeValue getGradOfOriginalOutputNamedResult() const { return GradOfOriginalOutputNamedResult_; }
  NodeValue getGradOfInputNamedInput() { return getNthResult(0); }
  const NodeValue getGradOfInputNamedInput() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::TanhGradNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const TanhGradNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Applies hyperbolic tangent to each element in the Input tensor.
class TanhNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  TanhNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::TanhNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::TanhNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const TanhNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  TanhGradNode *getGrad(GraphGradMapper &builder);
};
} // namespace glow


namespace glow {
/// Applies LeakyReLU = x for positive x and alpha * x for negative x to each element in the Input tensor.
class LeakyReluNode final : public Node {
  NodeHandle Input_;
  float Alpha_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  LeakyReluNode(llvm::StringRef name, TypeRef Result , NodeValue Input, float Alpha)
      : Node(Kinded::Kind::LeakyReluNodeKind, name), Input_(this, Input), Alpha_(Alpha) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  float getAlpha() const { return Alpha_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::LeakyReluNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const LeakyReluNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Reshape the Input tensor to shape Dims.
class ReshapeNode final : public Node {
  NodeHandle Input_;
  std::vector<dim_t> Dims_;
  std::string Layout_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ReshapeNode(llvm::StringRef name, TypeRef Result , NodeValue Input, std::vector<dim_t> Dims, std::string Layout)
      : Node(Kinded::Kind::ReshapeNodeKind, name), Input_(this, Input), Dims_(Dims), Layout_(Layout) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<dim_t> getDims() const { return Dims_; }
  std::string getLayout() const { return Layout_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ReshapeNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const ReshapeNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Transpose the Input tensor based on the vector Shuffle, which assigns a new axis for each dimension in Input.
class TransposeNode final : public Node {
  NodeHandle Input_;
  std::vector<unsigned_t> Shuffle_;
  std::string Layout_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  TransposeNode(llvm::StringRef name, TypeRef Result , NodeValue Input, std::vector<unsigned_t> Shuffle, std::string Layout)
      : Node(Kinded::Kind::TransposeNodeKind, name), Input_(this, Input), Shuffle_(Shuffle), Layout_(Layout) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<unsigned_t> getShuffle() const { return Shuffle_; }
  std::string getLayout() const { return Layout_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::TransposeNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const TransposeNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// The concat operator adds two tensors together.
/// The parameter 'dim' specifies the dimension to use when joining the tensors.
class ConcatNode final : public Node {
  std::vector<NodeHandle> Inputs_;
  unsigned_t Dim_;

 public:
  enum InputIndices {
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ConcatNode(llvm::StringRef name, TypeRef Result , std::vector<NodeValue> Inputs, unsigned_t Dim)
      : Node(Kinded::Kind::ConcatNodeKind, name), Dim_(Dim) {
    addResult(Result);
    Inputs_.resize(Inputs.size());
    for (size_t idx = 0, e = Inputs.size(); idx < e; ++idx) {
        Inputs_[idx] = Inputs[idx];
        Inputs_[idx].setParent(this);
    }
  }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  NodeValueArrayRef getInputs() const { return Inputs_; }
  unsigned_t getDim() const { return Dim_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ConcatNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const ConcatNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Produces a slice of the Input tensor. The Start vector defines the starting indices for each dimension from which the slice should be taken. The end index for each dimension is determined from the input type's shape.
class SliceNode final : public Node {
  NodeHandle Input_;
  std::vector<dim_t> Start_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  SliceNode(llvm::StringRef name, TypeRef Result , NodeValue Input, std::vector<dim_t> Start)
      : Node(Kinded::Kind::SliceNodeKind, name), Input_(this, Input), Start_(Start) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<dim_t> getStart() const { return Start_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SliceNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const SliceNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Insert tensor Small into tensor Big given indices Start. Small is inserted Count times along Axis. The resulting Tensor will have the same type as the input Big tensor.
class InsertTensorNode final : public Node {
  NodeHandle Big_;
  NodeHandle Small_;
  std::vector<dim_t> Start_;
  unsigned_t Count_;
  unsigned_t Axis_;

 public:
  enum InputIndices {
    BigIdx = 0,
    SmallIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  InsertTensorNode(llvm::StringRef name, NodeValue Big, NodeValue Small, std::vector<dim_t> Start, unsigned_t Count, unsigned_t Axis)
      : Node(Kinded::Kind::InsertTensorNodeKind, name), Big_(this, Big), Small_(this, Small), Start_(Start), Count_(Count), Axis_(Axis) {
    addResult(Big.getType());
  }
  const NodeValue getBig() const { return Big_; }
  const NodeValue getSmall() const { return Small_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<dim_t> getStart() const { return Start_; }
  unsigned_t getCount() const { return Count_; }
  unsigned_t getAxis() const { return Axis_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::InsertTensorNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const InsertTensorNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Gathers entries of the outer-most dimension of Data indexed by Indices, and concatenates them. Output tensor will have dimensions: {I_0, I_1, ... I_n, D_1, D_2, ... D_m}, where D_i and I_j denote Data and Indices dimensions respectively. If batchDims is not zero, the gather operator will treat the first batchDims as the batch and will concat the result of the gather operation on each sample in the batch.
class GatherNode final : public Node {
  NodeHandle Data_;
  NodeHandle Indices_;
  unsigned_t BatchDims_;

 public:
  enum InputIndices {
    DataIdx = 0,
    IndicesIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  GatherNode(llvm::StringRef name, TypeRef Result , NodeValue Data, NodeValue Indices, unsigned_t BatchDims)
      : Node(Kinded::Kind::GatherNodeKind, name), Data_(this, Data), Indices_(this, Indices), BatchDims_(BatchDims) {
    addResult(Result);
  }
  const NodeValue getData() const { return Data_; }
  const NodeValue getIndices() const { return Indices_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  unsigned_t getBatchDims() const { return BatchDims_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::GatherNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const GatherNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Given Data tensor of rank r >= 1, Indices tensor of rank q >= 1 This operator gathers slices of Data into an output tensor of rank q + r - Indices_shape[-1] - 1 .
class GatherNDNode final : public Node {
  NodeHandle Data_;
  NodeHandle Indices_;

 public:
  enum InputIndices {
    DataIdx = 0,
    IndicesIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  GatherNDNode(llvm::StringRef name, TypeRef Result , NodeValue Data, NodeValue Indices)
      : Node(Kinded::Kind::GatherNDNodeKind, name), Data_(this, Data), Indices_(this, Indices) {
    addResult(Result);
  }
  const NodeValue getData() const { return Data_; }
  const NodeValue getIndices() const { return Indices_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::GatherNDNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const GatherNDNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Gathers entries of Data into Output in groups specified by the elements of Ranges. Each element of Ranges contains a list of pairs of indices of the form (index, length) which specify which entries of data to gather. The ordering of elements in Ranges and of pairs within an element is preserved in Output. Lengths contains the lengths of the ranges gathered by each list of pairs in Ranges.
class GatherRangesNode final : public Node {
  NodeHandle Data_;
  NodeHandle Ranges_;

 public:
  enum InputIndices {
    DataIdx = 0,
    RangesIdx = 1,
  };

  enum ResultIndices {
    OutputIdx = 0,
    LengthsIdx = 1,
  };

  GatherRangesNode(llvm::StringRef name, TypeRef Output , TypeRef Lengths , NodeValue Data, NodeValue Ranges)
      : Node(Kinded::Kind::GatherRangesNodeKind, name), Data_(this, Data), Ranges_(this, Ranges) {
    addResult(Output);
    addResult(Lengths);
  }
  const NodeValue getData() const { return Data_; }
  const NodeValue getRanges() const { return Ranges_; }
  NodeValue getOutput() { return getNthResult(0); }
  const NodeValue getOutput() const { return getNthResult(0); }
  NodeValue getLengths() { return getNthResult(1); }
  const NodeValue getLengths() const { return getNthResult(1); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::GatherRangesNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const GatherRangesNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Copies each slice from Slices into Data at the corresponding index in Indices. For example, given input Data {{1,2},{3,4},{5,6}}, Slices {{-3,-4}}, and Indices {{1}}, the result is {{1,2},{-3,-4},{5,6}}. It also supports multi-dimensional indices. For example, given input Data {{1,2},{3,4},{5,6}}, Slices {-3,-4}, and Indices {{1,0},{1,1}} also produces {{1,2},{-3,-4},{5,6}}. If Cumulative is true, the node adds values from Slices to Data instead of copying. For example, given input Data {{1,2},{3,4},{5,6}}, Slices {{-3,-4}}, and Indices {1}, the result is {{1,2},{0,0},{5,6}}. If an index is specified several times, its updates will be added several times as well.
class ScatterDataNode final : public Node {
  NodeHandle Data_;
  NodeHandle Indices_;
  NodeHandle Slices_;
  bool Cumulative_;

 public:
  enum InputIndices {
    DataIdx = 0,
    IndicesIdx = 1,
    SlicesIdx = 2,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ScatterDataNode(llvm::StringRef name, NodeValue Data, NodeValue Indices, NodeValue Slices, bool Cumulative)
      : Node(Kinded::Kind::ScatterDataNodeKind, name), Data_(this, Data), Indices_(this, Indices), Slices_(this, Slices), Cumulative_(Cumulative) {
    addResult(Data.getType());
  }
  const NodeValue getData() const { return Data_; }
  const NodeValue getIndices() const { return Indices_; }
  const NodeValue getSlices() const { return Slices_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  bool getCumulative() const { return Cumulative_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ScatterDataNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const ScatterDataNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Tile an Input tensor Count times along Axis.
class TileNode final : public Node {
  NodeHandle Input_;
  unsigned_t Count_;
  unsigned_t Axis_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  TileNode(llvm::StringRef name, TypeRef Result , NodeValue Input, unsigned_t Count, unsigned_t Axis)
      : Node(Kinded::Kind::TileNodeKind, name), Input_(this, Input), Count_(Count), Axis_(Axis) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  unsigned_t getCount() const { return Count_; }
  unsigned_t getAxis() const { return Axis_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::TileNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const TileNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Expands each row of the Data to a row of zeros and ones, according to One Hot Encoding. i-th element of Result's row is one iff Values[i] equals to the corresponding element of Data.
class BatchOneHotNode final : public Node {
  NodeHandle Data_;
  NodeHandle Lengths_;
  NodeHandle Values_;

 public:
  enum InputIndices {
    DataIdx = 0,
    LengthsIdx = 1,
    ValuesIdx = 2,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BatchOneHotNode(llvm::StringRef name, TypeRef Result , NodeValue Data, NodeValue Lengths, NodeValue Values)
      : Node(Kinded::Kind::BatchOneHotNodeKind, name), Data_(this, Data), Lengths_(this, Lengths), Values_(this, Values) {
    addResult(Result);
  }
  const NodeValue getData() const { return Data_; }
  const NodeValue getLengths() const { return Lengths_; }
  const NodeValue getValues() const { return Values_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchOneHotNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BatchOneHotNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Given Input tensor of [N,H,W,C], where N is the batch axis, C is the channel or depth, H is the height and W is the width. This produces Output tensor of [N, H/BlockSize, W/BlockSize, C * BlockSize * BlockSize].
class SpaceToDepthNode final : public Node {
  NodeHandle Input_;
  unsigned_t BlockSize_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  SpaceToDepthNode(llvm::StringRef name, TypeRef Result , NodeValue Input, unsigned_t BlockSize)
      : Node(Kinded::Kind::SpaceToDepthNodeKind, name), Input_(this, Input), BlockSize_(BlockSize) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  unsigned_t getBlockSize() const { return BlockSize_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SpaceToDepthNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const SpaceToDepthNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Given Input tensor of 3D, 4D, 5D or 6D, generates an Output tensor with resized spatial dimensions using nearest neighbor interpolation. The Output tensor is of shape floor(input_dimension * scale)
class ResizeNearestNode final : public Node {
  NodeHandle Input_;
  std::vector<float> Scale_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ResizeNearestNode(llvm::StringRef name, TypeRef Result , NodeValue Input, std::vector<float> Scale)
      : Node(Kinded::Kind::ResizeNearestNodeKind, name), Input_(this, Input), Scale_(Scale) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<float> getScale() const { return Scale_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ResizeNearestNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const ResizeNearestNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Given Input tensor of [N,H,W,C], where N is the batch, C is the channel or depth, H is the height and W is the width, Generates an Output tensor with resized spatial dimensions using bilinear neighbor interpolation. The Output tensor is of shape floor(input_dimension * scale)
class ResizeBilinearNode final : public Node {
  NodeHandle Input_;
  std::vector<float> Scale_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ResizeBilinearNode(llvm::StringRef name, TypeRef Result , NodeValue Input, std::vector<float> Scale)
      : Node(Kinded::Kind::ResizeBilinearNodeKind, name), Input_(this, Input), Scale_(Scale) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<float> getScale() const { return Scale_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ResizeBilinearNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const ResizeBilinearNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Broadcast the Input tensor to TargetDim using Axis to indicate the offset between Input dimension and TargetDim
class BroadcastNode final : public Node {
  NodeHandle Input_;
  unsigned_t Axis_;
  std::vector<dim_t> TargetDim_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  BroadcastNode(llvm::StringRef name, TypeRef Result , NodeValue Input, unsigned_t Axis, std::vector<dim_t> TargetDim)
      : Node(Kinded::Kind::BroadcastNodeKind, name), Input_(this, Input), Axis_(Axis), TargetDim_(TargetDim) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  unsigned_t getAxis() const { return Axis_; }
  llvm::ArrayRef<dim_t> getTargetDim() const { return TargetDim_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BroadcastNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BroadcastNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Reverse the order of elements in a tensor along the given axis. The shape of the tensor is preserved, but the elements are reordered. The node is inspired from Python numpy.
class FlipNode final : public Node {
  NodeHandle Input_;
  unsigned_t Axis_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  FlipNode(llvm::StringRef name, TypeRef Result , NodeValue Input, unsigned_t Axis)
      : Node(Kinded::Kind::FlipNodeKind, name), Input_(this, Input), Axis_(Axis) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  unsigned_t getAxis() const { return Axis_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::FlipNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const FlipNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Generate a tensor of a specific type filled with 'Value'.Splat always keep floating point value internally but canquantize it based on the output type.
class SplatNode final : public Node {
  float Value_;

 public:
  enum InputIndices {
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  SplatNode(llvm::StringRef name, TypeRef Result , float Value)
      : Node(Kinded::Kind::SplatNodeKind, name), Value_(Value) {
    addResult(Result);
  }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  float getValue() const { return Value_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SplatNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const SplatNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Generate a tensor of a specific type without initializing it. This is useful when filling a big tensor entirely with multiple small slices using InsertTensor nodes such that the big tensor is not required to be initialized (filled) with some value prior to insertion. This node is intended to remove the overhead associated with the initialization in situations where it is not required.
class TouchNode final : public Node {

 public:
  enum InputIndices {
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  TouchNode(llvm::StringRef name, TypeRef Result )
      : Node(Kinded::Kind::TouchNodeKind, name) {
    addResult(Result);
  }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::TouchNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const TouchNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Stochastic Gradient Descent node used during training. Produces the updated weight that needs to be used instead of Weight for the next iteration.
class SGDNode final : public Node {
  NodeHandle Gradient_;
  NodeHandle Weight_;
  float L1Decay_;
  float L2Decay_;
  float LearningRate_;
  float Momentum_;
  unsigned_t BatchSize_;

 public:
  enum InputIndices {
    GradientIdx = 0,
    WeightIdx = 1,
  };

  enum ResultIndices {
    UpdatedWeightIdx = 0,
  };

  SGDNode(llvm::StringRef name, NodeValue Gradient, NodeValue Weight, float L1Decay, float L2Decay, float LearningRate, float Momentum, unsigned_t BatchSize)
      : Node(Kinded::Kind::SGDNodeKind, name), Gradient_(this, Gradient), Weight_(this, Weight), L1Decay_(L1Decay), L2Decay_(L2Decay), LearningRate_(LearningRate), Momentum_(Momentum), BatchSize_(BatchSize) {
    addResult(Weight.getType());
  }
  const NodeValue getGradient() const { return Gradient_; }
  const NodeValue getWeight() const { return Weight_; }
  NodeValue getUpdatedWeight() { return getNthResult(0); }
  const NodeValue getUpdatedWeight() const { return getNthResult(0); }
  float getL1Decay() const { return L1Decay_; }
  float getL2Decay() const { return L2Decay_; }
  float getLearningRate() const { return LearningRate_; }
  float getMomentum() const { return Momentum_; }
  unsigned_t getBatchSize() const { return BatchSize_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SGDNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 1; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const SGDNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Inserts a TraceEvent for profiling.
class TraceEventNode final : public Node {
  NodeHandle Data_;
  std::string EventName_;
  std::string EventType_;
  unsigned_t Index_;

 public:
  enum InputIndices {
    DataIdx = 0,
  };

  enum ResultIndices {
  };

  TraceEventNode(llvm::StringRef name, NodeValue Data, std::string EventName, std::string EventType, unsigned_t Index)
      : Node(Kinded::Kind::TraceEventNodeKind, name), Data_(this, Data), EventName_(EventName), EventType_(EventType), Index_(Index) {
  }
  const NodeValue getData() const { return Data_; }
  std::string getEventName() const { return EventName_; }
  std::string getEventType() const { return EventType_; }
  unsigned_t getIndex() const { return Index_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::TraceEventNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 1; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const TraceEventNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Generate profile (distribution of values) of the Input tensor. This data is used for quantization of the tensor later on. ProfiledNodeName contains the name of the node which is profiled by the QuantizationProfile node. ProfiledNodeName is helpful as lowering might transform the original graph. ProfiledOutputNumber contains the position of the node's output which gets profiled.
class QuantizationProfileNode final : public Node {
  NodeHandle Input_;
  NodeHandle Histogram_;
  NodeHandle ComputationInfo_;
  std::string ProfiledNodeName_;
  unsigned_t ProfiledOutputNumber_;

 public:
  enum InputIndices {
    InputIdx = 0,
    HistogramIdx = 1,
    ComputationInfoIdx = 2,
  };

  enum ResultIndices {
  };

  QuantizationProfileNode(llvm::StringRef name, NodeValue Input, NodeValue Histogram, NodeValue ComputationInfo, std::string ProfiledNodeName, unsigned_t ProfiledOutputNumber)
      : Node(Kinded::Kind::QuantizationProfileNodeKind, name), Input_(this, Input), Histogram_(this, Histogram), ComputationInfo_(this, ComputationInfo), ProfiledNodeName_(ProfiledNodeName), ProfiledOutputNumber_(ProfiledOutputNumber) {
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getHistogram() const { return Histogram_; }
  const NodeValue getComputationInfo() const { return ComputationInfo_; }
  std::string getProfiledNodeName() const { return ProfiledNodeName_; }
  unsigned_t getProfiledOutputNumber() const { return ProfiledOutputNumber_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::QuantizationProfileNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    if (idx == 2) return true;
    if (idx == 1) return true;
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 1; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const QuantizationProfileNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
  Placeholder *getHistogramPlaceholder() const ;
  Placeholder *getComputationInfoPlaceholder() const;
};
} // namespace glow


namespace glow {
/// Simple mapping between quantized numbers.This can be used as quantized sigmoid or tanh functions.
class IntLookupTableNode final : public Node {
  NodeHandle Input_;
  NodeHandle Mapping_;

 public:
  enum InputIndices {
    InputIdx = 0,
    MappingIdx = 1,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  IntLookupTableNode(llvm::StringRef name, TypeRef Result , NodeValue Input, NodeValue Mapping)
      : Node(Kinded::Kind::IntLookupTableNodeKind, name), Input_(this, Input), Mapping_(this, Mapping) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getMapping() const { return Mapping_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::IntLookupTableNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const IntLookupTableNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Quantize floating point tensor. This operation converts floating point numbers to integers based on the given Scale and Offset. Scale and Offset are deduced from the type of the output.x_q = clip(round(x/Scale) + Offset, -128, 127)
class QuantizeNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  QuantizeNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::QuantizeNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::QuantizeNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const QuantizeNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Convert quantized input tensor into the float representation. x = Scale * (x_q - Offset).
class DequantizeNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  DequantizeNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::DequantizeNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::DequantizeNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const DequantizeNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Rescale the input quantized tensor to a new Scale and Offset. The new Scale and Offset are specified by the output type passed to the constructor
class RescaleQuantizedNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  RescaleQuantizedNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::RescaleQuantizedNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::RescaleQuantizedNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const RescaleQuantizedNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Finds the top K maximal elements for each vector in the tensor. Vectors are defined as the last dimension in the tensor. The input shape {D_0, D_1, ... D_n} results in the outputs {D_0, D_1, ... D_n-1, K}, sorted in non-decreasing order.
class TopKNode final : public Node {
  NodeHandle Input_;
  unsigned_t K_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ValuesIdx = 0,
    IndicesIdx = 1,
  };

  TopKNode(llvm::StringRef name, TypeRef Values , TypeRef Indices , NodeValue Input, unsigned_t K)
      : Node(Kinded::Kind::TopKNodeKind, name), Input_(this, Input), K_(K) {
    addResult(Values);
    addResult(Indices);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getValues() { return getNthResult(0); }
  const NodeValue getValues() const { return getNthResult(0); }
  NodeValue getIndices() { return getNthResult(1); }
  const NodeValue getIndices() const { return getNthResult(1); }
  unsigned_t getK() const { return K_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::TopKNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const TopKNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// A LSTM unit node, take Input as I, F, G, O,takes F from forget gate, I from input gate,O from output gate, G from cell gate and C from cell state. Calulates newC = sigmoid(F) * C + sigmoid(I) * tanh(G), newH = tanh(newC) * sigmoid(O).
class LSTMUnitNode final : public Node {
  NodeHandle Input_;
  NodeHandle C_;

 public:
  enum InputIndices {
    InputIdx = 0,
    CIdx = 1,
  };

  enum ResultIndices {
    newCIdx = 0,
    newHIdx = 1,
  };

  LSTMUnitNode(llvm::StringRef name, NodeValue Input, NodeValue C)
      : Node(Kinded::Kind::LSTMUnitNodeKind, name), Input_(this, Input), C_(this, C) {
    addResult(C.getType());
    addResult(C.getType());
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getC() const { return C_; }
  NodeValue getnewC() { return getNthResult(0); }
  const NodeValue getnewC() const { return getNthResult(0); }
  NodeValue getnewH() { return getNthResult(1); }
  const NodeValue getnewH() const { return getNthResult(1); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::LSTMUnitNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const LSTMUnitNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Convert the input from its current type to the destination type. The input and output types must have the same shapes. Moreover the input and output types must not be quantized types. Quantized types should use the appropriate Quantize, Dequantize, and Rescale nodes.
class ConvertToNode final : public Node {
  NodeHandle Input_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ConvertToNode(llvm::StringRef name, TypeRef Result , NodeValue Input)
      : Node(Kinded::Kind::ConvertToNodeKind, name), Input_(this, Input) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ConvertToNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 1; }
  std::string getDebugDesc() const;
  bool isEqual(const ConvertToNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// This is a node representing an external function call. One possible use of this capability is to pass a source code for a function/kernel. When processing this node, a backend can compile and execute the source code. This node can also be used to pass binary or pointers to executable code. The semantics and implementation of this node not standardized and is very backend-specific.
class ExternalFunctionCallNode final : public Node {
  std::vector<NodeHandle> Inputs_;
  std::string FunctionName_;
  std::string FunctionImpl_;
  std::string FunctionKind_;

 public:
  enum InputIndices {
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ExternalFunctionCallNode(llvm::StringRef name, TypeRef Result , std::vector<NodeValue> Inputs, std::string FunctionName, std::string FunctionImpl, std::string FunctionKind)
      : Node(Kinded::Kind::ExternalFunctionCallNodeKind, name), FunctionName_(FunctionName), FunctionImpl_(FunctionImpl), FunctionKind_(FunctionKind) {
    addResult(Result);
    Inputs_.resize(Inputs.size());
    for (size_t idx = 0, e = Inputs.size(); idx < e; ++idx) {
        Inputs_[idx] = Inputs[idx];
        Inputs_[idx].setParent(this);
    }
  }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  NodeValueArrayRef getInputs() const { return Inputs_; }
  std::string getFunctionName() const { return FunctionName_; }
  std::string getFunctionImpl() const { return FunctionImpl_; }
  std::string getFunctionKind() const { return FunctionKind_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ExternalFunctionCallNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 1; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const ExternalFunctionCallNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Computes the spectrogram of a mono audio signal using given window size and stride. The FFT length used to compute the spectrogram is the next power of 2 (for a window size of 640 the FFT length is 1024). The length of each spectrogram window is FFT_length / 2 + 1. This node is inspired from TensorFlow.
class AudioSpectrogramNode final : public Node {
  NodeHandle Input_;
  NodeHandle Window_;
  NodeHandle TwiddleFactors_;
  NodeHandle BitReverseIndices_;
  NodeHandle ComplexToRealWeights_;
  unsigned_t WindowSize_;
  unsigned_t WindowStride_;
  bool MagnitudeSquared_;

 public:
  enum InputIndices {
    InputIdx = 0,
    WindowIdx = 1,
    TwiddleFactorsIdx = 2,
    BitReverseIndicesIdx = 3,
    ComplexToRealWeightsIdx = 4,
  };

  enum ResultIndices {
    SpectrogramIdx = 0,
  };

  AudioSpectrogramNode(llvm::StringRef name, TypeRef Spectrogram , NodeValue Input, NodeValue Window, NodeValue TwiddleFactors, NodeValue BitReverseIndices, NodeValue ComplexToRealWeights, unsigned_t WindowSize, unsigned_t WindowStride, bool MagnitudeSquared)
      : Node(Kinded::Kind::AudioSpectrogramNodeKind, name), Input_(this, Input), Window_(this, Window), TwiddleFactors_(this, TwiddleFactors), BitReverseIndices_(this, BitReverseIndices), ComplexToRealWeights_(this, ComplexToRealWeights), WindowSize_(WindowSize), WindowStride_(WindowStride), MagnitudeSquared_(MagnitudeSquared) {
    addResult(Spectrogram);
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getWindow() const { return Window_; }
  const NodeValue getTwiddleFactors() const { return TwiddleFactors_; }
  const NodeValue getBitReverseIndices() const { return BitReverseIndices_; }
  const NodeValue getComplexToRealWeights() const { return ComplexToRealWeights_; }
  NodeValue getSpectrogram() { return getNthResult(0); }
  const NodeValue getSpectrogram() const { return getNthResult(0); }
  unsigned_t getWindowSize() const { return WindowSize_; }
  unsigned_t getWindowStride() const { return WindowStride_; }
  bool getMagnitudeSquared() const { return MagnitudeSquared_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::AudioSpectrogramNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const AudioSpectrogramNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Computes the MFCC (Mel Frequency Cepstral Coefficient) for the given spectrogram. This node is mostly used as feature extractor for voice/speech audio data in voice command or keyword spotting applications. The input is assumed to be a power spectrogram and not a magnitude.This node is inspired from TensorFlow.
class MFCCNode final : public Node {
  NodeHandle Spectrogram_;
  NodeHandle MelWeights_;
  NodeHandle MelRanges_;
  NodeHandle DctMat_;
  float SampleRate_;
  float LowerFrequency_;
  float UpperFrequency_;
  unsigned_t FilterBankCount_;
  unsigned_t NumCoefficients_;

 public:
  enum InputIndices {
    SpectrogramIdx = 0,
    MelWeightsIdx = 1,
    MelRangesIdx = 2,
    DctMatIdx = 3,
  };

  enum ResultIndices {
    CoefficientsIdx = 0,
  };

  MFCCNode(llvm::StringRef name, TypeRef Coefficients , NodeValue Spectrogram, NodeValue MelWeights, NodeValue MelRanges, NodeValue DctMat, float SampleRate, float LowerFrequency, float UpperFrequency, unsigned_t FilterBankCount, unsigned_t NumCoefficients)
      : Node(Kinded::Kind::MFCCNodeKind, name), Spectrogram_(this, Spectrogram), MelWeights_(this, MelWeights), MelRanges_(this, MelRanges), DctMat_(this, DctMat), SampleRate_(SampleRate), LowerFrequency_(LowerFrequency), UpperFrequency_(UpperFrequency), FilterBankCount_(FilterBankCount), NumCoefficients_(NumCoefficients) {
    addResult(Coefficients);
  }
  const NodeValue getSpectrogram() const { return Spectrogram_; }
  const NodeValue getMelWeights() const { return MelWeights_; }
  const NodeValue getMelRanges() const { return MelRanges_; }
  const NodeValue getDctMat() const { return DctMat_; }
  NodeValue getCoefficients() { return getNthResult(0); }
  const NodeValue getCoefficients() const { return getNthResult(0); }
  float getSampleRate() const { return SampleRate_; }
  float getLowerFrequency() const { return LowerFrequency_; }
  float getUpperFrequency() const { return UpperFrequency_; }
  unsigned_t getFilterBankCount() const { return FilterBankCount_; }
  unsigned_t getNumCoefficients() const { return NumCoefficients_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::MFCCNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const MFCCNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// This is a mix of ONNX and TF NMSv4. It supports multiple classes and does per class NMS. It also supports TF NMS V4 by outputting indices and scalar tensor with number of valid indices. It pads the rest with global MIN box.
class NonMaxSuppressionNode final : public Node {
  NodeHandle Boxes_;
  NodeHandle Scores_;
  unsigned_t CenterPointBox_;
  unsigned_t MaxOutputBoxesPerClass_;
  float IouThreshold_;
  float ScoreThreshold_;
  bool IsTFVersion4_;

 public:
  enum InputIndices {
    BoxesIdx = 0,
    ScoresIdx = 1,
  };

  enum ResultIndices {
    IndicesIdx = 0,
    NumberOfSelectedIndicesIdx = 1,
  };

  NonMaxSuppressionNode(llvm::StringRef name, TypeRef Indices , TypeRef NumberOfSelectedIndices , NodeValue Boxes, NodeValue Scores, unsigned_t CenterPointBox, unsigned_t MaxOutputBoxesPerClass, float IouThreshold, float ScoreThreshold, bool IsTFVersion4)
      : Node(Kinded::Kind::NonMaxSuppressionNodeKind, name), Boxes_(this, Boxes), Scores_(this, Scores), CenterPointBox_(CenterPointBox), MaxOutputBoxesPerClass_(MaxOutputBoxesPerClass), IouThreshold_(IouThreshold), ScoreThreshold_(ScoreThreshold), IsTFVersion4_(IsTFVersion4) {
    addResult(Indices);
    addResult(NumberOfSelectedIndices);
  }
  const NodeValue getBoxes() const { return Boxes_; }
  const NodeValue getScores() const { return Scores_; }
  NodeValue getIndices() { return getNthResult(0); }
  const NodeValue getIndices() const { return getNthResult(0); }
  NodeValue getNumberOfSelectedIndices() { return getNthResult(1); }
  const NodeValue getNumberOfSelectedIndices() const { return getNthResult(1); }
  unsigned_t getCenterPointBox() const { return CenterPointBox_; }
  unsigned_t getMaxOutputBoxesPerClass() const { return MaxOutputBoxesPerClass_; }
  float getIouThreshold() const { return IouThreshold_; }
  float getScoreThreshold() const { return ScoreThreshold_; }
  bool getIsTFVersion4() const { return IsTFVersion4_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::NonMaxSuppressionNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const NonMaxSuppressionNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Performs region of interest align (ROI) operator. FeatureMap - a tensor of [N,H,W,C]. N is the batch, C is the channel, H is the height, W is the width. Boxes - a tensor of [K,4] or [K,5] with format [[optinal_batch_index] x0, y0, x1, y1]. K is the number of boxes. BatchIndices - a tensor of [K,]. If N > 1 and Box shape is [K,4], BatchIndices must be valid. Output is a tensor with shape [K, OutputHeight, OutputWidth, C]. Aligned - if true, coordinates are aligned to a center of a pixel.
class ROIAlignNode final : public Node {
  NodeHandle FeatureMap_;
  NodeHandle Boxes_;
  NodeHandle BatchIndices_;
  unsigned_t Mode_;
  unsigned_t OutputHeight_;
  unsigned_t OutputWidth_;
  unsigned_t SamplingRatio_;
  float SpatialScale_;
  bool Aligned_;
  bool Rotated_;

 public:
  enum InputIndices {
    FeatureMapIdx = 0,
    BoxesIdx = 1,
    BatchIndicesIdx = 2,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  ROIAlignNode(llvm::StringRef name, TypeRef Result , NodeValue FeatureMap, NodeValue Boxes, NodeValue BatchIndices, unsigned_t Mode, unsigned_t OutputHeight, unsigned_t OutputWidth, unsigned_t SamplingRatio, float SpatialScale, bool Aligned, bool Rotated)
      : Node(Kinded::Kind::ROIAlignNodeKind, name), FeatureMap_(this, FeatureMap), Boxes_(this, Boxes), BatchIndices_(this, BatchIndices), Mode_(Mode), OutputHeight_(OutputHeight), OutputWidth_(OutputWidth), SamplingRatio_(SamplingRatio), SpatialScale_(SpatialScale), Aligned_(Aligned), Rotated_(Rotated) {
    addResult(Result);
  }
  const NodeValue getFeatureMap() const { return FeatureMap_; }
  const NodeValue getBoxes() const { return Boxes_; }
  const NodeValue getBatchIndices() const { return BatchIndices_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  unsigned_t getMode() const { return Mode_; }
  unsigned_t getOutputHeight() const { return OutputHeight_; }
  unsigned_t getOutputWidth() const { return OutputWidth_; }
  unsigned_t getSamplingRatio() const { return SamplingRatio_; }
  float getSpatialScale() const { return SpatialScale_; }
  bool getAligned() const { return Aligned_; }
  bool getRotated() const { return Rotated_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ROIAlignNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const ROIAlignNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// Transform proposal bounding boxes to target bounding box using bounding box regression deltas. Rois tensor's format is: <[optional_batch_index], x1, y1, x2, y2>, shape (M, 4) or (M, 5) where M is the number of Rois. For rotated boxes, this would have an additional angle (in degrees) in the format <[optional_batch_id], ctr_x, ctr_y, w, h, angle> Deltas are of shape (M, K*4) with format <dx, dy, dw, dh>, where K is the number of classes. For rotated Rois: shape (M, K*5), format <dx, dy, dw, dh, da>. ImInfo is of shape <batch_size, 3> with format <img_height, img_width, img_scale>.If proposals from multiple images in a batch are present, they should be grouped sequentially and in incremental order.
class BBoxTransformNode final : public Node {
  NodeHandle Rois_;
  NodeHandle Deltas_;
  NodeHandle ImInfo_;
  std::vector<float> Weights_;
  bool ApplyScale_;
  bool Rotated_;
  bool AngleBoundOn_;
  int64_t AngleBoundLo_;
  int64_t AngleBoundHi_;
  float ClipAngleThresh_;
  bool LegacyPlusOne_;

 public:
  enum InputIndices {
    RoisIdx = 0,
    DeltasIdx = 1,
    ImInfoIdx = 2,
  };

  enum ResultIndices {
    BoxOutIdx = 0,
    RoiBatchSplitsIdx = 1,
  };

  BBoxTransformNode(llvm::StringRef name, TypeRef BoxOut , TypeRef RoiBatchSplits , NodeValue Rois, NodeValue Deltas, NodeValue ImInfo, std::vector<float> Weights, bool ApplyScale, bool Rotated, bool AngleBoundOn, int64_t AngleBoundLo, int64_t AngleBoundHi, float ClipAngleThresh, bool LegacyPlusOne)
      : Node(Kinded::Kind::BBoxTransformNodeKind, name), Rois_(this, Rois), Deltas_(this, Deltas), ImInfo_(this, ImInfo), Weights_(Weights), ApplyScale_(ApplyScale), Rotated_(Rotated), AngleBoundOn_(AngleBoundOn), AngleBoundLo_(AngleBoundLo), AngleBoundHi_(AngleBoundHi), ClipAngleThresh_(ClipAngleThresh), LegacyPlusOne_(LegacyPlusOne) {
    addResult(BoxOut);
    addResult(RoiBatchSplits);
  }
  const NodeValue getRois() const { return Rois_; }
  const NodeValue getDeltas() const { return Deltas_; }
  const NodeValue getImInfo() const { return ImInfo_; }
  NodeValue getBoxOut() { return getNthResult(0); }
  const NodeValue getBoxOut() const { return getNthResult(0); }
  NodeValue getRoiBatchSplits() { return getNthResult(1); }
  const NodeValue getRoiBatchSplits() const { return getNthResult(1); }
  llvm::ArrayRef<float> getWeights() const { return Weights_; }
  bool getApplyScale() const { return ApplyScale_; }
  bool getRotated() const { return Rotated_; }
  bool getAngleBoundOn() const { return AngleBoundOn_; }
  int64_t getAngleBoundLo() const { return AngleBoundLo_; }
  int64_t getAngleBoundHi() const { return AngleBoundHi_; }
  float getClipAngleThresh() const { return ClipAngleThresh_; }
  bool getLegacyPlusOne() const { return LegacyPlusOne_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BBoxTransformNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 1; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const BBoxTransformNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// A Max node with one splat input; CPU specific.
class CPUMaxSplatNode final : public Node {
  NodeHandle Input_;
  float SplatValue_;

 public:
  enum InputIndices {
    InputIdx = 0,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  CPUMaxSplatNode(llvm::StringRef name, NodeValue Input, float SplatValue)
      : Node(Kinded::Kind::CPUMaxSplatNodeKind, name), Input_(this, Input), SplatValue_(SplatValue) {
    addResult(Input.getType());
  }
  const NodeValue getInput() const { return Input_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  float getSplatValue() const { return SplatValue_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::CPUMaxSplatNodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 0; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const CPUMaxSplatNode &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow


namespace glow {
/// This is a cpu-specific convolution implementation where the filter is transposed to the shape [D/8, K, K, C, 8]
class CPUConvDKKC8Node final : public Node {
  NodeHandle Input_;
  NodeHandle Filter_;
  NodeHandle Bias_;
  std::vector<unsigned_t> Kernels_;
  std::vector<unsigned_t> Strides_;
  std::vector<unsigned_t> Pads_;
  unsigned_t Group_;

 public:
  enum InputIndices {
    InputIdx = 0,
    FilterIdx = 1,
    BiasIdx = 2,
  };

  enum ResultIndices {
    ResultIdx = 0,
  };

  CPUConvDKKC8Node(llvm::StringRef name, TypeRef Result , NodeValue Input, NodeValue Filter, NodeValue Bias, std::vector<unsigned_t> Kernels, std::vector<unsigned_t> Strides, std::vector<unsigned_t> Pads, unsigned_t Group)
      : Node(Kinded::Kind::CPUConvDKKC8NodeKind, name), Input_(this, Input), Filter_(this, Filter), Bias_(this, Bias), Kernels_(Kernels), Strides_(Strides), Pads_(Pads), Group_(Group) {
    addResult(Result);
  }
  const NodeValue getInput() const { return Input_; }
  const NodeValue getFilter() const { return Filter_; }
  const NodeValue getBias() const { return Bias_; }
  NodeValue getResult() { return getNthResult(0); }
  const NodeValue getResult() const { return getNthResult(0); }
  llvm::ArrayRef<unsigned_t> getKernels() const { return Kernels_; }
  llvm::ArrayRef<unsigned_t> getStrides() const { return Strides_; }
  llvm::ArrayRef<unsigned_t> getPads() const { return Pads_; }
  unsigned_t getGroup() const { return Group_; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::CPUConvDKKC8NodeKind;
  }


  bool isOverwrittenNthInput(unsigned idx) const {
    return false;
  }

  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const { return 0; }
  bool isCanonical() const { return 0; }
  bool isDataParallel() const { return 0; }
  std::string getDebugDesc() const;
  bool isEqual(const CPUConvDKKC8Node &other) const;
  llvm::hash_code getHash() const;
  void visit(Node *parent, NodeWalker *visitor);
  Node* clone() const;
  bool verify() const;
};
} // namespace glow
