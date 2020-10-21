namespace tflite {
namespace delegates {
namespace metawarenn {

class ModelBuilder;

class IOpBuilder {
 public:
  virtual ~IOpBuilder() = default;

  // Add the operator to MetaWareNN model
  virtual TfLiteStatus AddToModelBuilder(ModelBuilder& model_builder, int32_t op_type) = 0;
};

// Generate a lookup table with IOpBuilder delegates
// for different onnx operators
std::unordered_map<std::int32_t, std::shared_ptr<IOpBuilder>> CreateOpBuilders();

} //namespace metawarenn
} //namespace delegates
} //namespace tflite
