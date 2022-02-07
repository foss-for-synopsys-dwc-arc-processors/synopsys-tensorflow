#include "metawarenn_lib/mwnnconvert/parse_mwnnproto.h"
#include "metawarenn_lib/inference_engine/mwnn_inference_engine.h"
#include "metawarenn_lib/inference_engine/mwnn_builder.h"
#include "metawarenn_lib/logger/logger.h"

int main(int argc, char *argv[]) {
  const char* model_path = argv[1];
  std::cout << model_path;
  std::shared_ptr<::metawarenn::Graph> mwnn_graph = ::metawarenn::read_mwnn_graph(std::string(model_path));

  #if INFERENCE_ENGINE
    std::shared_ptr<metawarenn::Builder> inference_builder_ = std::make_shared<metawarenn::Builder>();
    std::shared_ptr<metawarenn::OptimizationProfile> optimization_profile_ = nullptr;
    std::shared_ptr<metawarenn::BuilderConfig> builder_config_;    
    std::shared_ptr<metawarenn::InferenceEngine> inference_engine_;
    std::shared_ptr<metawarenn::ExecutionContext> execution_context_;

    metawarenn::Logger* logger = inference_builder_->GetLogger();
    // Comment Below line to run the Logger in INFO level
    logger->SetLogLevel(metawarenn::logtype::DEBUG);

    builder_config_ = inference_builder_->CreateBuilderConfig();
    inference_engine_ = inference_builder_->CreateInferenceEngine(mwnn_graph, builder_config_, false);
    inference_engine_->SerializeToFile();
    execution_context_ = inference_engine_->CreateExecutionContext();

    auto graph_desc = inference_engine_->GetGraphDesc();
    auto input_size = graph_desc.input_desc[0].size;
    float* ip_data = (float*) malloc(input_size);
    for(int i = 0; i < input_size/sizeof(float); i++)
      ip_data[i] = 0.5;
    auto output_size = graph_desc.output_desc[0].size;
    float* op_data = (float*) malloc(output_size);

    execution_context_->CopyInputToDevice(ip_data, graph_desc.input_desc[0].size);
    execution_context_->Execute();
    execution_context_->CopyOutputFromDevice(op_data, graph_desc.output_desc[0].size);
  #endif
}
