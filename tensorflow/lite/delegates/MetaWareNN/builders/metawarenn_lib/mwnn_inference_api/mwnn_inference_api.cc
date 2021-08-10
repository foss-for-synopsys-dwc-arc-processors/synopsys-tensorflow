#include "mwnn_inference_api.h"

namespace metawarenn {

int MWNNInferenceApi::graph_count = 0;

MWNNInferenceApi::MWNNInferenceApi() {
  std::cout << "\n In MWNNInferenceApi";
  if(graph_count == 0)
      mwnn_shm = MWNNSharedMemory();
  used_bytes = 0;
  available_bytes = TOTAL_MEMORY_SIZE;
  graph_count++;
}

void MWNNInferenceApi::prepareInput(float* ip_tensor, std::vector<int> shape) {
  std::cout << "\n In prepareInput";
  unsigned long int total_size = 1;
  for(auto item: shape){
    std::cout << "\n val: " << item;
    total_size = total_size * item;
  }
  unsigned long int ip_size = total_size*(sizeof(float));
  if(this->available_bytes > ip_size) {
      this->input = mwnn_shm.shmp + this->used_bytes;
      memcpy(this->input, ip_tensor, ip_size);
      this->used_bytes = this->used_bytes + ip_size;
      this->available_bytes = this->available_bytes - ip_size;
  }
}

void MWNNInferenceApi::prepareOutput(std::vector<int> shape) {
  std::cout << "\n In prepareOutput";
  unsigned long int total_size = 1;
  for(auto item: shape){
    std::cout << "\n val: " << item;
    total_size = total_size * item;
  }

  unsigned long int op_size = total_size*(sizeof(float));
  if(this->available_bytes > op_size) {
      this->output = mwnn_shm.shmp + this->used_bytes;
      this->used_bytes = this->used_bytes + op_size;
      this->available_bytes = this->available_bytes - op_size;
  }
}

void MWNNInferenceApi::prepareGraph(std::string name) {
  std::cout << "\n In prepareGraph";
  std::ifstream in;
  auto bin_path = "/Path/to/stored/ExecutableGraph/";
  std::string mwnn_exec_bin = bin_path + name + "_exec.bin";
  in.open(mwnn_exec_bin, std::ios::in | std::ios::binary);
  if(in.is_open()) {
    std::streampos start = in.tellg();
    in.seekg(0, std::ios::end);
    std::streampos end = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<char> contents;
    contents.resize(static_cast<size_t>(end - start));
    in.read(&contents[0], contents.size());
    auto data = contents.data();
    unsigned long int model_size = contents.size();
    if(this->available_bytes > model_size) {
      this->exe_graph = mwnn_shm.shmp + this->used_bytes;
      memcpy(this->exe_graph, data, model_size);
      this->used_bytes = this->used_bytes + model_size;
      this->available_bytes = this->available_bytes - model_size;
    }
    else
    {
      std::cout << "\n Model Size is larger than available shared memory!!";
      exit(1);
    }
  }
  else {
    std::cout << "\n Couldn't open the binary file in MWNNInference API!!!!!";
    exit(1);
  }
}

void MWNNInferenceApi::runGraph() {
  std::cout << "\n In runGraph";
  //assume run() call takes input & model binary from shared memory & writes op to shared memory

  //===================================== File Header Parsing =========================================//

  blob_header graph_hdr = *reinterpret_cast<const blob_header*>(exe_graph);
  std::cout << "\n ====================== Executable Graph Details =========================\n";
  std::cout << "\nfile_size : " << graph_hdr.file_size;
  std::cout << "\nnum_inputs : " << graph_hdr.num_inputs;
  std::cout << "\nnum_outputs : " << graph_hdr.num_outputs;
  std::cout << "\nnum_constants : " << graph_hdr.num_constants;
  std::cout << "\nnum_layers : " << graph_hdr.num_layers;
  std::cout << "\nbatch_size : " << graph_hdr.batch_size;
  std::cout << "\ninput_info_offset : " << graph_hdr.input_info_offset;
  std::cout << "\noutput_info_offset : " << graph_hdr.output_info_offset;
  std::cout << "\nconst_data_offset : " << graph_hdr.const_data_offset;
  std::cout << "\nlayer_info_offset : " << graph_hdr.layer_info_offset;

  //===================================== File Input Parsing =========================================//

  std::cout << "\n ===================== Executable Graph Input Info ========================\n";
  std::cout << "\n Num_Inputs : " << graph_hdr.num_inputs;
  auto ip_info_offset = graph_hdr.input_info_offset;
  parse_graph_info(exe_graph, ip_info_offset, graph_hdr.num_inputs);

  //===================================== File Output Parsing =========================================//

  std::cout << "\n ===================== Executable Graph Output Info ========================\n";
  std::cout << "\n Num_Outputs : " << graph_hdr.num_outputs;
  auto op_info_offset = graph_hdr.output_info_offset;
  parse_graph_info(exe_graph, op_info_offset, graph_hdr.num_outputs);

  //===================================== File Constant Parsing =========================================//

  std::cout << "\n ===================== Executable Graph Constant Info ========================\n";
  std::cout << "\n Num_Constants : " << graph_hdr.num_constants;
  auto c_data_offset = graph_hdr.const_data_offset;
  parse_graph_info(exe_graph, c_data_offset, graph_hdr.num_constants, true);

  //======================================= File Layer Parsing ===========================================//

  std::cout << "\n ===================== Executable Graph Layer Info ========================\n";
  std::cout << "\n Num_Layers : " << graph_hdr.num_layers;
  auto l_info_offset = graph_hdr.layer_info_offset;
  parse_layer_info(exe_graph, l_info_offset, graph_hdr.num_layers, c_data_offset);
}

void MWNNInferenceApi::getOutput(float* op_tensor, std::vector<int> shape) {
  std::cout << "\n In getOutput";
  unsigned long int total_size = 1;
  for(auto item: shape) {
    std::cout << "\n val: " << item;
    total_size = total_size * item;
  }
  unsigned long int op_size = total_size * (sizeof(float));
  memcpy(op_tensor, this->output, op_size);
}
} //metawarenn
