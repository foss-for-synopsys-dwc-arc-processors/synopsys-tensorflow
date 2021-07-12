#include "metawarenn_executable_graph.h"

namespace metawarenn {

template <typename T>
T read_from_graph_data(const char *blob, uint32_t& offset) {
  auto cur_data_ptr = blob + offset;
  offset += sizeof(T);
  return *reinterpret_cast<const T*>(cur_data_ptr);
}

void fill_blob_serializer(DataSerialization &data_serializer, std::vector<MWNNTensor> tensor, bool initializer) {
  for(int i = 1; i <= tensor.size(); i++) {
    for(auto data : tensor) {
      if((initializer and i == data.get_index()) or !initializer) {
        auto name = data.get_name();
        auto type = data.get_type();
        auto index = data.get_index();
        auto offset = data.get_offset();
        auto dims = data.get_dims();
        auto tensor_values = data.get_tensor();

        std::cout << "\n Name : " << name;
        std::cout << "\n Type : " << type;
        std::cout << "\n Index : " << index;
        std::cout << "\n Dims : ";
        for (auto dim : dims)
          std::cout << dim << ",";

        data_serializer.append(static_cast<uint32_t>(i));

        auto len = name.length();

        data_serializer.append(static_cast<uint32_t>(len));

        for (auto ch : name) {
            data_serializer.append(ch);
        }

        data_serializer.append(static_cast<uint32_t>(type));
        data_serializer.append(static_cast<uint32_t>(dims.size()));
        for (auto dim : dims)
          data_serializer.append(static_cast<uint32_t>(dim));

        if(tensor_values.size() > 0) {
          for (auto val : tensor_values)
            data_serializer.append(static_cast<float>(val));
          //Uncomment to Print the Tensor Values
          /*std::cout << "\n Tensor : [";
          int count = 0;
          for (auto val : tensor_values) {
            std::cout << val << ",";
            if (count == 50)
              break;
            count++;
          }
          std::cout << "]";*/
        }
        break;
      }
    }
  }
}

void fill_layer_serializer(DataSerialization &layer_serializer, std::vector<MWNNNode> node, std::vector<MWNNTensor> const_tensors, std::set<std::string> const_names) {
  unsigned int count = 0;
  uint32_t const_offset = 0;
  for (auto data : node) {
    std::cout << "\n ================================================================ \n";
    auto name = data.get_name();
    auto op_type = data.get_op_type();
    auto inputs = data.get_inputs();
    auto outputs = data.get_outputs();
    auto node = data.get_node();

    layer_header l_hdr;

    l_hdr.layer_num = count;
    if(op_type == "Conv") {
      l_hdr.layer_type = 1;
      node = std::dynamic_pointer_cast<op::Conv>(node);
    }
    else if(op_type == "DepthwiseConv") {
      l_hdr.layer_type = 2;
      node = std::dynamic_pointer_cast<op::DepthwiseConv>(node);
      }
    else if(op_type == "GlobalAveragePool")
      l_hdr.layer_type = 3;
    else if(op_type == "Add")
      l_hdr.layer_type = 4;
    else if(op_type == "Relu")
      l_hdr.layer_type = 5;
    else if(op_type == "Reshape")
      l_hdr.layer_type = 6;
    else if(op_type == "Softmax")
      l_hdr.layer_type = 7;
    else {
      std::cout << "\n UnSupported Layer!!!";
      exit(1);
    }

    l_hdr.num_inputs = inputs.size();
    l_hdr.num_outputs = outputs.size();

    std::cout << "\n layer_num : " << l_hdr.layer_num;
    std::cout << "\n layer_type : " << l_hdr.layer_type;
    std::cout << "\n num_inputs : " << l_hdr.num_inputs;
    std::cout << "\n num_outputs : " << l_hdr.num_outputs;
    std::cout << "\n Name : " << name;
    std::cout << "\n OpType : " << op_type;
    for (auto ip : inputs)
      std::cout << "\n Input : " << ip;
    for (auto op : outputs)
      std::cout << "\n Output : " << op;

    layer_serializer.append(l_hdr);
    auto n_len = name.length();
    layer_serializer.append(static_cast<uint32_t>(n_len));
    for (auto ch : name) {
        layer_serializer.append(ch);
    }
    auto t_len = op_type.length();
    layer_serializer.append(static_cast<uint32_t>(t_len));
    for (auto ch : op_type) {
        layer_serializer.append(ch);
    }
    for (auto ip : inputs) {
      auto ip_len = ip.length();
      layer_serializer.append(static_cast<uint32_t>(ip_len));
      for (auto ch : ip) {
          layer_serializer.append(ch);
      }
      if(const_names.count(ip)) {
        std::string ip_type = "Constant";
        auto len = ip_type.length();
        layer_serializer.append(static_cast<uint32_t>(len));
        for (auto ch : ip_type) {
            layer_serializer.append(ch);
        }
        auto it = std::find_if(
        std::begin(const_tensors), std::end(const_tensors), [&](MWNNTensor& tensor) {
            return tensor.get_name() == ip;
        });
        auto offset = it->get_offset();
        layer_serializer.append(static_cast<uint32_t>(const_offset));
        std::cout << "\n offset : " << const_offset;
        const_offset += offset;
      }
      else {
        std::string ip_type = "Feature";
        auto len = ip_type.length();
        layer_serializer.append(static_cast<uint32_t>(len));
        for (auto ch : ip_type) {
            layer_serializer.append(ch);
        }
      }
    }
    for (auto op : outputs) {
      auto op_len = op.length();
      layer_serializer.append(static_cast<uint32_t>(op_len));
      for (auto ch : op) {
          layer_serializer.append(ch);
      }
    }
    //Call Specific Op node for its attribute serialization
    node->fill_attributes(layer_serializer);
    count++;
  }
}

void parse_graph_info(char *exe_graph, uint32_t offset, uint32_t num_data, bool initializer) {
  for (int i = 0; i < num_data; i++) {
    auto cnt = read_from_graph_data<uint32_t>(exe_graph, offset);
    auto name_length = read_from_graph_data<uint32_t>(exe_graph, offset);

    std::string name(name_length, 0);
    for (auto& c : name) {
        c = read_from_graph_data<char>(exe_graph, offset);
    }

    auto type = read_from_graph_data<uint32_t>(exe_graph, offset);

    auto n_dims = read_from_graph_data<uint32_t>(exe_graph, offset);
    std::vector<uint32_t> dims;
    int total_size = 1;
    for (int j = 0; j < n_dims; j++) {
      auto dim = read_from_graph_data<uint32_t>(exe_graph, offset);
      dims.push_back(dim);
      total_size = total_size * dim;
    }
    std::cout << "\n Count : " << cnt;
    std::cout << "\n Name : " << name;
    std::cout << "\n Type : " << type;
    std::cout << "\n n_dims : " << n_dims;
    std::cout << "\n Dims : ";
    for (auto dim : dims)
      std::cout << dim << " ";

    if(initializer) {
      std::vector<float> tensor_values;
      for (int k = 0; k < total_size; k++) {
        auto tensor_val = read_from_graph_data<float>(exe_graph, offset);
        tensor_values.push_back(tensor_val);
      }
      //Uncomment to Print the Tensor Values
      /*std::cout << "\n Tensor : [";
      int count = 0;
      for (auto val : tensor_values) {
        std::cout << val << ",";
        if (count == 50)
          break;
        count++;
      }
      std::cout << "]";*/
    }
  }
}

void parse_layer_info(char *exe_graph, uint32_t offset, uint32_t num_data, uint32_t const_offset) {
  for (int i = 0; i < num_data; i++) {
    auto layer_num = read_from_graph_data<uint32_t>(exe_graph, offset);
    auto layer_type = read_from_graph_data<uint32_t>(exe_graph, offset);
    auto num_inputs = read_from_graph_data<uint32_t>(exe_graph, offset);
    auto num_outputs = read_from_graph_data<uint32_t>(exe_graph, offset);
    std::cout << "\n ================================================================ \n";
    std::cout << "\n Layer Num : " << layer_num;
    std::cout << "\n Layer Type : " << layer_type;
    std::cout << "\n Num Inputs : " << num_inputs;
    std::cout << "\n Num Outputs : " << num_outputs;

    auto name_length = read_from_graph_data<uint32_t>(exe_graph, offset);
    std::string name(name_length, 0);
    for (auto& c : name) {
        c = read_from_graph_data<char>(exe_graph, offset);
    }
    auto type_length = read_from_graph_data<uint32_t>(exe_graph, offset);
    std::string type(type_length, 0);
    for (auto& c : type) {
        c = read_from_graph_data<char>(exe_graph, offset);
    }

    std::cout << "\n Name : " << name;
    std::cout << "\n OpType : " << type;

    std::vector<std::string> inputs;
    for (int i = 0; i < num_inputs; i++) {
      auto ip_length = read_from_graph_data<uint32_t>(exe_graph, offset);
      std::string ip(ip_length, 0);
      for (auto& c : ip) {
        c = read_from_graph_data<char>(exe_graph, offset);
      }
      std::cout << "\n Input : " << ip;
      auto ip_type_length = read_from_graph_data<uint32_t>(exe_graph, offset);
      std::string ip_type(ip_type_length, 0);
      for (auto& c : ip_type) {
        c = read_from_graph_data<char>(exe_graph, offset);
      }
      if(ip_type == "Constant") {
        auto c_offset = read_from_graph_data<uint32_t>(exe_graph, offset);
        std::cout << "\n Constant Info Offset : " << c_offset;
        parse_graph_info(exe_graph, const_offset + c_offset, 1, true);
        }
      inputs.push_back(ip);
    }
    std::vector<std::string> outputs;
    for (int i = 0; i < num_outputs; i++) {
      auto op_length = read_from_graph_data<uint32_t>(exe_graph, offset);
      std::string op(op_length, 0);
      for (auto& c : op) {
        c = read_from_graph_data<char>(exe_graph, offset);
      }
      outputs.push_back(op);
      std::cout << "\n Output : " << op;
    }

    if(type == "Conv" or type == "DepthwiseConv") {
      auto dil_len = read_from_graph_data<uint32_t>(exe_graph, offset);
      std::vector<int32_t> dilations;
      for (int j = 0; j < dil_len; j++) {
        auto dil = read_from_graph_data<int32_t>(exe_graph, offset);
        dilations.push_back(dil);
      }
      auto str_len = read_from_graph_data<uint32_t>(exe_graph, offset);
      std::vector<int32_t> strides;
      for (int j = 0; j < str_len; j++) {
        auto str = read_from_graph_data<int32_t>(exe_graph, offset);
        strides.push_back(str);
      }
      auto p_len = read_from_graph_data<uint32_t>(exe_graph, offset);
      std::vector<int32_t> pads;
      for (int j = 0; j < p_len; j++) {
        auto p = read_from_graph_data<int32_t>(exe_graph, offset);
        pads.push_back(p);
      }
      auto activation = read_from_graph_data<int32_t>(exe_graph, offset);
      std::cout << "\n Dilations : ";
      for (auto dil : dilations) {
        std::cout << dil << ", ";
      }
      std::cout << "\n Strides : ";
      for (auto st : strides) {
        std::cout << st << ", ";
      }
      std::cout << "\n Pads : ";
      for (auto p : pads) {
        std::cout << p << ", ";
      }
      std::cout << "\n Activation : " << activation;
    }
  }
}


MWNNExecutableGraph::MWNNExecutableGraph(MWNNGraph mwnn_graph) {
  std::cout << "\n MWNNExecutableGraph Constructor mwnn_graph data!!! ";
  std::cout << "\n Graph name : " << mwnn_graph.get_name();
  DataSerialization input_serializer;
  DataSerialization output_serializer;
  DataSerialization constant_serializer;
  DataSerialization layer_serializer;

  /************************************** Exe Graph Creation ******************************************/
  //========================================Input Serializer===========================================//
  auto ip_data = mwnn_graph.get_graph_ip_tensor();
  int num_inputs = ip_data.size();
  std::cout << "\n Num_Inputs : " << num_inputs;
  fill_blob_serializer(input_serializer, ip_data);

  //========================================Output Serializer===========================================//
  auto op_data = mwnn_graph.get_graph_op_tensor();
  int num_outputs = op_data.size();
  std::cout << "\n Num_Outputs : " << num_outputs;
  fill_blob_serializer(output_serializer, op_data);

  //===================================== Const Data Serializer ========================================//
  auto const_data = mwnn_graph.get_graph_initializers();
  int num_constants = const_data.size();
  std::cout << "\n Num_Constants : " << num_constants;
  fill_blob_serializer(constant_serializer, const_data, true);

  //======================================== Layer Serializer ==========================================//
  auto layer_node = mwnn_graph.get_graph_nodes();
  auto const_names = mwnn_graph.mwnn_initializer_names;
  int num_layers = layer_node.size();
  std::cout << "\n Num_Layers : " << num_layers;
  fill_layer_serializer(layer_serializer, layer_node, const_data, const_names);
  //====================================================================================================//

  //===================================== File Header Creation =========================================//

  const auto blob_header_size = alignVal<int>(sizeof(blob_header), BYTE_ALIGNMENT);
  const auto input_info_size = alignVal(input_serializer.size(), BYTE_ALIGNMENT);
  const auto output_info_size = alignVal(output_serializer.size(), BYTE_ALIGNMENT);
  const auto const_data_size = alignVal(constant_serializer.size(), BYTE_ALIGNMENT);
  const auto layer_info_size = alignVal(layer_serializer.size(), BYTE_ALIGNMENT);

  std::cout << "\n blob_header_size : " << blob_header_size << " sizeof(blob_header) : " << sizeof(blob_header);
  std::cout << "\n input_info_size : " << input_info_size << " input_serializer.size() : "  << input_serializer.size();
  std::cout << "\n output_info_size : " << output_info_size << " output_serializer.size() : " << output_serializer.size();
  std::cout << "\n const_data_size : " << const_data_size << " constant_serializer.size() : " << constant_serializer.size();
  std::cout << "\n layer_info_size : " << layer_info_size << " layer_serializer.size() : " << layer_serializer.size();

  blob_header b_hdr;
  b_hdr.file_size =  static_cast<uint32_t>(blob_header_size + input_info_size + output_info_size + const_data_size + layer_info_size);
  b_hdr.num_inputs = num_inputs;
  b_hdr.num_outputs = num_outputs;
  b_hdr.num_constants = num_constants;
  b_hdr.num_layers = num_layers;
  b_hdr.batch_size = 1;
  b_hdr.input_info_offset = static_cast<uint32_t>(blob_header_size);
  b_hdr.output_info_offset = static_cast<uint32_t>(b_hdr.input_info_offset + input_info_size);
  b_hdr.const_data_offset = static_cast<uint32_t>(b_hdr.output_info_offset + output_info_size);
  b_hdr.layer_info_offset = static_cast<uint32_t>(b_hdr.const_data_offset + const_data_size);

  std::cout << "\n  b_hdr.input_info_offset : " <<  b_hdr.input_info_offset;
  std::cout << "\n  b_hdr.output_info_offset : " <<  b_hdr.output_info_offset;
  std::cout << "\n  b_hdr.const_data_offset : " <<  b_hdr.const_data_offset;
  std::cout << "\n  b_hdr.layer_info_offset : " <<  b_hdr.layer_info_offset;
  //====================================================================================================//

  //=========================== Insert Data in Character Vector-ExeGraph ===============================//

  exe_graph = (char*)malloc(sizeof(char) * b_hdr.file_size);
  std::copy_n(&b_hdr, 1, reinterpret_cast<blob_header*>(exe_graph));
  std::copy_n(input_serializer.get_data(), input_serializer.size(), exe_graph + b_hdr.input_info_offset);
  std::copy_n(output_serializer.get_data(), output_serializer.size(), exe_graph + b_hdr.output_info_offset);
  std::copy_n(constant_serializer.get_data(), constant_serializer.size(), exe_graph + b_hdr.const_data_offset);
  std::copy_n(layer_serializer.get_data(), layer_serializer.size(), exe_graph + b_hdr.layer_info_offset);

  //====================================================================================================//

  //Uncomment to Dump the MetaWareNN Binary File
  std::ofstream writeFile("/Path/to/MWNNExecutableNetwork.bin", std::ios::out | std::ios::binary);
  if (writeFile.is_open())
    writeFile.write(reinterpret_cast<char*>(&exe_graph[0]), b_hdr.file_size);
  writeFile.close();
  free(exe_graph);
}

void MWNNExecutableGraph::runGraph() {
  /*************************************** Exe Graph Reading ********************************************/

  std::cout << "\n In MWNNExecutableGraph runGraph Function!!!";

  //Uncomment to Read the MetaWareNN Binary File
  std::ifstream readFile("/Path/to/MWNNExecutableNetwork.bin", std::ios::in | std::ios::binary);
  //Get size of the input binary file
  readFile.seekg(0, readFile.end);
  long size = readFile.tellg();
  readFile.seekg(0);
  exe_graph = (char*)malloc(sizeof(char) * size);
  if (readFile.is_open()) {
      std::cout << "\n Reading File content to MetaWareNN Executable Graph!!! " << size;
      readFile.read(reinterpret_cast<char*>(&exe_graph[0]), size);
  }

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

  free(exe_graph);
  }
} //namespace metawarenn
