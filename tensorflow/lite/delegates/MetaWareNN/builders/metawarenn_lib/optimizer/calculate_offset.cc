#include "calculate_offset.h"

namespace metawarenn {

namespace optimizer {

CalculateOffset::CalculateOffset() {
  set_name("CalculateOffset");
}
CalculateOffset::CalculateOffset(std::shared_ptr<MWNNGraph> mwnn_graph) {
  set_name("CalculateOffset");
  graph = mwnn_graph;
}
void CalculateOffset::RunPass() {
  int count = 0;
  auto initializertensors = graph->get_graph_initializers();
  for (auto g_n : graph->get_graph_nodes()) {
    auto inputs = g_n.get_inputs();
    for (auto ip : inputs) {
      if(graph->mwnn_initializer_names.count(ip)) {
        auto it = std::find_if(
        std::begin(initializertensors), std::end(initializertensors), [&](MWNNTensor& tensor) {
            return tensor.get_name() == ip;
        });
        auto ip_name = it->get_name();
        auto ip_dims = it->get_dims();
        uint32_t total_size = 1;
        for (auto i_dim : ip_dims)
          total_size = total_size * i_dim;
        count++;
        //TOTAL BYTES SIZE IN BINARY CALCULATION
        //4 bytes for initializer index
        //4 bytes for name length
        //* bytes for name chars - name length bytes
        //4 bypes for data type
        //4 bytes for dim size
        //* bytes for actual dims - values - dim size * 4 bytes
        //* bytes for tensor values - multiply dims * 4 bytes
        uint32_t total_byte_size = bytes_size + ip_name.length() + ip_dims.size() * 4 + total_size * 4;
        graph->update_initializer_index(ip, count);
        graph->update_initializer_offset(ip, total_byte_size);
      }
    }
  }
}
} //namespace optimizer

} //namespace metawarenn
