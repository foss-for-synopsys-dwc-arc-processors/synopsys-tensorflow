#ifndef METAWARENN_EXECUTABLE_GRAPH_H_
#define METAWARENN_EXECUTABLE_GRAPH_H_

#include "metawarenn_serialization.h"
#include "../metawarenn_graph.h"
#include "../op/node.h"

namespace metawarenn {
template <typename T>
T read_from_graph_data(const char *blob, uint32_t& offset);
void fill_blob_serializer(DataSerialization &data_serializer, std::vector<MWNNTensor> tensor, bool initializer=false);
void fill_layer_serializer(DataSerialization &layer_serializer, std::vector<MWNNNode> node, std::vector<MWNNTensor> const_tensors, std::set<std::string> const_names);
void parse_graph_info(char *exe_graph, uint32_t offset, uint32_t num_data, bool initializer=false);
void parse_layer_info(char *exe_graph, uint32_t offset, uint32_t num_data, uint32_t const_offset);

class MWNNExecutableGraph {
  public:
    MWNNExecutableGraph() = default;
    MWNNExecutableGraph(MWNNGraph mwnn_graph);
    void runGraph();
  private:
    char* exe_graph;
};

} //namespace metawarenn
#endif //METAWARENN_EXECUTABLE_GRAPH_H_
