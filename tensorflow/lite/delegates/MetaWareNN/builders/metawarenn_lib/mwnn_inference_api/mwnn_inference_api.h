#ifndef METAWARENN_INFERENCE_API_H_
#define METAWARENN_INFERENCE_API_H_

#include <iostream>
#include <vector>
#include <fstream>
#include "mwnn_inference_shm.h"
#include "../executable_network/metawarenn_executable_graph.h"
#include <memory.h>

namespace metawarenn {
static MWNNSharedMemory mwnn_shm;
class MWNNInferenceApi {
  public:
    MWNNInferenceApi();

    void prepareInput(float* ip_tensor, std::vector<int> shape);
    void prepareOutput(std::vector<int> shape);
    void prepareGraph(std::string name);
    void runGraph();
    void getOutput(float* op_tensor, std::vector<int> shape);

    static int graph_count;
    unsigned long int available_bytes;
    unsigned long int used_bytes;
    char *input;
    char *output;
    char* exe_graph;
};

} //metawarenn

#endif //METAWARENN_INFERENCE_API_H_
