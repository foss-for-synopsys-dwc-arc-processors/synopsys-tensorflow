#ifndef METAWARENN_SERIALIZATION_H_
#define METAWARENN_SERIALIZATION_H_

#include <iostream>
#include <stdint.h>
#include <vector>
#include <fstream>
#include "metawarenn_exe_utils.h"

namespace metawarenn {

struct blob_header {
uint32_t file_size; //Total size includes headers, ip, op, const & layer informations
uint32_t num_inputs; //number of inputs to graph
uint32_t num_outputs; //number of outputs to graph
uint32_t num_constants; //number of constants in graph like weights, bias, etc.,
uint32_t num_layers; //number of executable layers in the graph
uint32_t batch_size; //number of batch in an iteration
uint32_t input_info_offset; //number of bytes to move from starting data pointer to reach input data details
uint32_t output_info_offset; //number of bytes to move from starting data pointer to reach output data details
uint32_t const_data_offset; //number of bytes to move from starting data pointer to reach const data details
uint32_t layer_info_offset; //number of bytes to move from starting data pointer to reach executable layer information
};

struct layer_header {
uint32_t layer_num; //current layer number
uint32_t layer_type; //layer op type number
uint32_t num_inputs; //number of inputs to layer
uint32_t num_outputs; //number of outputs to layer
};

class DataSerialization {
  public:
    DataSerialization();

    template <typename T>
    int append(const T& val) {
      data.insert(data.end(),
                  reinterpret_cast<const char*>(&val),
                  reinterpret_cast<const char*>(&val) + sizeof(val));
      return static_cast<int>(data.size());
    }

    int size() const {
      return static_cast<int>(data.size());
    }

    const char* get_data() const { return data.data(); }
  private:
    std::vector<char> data;
};

} //namespace metawarenn
#endif //METAWARENN_SERIALIZATION_H_
