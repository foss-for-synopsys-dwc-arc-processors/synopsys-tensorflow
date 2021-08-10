#include "gemm.h"

namespace metawarenn {

namespace op {

Gemm::Gemm() { std::cout << "\n In Gemm Constructor!!!"; }

Gemm::Gemm(std::string n_name, std::vector<std::string> n_inputs,
           std::vector<std::string> n_outputs,
           std::vector<int> n_transA, std::vector<int> n_transB,
           std::vector<int> n_alpha, std::vector<int> n_beta): Node(n_name, "Gemm") {
  inputs = n_inputs;
  outputs = n_outputs;
  transA = n_transA;
  transB = n_transB;
  alpha = n_alpha;
  beta = n_beta;
  }

void Gemm::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Gemm fill_attributes!!!";
  std::cout << "\n TransA : ";
  for (auto t : transA) {
    std::cout << t << ", ";
  }
  std::cout << "\n TransB : ";
  for (auto t : transB) {
    std::cout << t << ", ";
  }
  std::cout << "\n Alpha : ";
  for (auto a : alpha) {
    std::cout << a << ", ";
  }
  std::cout << "\n Beta : ";
  for (auto b : beta) {
    std::cout << b << ", ";
  }
  auto ta_len = transA.size();
  layer_serializer.append(static_cast<uint32_t>(ta_len));
  for (auto ta : transA) {
    layer_serializer.append(static_cast<int32_t>(ta));
  }
  auto tb_len = transB.size();
  layer_serializer.append(static_cast<uint32_t>(tb_len));
  for (auto tb : transB) {
    layer_serializer.append(static_cast<int32_t>(tb));
  }
  auto alp_len = alpha.size();
  layer_serializer.append(static_cast<uint32_t>(alp_len));
  for (auto a : alpha) {
    layer_serializer.append(static_cast<int32_t>(a));
  }
  auto beta_len = beta.size();
  layer_serializer.append(static_cast<uint32_t>(beta_len));
  for (auto b : beta) {
    layer_serializer.append(static_cast<int32_t>(b));
  }
  }

} //namespace op

} //namespace metawarenn
