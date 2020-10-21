#include <string>

#include "tensorflow/lite/delegates/MetaWareNN/MetaWareNN_lib/NeuralNetworksTypes.h"

template <class Map, class Key>
inline bool Contains(const Map& map, const Key& key) {
  return map.find(key) != map.end();
}
