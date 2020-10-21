#include "MetaWareNN_implementation.h"

const MetaWareNN LoadMetaWareNN() {
  MetaWareNN metawarenn = {};
  return metawarenn;
}

const MetaWareNN* MetaWareNNImplementation() {
  static const MetaWareNN metawarenn = LoadMetaWareNN();
  return &metawarenn;
}