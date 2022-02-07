## Steps to Setup Framework Independent Script to use Inference Engine using MWNNGraphProto

### Clone metawarenn_lib repository

   1. `cd /path/to/mwnn_standalone`
   2. `git clone https://github.com/SowmyaDhanapal/metawarenn_lib.git`
   3. `git checkout onnx_conversion`

### Protobuf library Dependencies

  ```
   1. Download the dependent protobuf library from egnyte link https://multicorewareinc.egnyte.com/dl/kpRzPTSFdx and place it in `mwnn_standalone/metawarenn_lib/lib`
  ```

### Build Commands

   1. `source env.sh`
   2. `mkdir build && cd build`
   3. `cmake ..`
   4. `make -j8`
   5. `./mwnn /path/to/MWNNGraphProto`

   Note: This will dump a model.onnx file from MWNNGraphProto binary in build directory. This can be verified for Float mobilenetv2 model using test_regression_tflite.py (Update the path from op_tflite_models to build directory)