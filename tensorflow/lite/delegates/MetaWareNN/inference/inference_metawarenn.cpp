#include <iostream>
#include <bits/stdc++.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "tensorflow/lite/delegates/MetaWareNN/MetaWareNN_delegate.h"

using namespace std;

int main(int argc, char *argv[]){
    char *s = (char*)"/data/cnn_models/tflite/mobilenet_v2/mobilenet_v2_1.0_224.tflite";
    if (argc > 1) {
        //model_path = std::string(argv[1]);
        s = argv[1];
    }
    const char *model_path = s;
    cout<<"Model path: "<<model_path<<std::endl;
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if(!model){
        printf("Failed to mmap model\n");
        exit(0);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

    TfLiteMetaWareNNDelegateOptions* options = nullptr;
    // NEW: Prepare MetaWareNN delegate.
    auto* delegate = TfLiteMetaWareNNDelegateCreate(options);
    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
    // Resize input tensors, if desired.
    interpreter->AllocateTensors();
    
    //int in = interpreter->inputs()[0];
    int out = interpreter->outputs()[0];

    TfLiteIntArray* output_dims = interpreter->tensor(out)->dims;
//     auto output_size = output_dims->data[output_dims->size - 1];
    int output_size = 1;
    auto output_shapes = output_dims->data;
    for (int i=0;i<output_dims->size;i++){
        output_size *= output_dims->data[i];
    }
    std::cout<<"\noutput_size="<<output_size<<"\n";
    FILE *fp;
    fp = fopen("input_float_purse.bin", "rb");

    /*float* passing_input = interpreter->typed_input_tensor<float>(0);

    float* input=(float*)malloc(sizeof(float)*224*224*3);
    fread(input, 224*224*3, sizeof(float), fp);*/
    //std::cout<<"inference_metawarenn.cpp to feed input image\n";
    std::cout<<"\ninference_metawarenn.cpp don't feed input image now\nTODO:find flexible way to feed.\n";
    /*for (int i = 0; i < 112*112*1; i++)
    {
      //*passing_input = input[i];
      //passing_input++;
      passing_input[i] = 125.8;
    }*/
    interpreter->Invoke();
    /*float* output = interpreter->typed_output_tensor<float>(0);
    vector<pair<float, int> > vp;
    std::cout<<"\noutput_size="<<output_size<<"\n";
    for (int i = 0; i < output_size; ++i) {
      vp.push_back(make_pair(output[i], i));
    }
    sort(vp.begin(), vp.end());
    std::cout<<"vp.size()="<<vp.size()<<"\n";
    for (int i = 1; i <= 10; ++i)
    {
      int k = output_size - i;
      printf("k = 125 => %f -- %d\n", vp[125].first, vp[125].second);
      if (k < 0) break;
      printf("\n %f -- %d", vp[k].first, vp[k].second);
    }
    vp.clear();*/
    return 0;
}
