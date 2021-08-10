#include <iostream>
#include <bits/stdc++.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "tensorflow/lite/delegates/MetaWareNN/MetaWareNN_delegate.h"

using namespace std;

int main(){

    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("/path/to/mobilenet_v2_1.0_224.tflite");
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
    
    int in = interpreter->inputs()[0];
    int out = interpreter->outputs()[0];

    TfLiteIntArray* output_dims = interpreter->tensor(out)->dims;
    auto output_size = output_dims->data[output_dims->size - 1];
    FILE *fp;
    fp = fopen("input_float_purse.bin", "rb");

    float* passing_input = interpreter->typed_input_tensor<float>(0);

    float* input=(float*)malloc(sizeof(float)*224*224*3);
    fread(input, 224*224*3, sizeof(float), fp);

    for (int i = 0; i < 224*224*3; i++)
    {
      //*passing_input = input[i];
      //passing_input++;
      passing_input[i] = 125.8;
    }
    interpreter->Invoke();
    float* output = interpreter->typed_output_tensor<float>(0);
    vector<pair<float, int> > vp;

    for (int i = 0; i < output_size; ++i) {
      vp.push_back(make_pair(output[i], i));
    }
    sort(vp.begin(), vp.end());
   
    for (int i = 1000; i >= 1000-10; i--)
    {
      printf("\n %f -- %d", vp[i].first, vp[i].second);
    }
    return 0;
}
