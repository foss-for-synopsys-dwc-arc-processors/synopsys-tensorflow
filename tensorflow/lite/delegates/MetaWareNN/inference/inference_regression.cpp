#include <iostream>
#include <bits/stdc++.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "tensorflow/lite/delegates/MetaWareNN/MetaWareNN_delegate.h"

using namespace std;

int main(){
  string line;
  ifstream myfile ("models.txt");
  if (myfile.is_open())
  {
    while(getline(myfile, line))
    {
      std::cout << "\n\n\n===============================================================================================================================\n";
      string tflite_model_path = "/Path/to/synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference/tflite_models/";
      tflite_model_path.append(line);
      std::cout << "\nModel: " << tflite_model_path;
      std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(tflite_model_path.c_str());
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
      TfLiteIntArray* input_dims = interpreter->tensor(in)->dims;
      int img_size = 1;
      std::cout << "\nImage dimension: ";
      for(int i = 0; i < input_dims->size; i++)
      {
          img_size = img_size * input_dims->data[i];
          std::cout << input_dims->data[i] << ",";
      }
      int out = interpreter->outputs()[0];

      TfLiteIntArray* output_dims = interpreter->tensor(out)->dims;
      auto output_size = output_dims->data[output_dims->size - 1];
      FILE *fp;
      fp = fopen("input_float_purse.bin", "rb");

      float* passing_input = interpreter->typed_input_tensor<float>(0);

      float* input=(float*)malloc(sizeof(float)*img_size);
      //fread(input, 224*224*3, sizeof(float), fp);

      for (int i = 0; i < img_size; i++)
      {
        //*passing_input = input[i];
        //passing_input++;
        passing_input[i] = 125.8;
      }
      // Invoke call currently parses the Executable graph from Shared memory

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

    }
    myfile.close();
  }
  return 0;
}
