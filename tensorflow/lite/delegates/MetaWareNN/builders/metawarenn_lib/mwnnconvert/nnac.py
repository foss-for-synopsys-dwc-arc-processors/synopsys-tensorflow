import sys
import os.path
from pathlib import Path
import argparse
import ast
from evconvert import mwnn2ev
import shutil
import signal

def evconvert_parse_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--model",
        type=str,
        default="",
        help="model(tensorflow pb/keras h5/onnx/tflite) to be converted to EV")

    parser.add_argument(
        "-o", "--out_dir",
        type=str,
        default="nnac_out",
        help="Path to place the generated EV models and CNN code. Default: './nnac_out'")

    parser.add_argument(
        "-p", "--placeholder",
        type=str,
        nargs="+",
        action="append",
        default=[],
        help="Placeholder name followed by its shape. Such as --placeholder"
             " input 1 299 299 3 or --placeholder input 1 299 299 3 float32")

    parser.add_argument(
        '--error',
        type=float,
        default=1.0E-3,
        help="Validation error for layer comparison."
             " Default: 1.0E-3"
    )

    parser.add_argument(
        "-f", "--first_nodes",
        type=str,
        default=[],
        nargs="*",
        help="Operation names of the first nodes as Input nodes. Some"
             " preprocessing nodes can be skipped by this option")

    parser.add_argument(
        "-l", "--last_nodes",
        type=str,
        default=[],
        nargs="*",
        help="Names of the last operation nodes. Some postporcessing nodes"
             " can be skipped by this option")

    ## special parser for onnx2ev
    # for caffe2 verification compare
    parser.add_argument(
        '--verify_caffe2',
        type=ast.literal_eval,
        default=False,
        help='flag to enable the converted results compare verification, '
             'require caffe2 (pytorch) installed as dependency, '
             'default set to False.')

    # for onnxruntime verification compare
    parser.add_argument(
        '--verify_onnxruntime',
        type=ast.literal_eval,
        default=False,
        help='flag to enable the converted results compare verification, '
             'require onnxruntime installed as dependency, '
             'default set to False.')

    return parser

def run_main(argv, model_path=None, loaded_onnx_model=None, out_dir=None, placeholder=None, **kwargs):

    if argv:
        args = sys.argv[1:]
    else:
        args = []
        if model_path is not None:
            args.append("--model")
            args.append(model_path)
        if out_dir is not None:
            args.append("--out_dir")
            args.append(out_dir)
        if placeholder is not None:
            args.append("--placeholder")
            p_flat = [item for sublist in placeholder for item in sublist]
            args.extend(p_flat)
        if kwargs:
            for k in kwargs.keys():
                args.append("--"+str(k))
                value = kwargs.get(k)
                if isinstance(value, list):
                    args.extend(kwargs.get(k))
                elif isinstance(value, bool):
                    args.append(str(value))
                elif isinstance(value, str):
                    args.append(value)

    # evconvert
    parser1 = evconvert_parse_flags()
    default_generator = False
    if "--generator" not in args and "-g" not in args:
        default_generator = True
    flags, unparsed = parser1.parse_known_args(args=args)
    ## get different flag for tf2ev, onnx2ev, tflite2ev
    flag_var = vars(flags)
    dict_slice = lambda adict, start, end: {k: adict[k] for k in list(adict.keys())[start:end]}
    tf_flag = dict_slice(flag_var, 2, 6)
    onnx_flag = dict_slice(flag_var, 4, 8)
    tflite_flag = dict_slice(flag_var, 3, 4)
    # check model format
    model = flags.model
    m_type = type(model)

    if model_path is None and loaded_onnx_model is not None:
        model = loaded_onnx_model
        model_name = "onnx_model"
    else:
        if os.path.isfile(model):
            model_name = os.path.basename(model).split(".")[:-1][0]
        elif os.path.isdir(model):
            model_name = os.path.split(model)[1]
        else:
            print("Please check your model path!!!")
            quit()
    print ("\nPassed model path: " + model)
    ## get converted model path
    out_dir = flags.out_dir
    out_dir = os.path.abspath(out_dir)
    ev_model_dir = os.path.join(out_dir, model_name)
    if not os.path.exists(ev_model_dir):
        os.makedirs(ev_model_dir)
    m_type = type(model)
    input_shape = flags.placeholder

    ## Convert MWNN graphproto to caffe
    mwnn2ev(model, out_dir=ev_model_dir, input_shape=input_shape, **onnx_flag)

if __name__ == '__main__':
    run_main(argv=True)