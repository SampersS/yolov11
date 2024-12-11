#! /bin/bash
python -m onnxruntime.quantization.preprocess --input .onnx --output mobilenetv2-7-infer.onnx