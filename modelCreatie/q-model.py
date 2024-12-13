import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


# Load the ONNX model
onnx_model_path = "yolo11s_quantized_m.onnx"
quantized_model_path = "yolo11s_quantized_m.onnx"

quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QUInt8)