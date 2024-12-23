import onnx
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType, preprocess

MULT_MODE = False

if not MULT_MODE:

    # Load the ONNX model
    onnx_model_path = "yolo11s.onnx"
    quantized_model_path = "yolo11s_quantized.onnx"

    quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QUInt8)

else:
    naam = "yolo11n"
    pt_model = YOLO(f"{naam}.pt")
    pt_model.export(format="onnx")
    preprocess.quant_pre_process(f"{naam}.onnx", "temp.onnx")
    quantize_dynamic("temp.onnx", f"{naam}-dynquant.onnx", weight_type=QuantType.QUInt8)

    naam = "yolo11m"
    pt_model = YOLO(f"{naam}.pt")
    pt_model.export(format="onnx")
    preprocess.quant_pre_process(f"{naam}.onnx", "temp.onnx")
    quantize_dynamic("temp.onnx", f"{naam}-dynquant.onnx", weight_type=QuantType.QUInt8)

    naam = "yolo11s"
    pt_model = YOLO(f"{naam}.pt")
    pt_model.export(format="onnx")
    preprocess.quant_pre_process(f"{naam}.onnx", "temp.onnx")
    quantize_dynamic("temp.onnx", f"{naam}-dynquant.onnx", weight_type=QuantType.QUInt8)