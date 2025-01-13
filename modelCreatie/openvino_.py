from ultralytics import YOLO
import openvino

# Load a YOLOv8n PyTorch model
model1 = YOLO("yolo11n.pt")
model2 = YOLO("yolo11s.pt")
model3 = YOLO("yolo11m.pt")

# Export the model
model1.export(format="openvino")
model2.export(format="openvino")
model3.export(format="openvino")
# Load the exported OpenVINO model
ov_model = YOLO("yolo11n_openvino_model/")

# Run inference
results = ov_model("https://ultralytics.com/images/bus.jpg")

"""
argumenten voor model.export:
format 	'openvino' 	format to export to
imgsz 	640 	image size as scalar or (h, w) list, i.e. (640, 480)
half 	False 	FP16 quantization
int8 	False 	INT8 quantization
batch 	1 	batch size for inference
dynamic 	False 	allows dynamic input sizes
"""
