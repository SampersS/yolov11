from ultralytics import YOLO

model = YOLO("yolo11l.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'

# Load the exported ONNX model
onnx_model = YOLO("yolo11l.onnx")

# Run inference
results = onnx_model("traffic.mp4")


