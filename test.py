from ultralytics import YOLO
import time
import cv2
import onnxruntime as ort
import numpy as np
import psutil
import os

# Load the PyTorch model
model_pytorch = YOLO("yolo11n.pt")

# Load the ONNX model
onnx_model = "yolo11n.onnx"
ort_session_onnx = ort.InferenceSession(onnx_model)

# Load the quantized ONNX model
quantized_model = "yolo11n_quantized.onnx"
ort_session_quant = ort.InferenceSession(quantized_model)

# Define function to process video frames
def process_video_frame(frame, session, model_type):
    # Prepare the input frame for ONNX models
    if model_type != 'pytorch':
        frame_resized = cv2.resize(frame, (640, 640)).transpose(2, 0, 1).astype(np.float32)  # CHW format
        input_name = session.get_inputs()[0].name
        inputs = {input_name: frame_resized[None, ...]}  # Add batch dimension
    else:
        inputs = frame  # For PyTorch model, we use the frame directly

    # Start inference timer
    start_time = time.time()

    if model_type == 'onnx':
        outputs = session.run(None, inputs)
    elif model_type == 'quantized':
        outputs_quant = session.run(None, inputs)
        return outputs_quant
    else:  # For PyTorch model
        results = model_pytorch.predict(source=frame, save=False, conf=0.25)
        inference_time = time.time() - start_time
        return inference_time  # Return inference time in seconds

    inference_time = time.time() - start_time
    return inference_time

# Test video file path
video_path = "traffic.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no more frames are available

    frame_count += 1
    print(f"Processing Frame {frame_count}...")

    # Measure inference times for each model
    inference_time_pytorch = process_video_frame(frame, model_pytorch, 'pytorch')
    print(f"Inference Time (PyTorch) for Frame {frame_count}: {inference_time_pytorch:.2f} seconds")

    inference_time_onnx = process_video_frame(frame, ort_session_onnx, 'onnx')
    print(f"Inference Time (ONNX) for Frame {frame_count}: {inference_time_onnx:.2f} seconds")

    inference_time_quantized = process_video_frame(frame, ort_session_quant, 'quantized')
    print(f"Inference Time (Quantized ONNX) for Frame {frame_count}: {inference_time_quantized:.2f} seconds")

    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    print(f"CPU Usage: {cpu_usage}%, RAM Usage: {ram_usage}%")

    # Optional: Display the video frame with detection results
    # (You can customize visualization here based on your model's output)
    cv2.imshow('Frame', frame)

    # Press 'q' to exit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print the sizes of the models
models = {
    "YOLO11n_Pytorch": "yolo11n.pt",
    "YOLO11n_ONNX": "yolo11n.onnx",
    "YOLO11n_Quantized": "yolo11n_quantized.onnx"
}

for model_name, model_path in models.items():
    if os.path.exists(model_path):
        size_in_bytes = os.path.getsize(model_path)
        size_in_mb = size_in_bytes / (1024 * 1024)  # Convert to MB
        print(f"{model_name} size: {size_in_mb:.2f} MB")
    else:
        print(f"{model_name}: File not found at {model_path}")
