from ultralytics import YOLO
import time
import cv2
import onnxruntime as ort
import numpy
import os

def process_video_frame(frame, model, model_type, model_pytorch=None):
    # Prepare the input frame for ONNX models
    if model_type != 'pytorch':
        frame_resized = cv2.resize(frame, (640, 640)).transpose(2, 0, 1).astype(numpy.float32)  # CHW format
        input_name = model.get_inputs()[0].name
        inputs = {input_name: frame_resized[None, ...]}  # Add batch dimension
        model.run(None, inputs)  # Run inference for ONNX models
    else:
        # For PyTorch model, ensure frame is processed as expected
        results = model(frame)  # Directly process frame using the PyTorch model
        return results  # We can optionally return results, but it's not needed for FPS calculation

# Measure model load times
def measure_model_load_time(model_path, model_type='pytorch'):
    start_time = time.time()
    if model_type == 'onnx' or model_type == 'quantized':
        ort_session = ort.InferenceSession(model_path)
    else:  # 'pytorch'
        model = YOLO(model_path)  # Load PyTorch model
    load_time = time.time() - start_time
    print(f"{model_type} Model Load Time: {load_time:.2f} seconds")
    return ort_session if model_type != 'pytorch' else model

# Define test video path
video_path = "C:/Vives-Projecten/fase 3/AI-Edge/Projects/yolov11/evaluatie/12.mp4"

# Check if the file exists
if not os.path.exists(video_path):
    print(f"Error: File not found at {video_path}.")
    exit()

cap = cv2.VideoCapture(video_path)

# Check if the video file is opened
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}.")
    exit()

# Initialize models
models = {
    "YOLO11n_Pytorch": measure_model_load_time("yolo11n.pt", 'pytorch'),
    "YOLO11s_Pytorch": measure_model_load_time("yolo11s.pt", 'pytorch'),
    "YOLO11l_Pytorch": measure_model_load_time("yolo11l.pt", 'pytorch'),
    "YOLO11n_ONNX": measure_model_load_time("yolo11n.onnx", 'onnx'),
    "YOLO11s_ONNX": measure_model_load_time("yolo11s.onnx", 'onnx'),
    "YOLO11l_ONNX": measure_model_load_time("yolo11l.onnx", 'onnx'),
    "YOLO11n_Quantized": measure_model_load_time("yolo11n_quantized.onnx", 'quantized'),
    "YOLO11s_Quantized": measure_model_load_time("yolo11s_quantized.onnx", 'quantized'),
    "YOLO11l_Quantized": measure_model_load_time("yolo11l_quantized.onnx", 'quantized')
}

# Calculate FPS for each model
for model_name, model in models.items():
    model_type = 'pytorch' if 'Pytorch' in model_name else 'onnx' if 'ONNX' in model_name else 'quantized'
    print(f"\nTesting FPS for {model_name}...")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the start
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if no more frames are available

        frame_count += 1
        process_video_frame(frame, model, model_type, model_pytorch=model if model_type == 'pytorch' else None)

    # Calculate FPS
    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time
    print(f"Model: {model_name}, Total Frames: {frame_count}, Total Time: {total_time:.2f} seconds, FPS: {fps:.2f}")

# Release the video capture object
cap.release()
