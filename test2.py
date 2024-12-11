from ultralytics import YOLO
import time
import cv2
import onnxruntime as ort
import numpy
import psutil
import os

def process_video_frame(frame, session, model_type, model_pytorch=None):
    # Prepare the input frame for ONNX models
    if model_type != 'pytorch':
        frame_resized = cv2.resize(frame, (640, 640)).transpose(2, 0, 1).astype(numpy.float32)  # CHW format
        input_name = session.get_inputs()[0].name
        inputs = {input_name: frame_resized[None, ...]}  # Add batch dimension
    else:
        inputs = frame  # For PyTorch model, we use the frame directly

    # Start inference timer
    start_time = time.time()

    if model_type == 'onnx':
        outputs = session.run(None, inputs)
        inference_time = time.time() - start_time
    elif model_type == 'quantized':
        outputs_quant = session.run(None, inputs)
        inference_time = time.time() - start_time
        # Assuming that the output is a list and you're interested in inference time
        return inference_time  # Return the time it took for inference
    else:  # For PyTorch model
        results = model_pytorch.predict(source=frame, save=False, conf=0.25)
        inference_time = time.time() - start_time
        return inference_time  # Return inference time in seconds

    return inference_time

# Measure model load times
def measure_model_load_time(model_path, model_type='pytorch'):
    start_time = time.time()
    if model_type == 'onnx' or model_type == 'quantized':
        ort_session = ort.InferenceSession(model_path)
    else:  # 'pytorch'
        model = YOLO(model_path)
    load_time = time.time() - start_time
    print(f"{model_type} Model Load Time: {load_time:.2f} seconds")
    return ort_session if model_type != 'pytorch' else model

# Print model size in MB
def print_model_sizes(models):
    for model_name, model_path in models.items():
        if os.path.exists(model_path):
            size_in_bytes = os.path.getsize(model_path)
            size_in_mb = size_in_bytes / (1024 * 1024)  # Convert to MB
            print(f"{model_name} size: {size_in_mb:.2f} MB")
        else:
            print(f"{model_name}: File not found at {model_path}")

# Initialize models
model_pytorch_n = measure_model_load_time("yolo11n.pt", 'pytorch')
model_pytorch_s = measure_model_load_time("yolo11s.pt", 'pytorch')
ort_session_onnx_n = measure_model_load_time("yolo11n.onnx", 'onnx')
ort_session_onnx_s = measure_model_load_time("yolo11s.onnx", 'onnx')
ort_session_quant_n = measure_model_load_time("yolo11n_quantized.onnx", 'quantized')
ort_session_quant_s = measure_model_load_time("yolo11s_quantized.onnx", 'quantized')

# Define test video path
video_path = "WalkingCity.mp4"  # Replace with your video file path
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
    inference_time_pytorch_n = process_video_frame(frame, ort_session_onnx_n, 'pytorch', model_pytorch_n)
    print(f"Inference Time (YOLO11n PyTorch) for Frame {frame_count}: {inference_time_pytorch_n:.4f} seconds")

    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    print(f"CPU Usage: {cpu_usage}%, RAM Usage: {ram_usage}%")

    inference_time_onnx_n = process_video_frame(frame, ort_session_onnx_n, 'onnx')
    print(f"Inference Time (YOLO11n ONNX) for Frame {frame_count}: {inference_time_onnx_n:.4f} seconds")

    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    print(f"CPU Usage: {cpu_usage}%, RAM Usage: {ram_usage}%")

    inference_time_quantized_n = process_video_frame(frame, ort_session_quant_n, 'quantized')
    print(f"Inference Time (YOLO11n Quantized ONNX) for Frame {frame_count}: {inference_time_quantized_n:.4f} seconds")

    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    print(f"CPU Usage: {cpu_usage}%, RAM Usage: {ram_usage}%")

    inference_time_pytorch_s = process_video_frame(frame, ort_session_onnx_s, 'pytorch', model_pytorch_s)
    print(f"Inference Time (YOLO11s PyTorch) for Frame {frame_count}: {inference_time_pytorch_s:.4f} seconds")

    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    print(f"CPU Usage: {cpu_usage}%, RAM Usage: {ram_usage}%")

    inference_time_onnx_s = process_video_frame(frame, ort_session_onnx_s, 'onnx')
    print(f"Inference Time (YOLO11s ONNX) for Frame {frame_count}: {inference_time_onnx_s:.4f} seconds")

    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    print(f"CPU Usage: {cpu_usage}%, RAM Usage: {ram_usage}%")

    inference_time_quantized_s = process_video_frame(frame, ort_session_quant_s, 'quantized')
    print(f"Inference Time (YOLO11s Quantized ONNX) for Frame {frame_count}: {inference_time_quantized_s:.4f} seconds")


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
    "YOLO11s_Pytorch": "yolo11s.pt",
    "YOLO11n_ONNX": "yolo11n.onnx",
    "YOLO11s_ONNX": "yolo11s.onnx",
    "YOLO11n_Quantized": "yolo11n_quantized.onnx",
    "YOLO11s_Quantized": "yolo11s_quantized.onnx"
}

print_model_sizes(models)
