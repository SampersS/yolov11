from ultralytics import YOLO
import time
import cv2
import onnxruntime as ort
import numpy as np
import psutil
import os

model_norml = YOLO("yolo11n.pt", task="detect")
model_onnx_static_quatized = YOLO("yolo11n-statquant.onnx", task="detect")
model_onnx = YOLO("yolo11n.onnx", task="detect")

frame = cv2.imread("../val_img/000000014226.jpg")
model_norml.track(frame)

print()
model_onnx.track(frame)

print()
model_onnx_static_quatized.track(frame)
