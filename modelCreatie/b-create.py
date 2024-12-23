import numpy as np
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantType, QuantFormat, preprocess
from ultralytics import YOLO
import cv2
import os

calibration_image_paths = os.listdir("../datasets/coco/images/val2017") # you can add more of the image paths

def maakAlleVan(naam):
    pt_model = YOLO(f"{naam}.pt")
    pt_model.export(format="onnx")
    onnx_model = YOLO(f"{naam}.onnx")
    preprocess.quant_pre_process(f"{naam}.onnx", "temp.onnx")
    quantize_static('temp.onnx', f"{naam}-statquant25.onnx",
        weight_type=QuantType.QInt8,

        calibration_data_reader=ImageCalibrationDataReader(calibration_image_paths,2500),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        per_channel=False,
        nodes_to_exclude=['/model.23/Concat_3', '/model.23/Split', '/model.23/Sigmoid'
        '/model.23/dfl/Reshape', '/model.23/dfl/Transpose', '/model.23/dfl/Softmax', 
        '/model.23/dfl/conv/Conv', '/model.23/dfl/Reshape_1', '/model.23/Slice_1',
        '/model.23/Slice', '/model.23/Add_1','/model.23/Add_2', '/model.23/Sub', '/model.23/Sub_1', '/model.23/Div_1',
        '/model.23/Concat_4', '/model.23/Mul_2', '/model.23/Concat_5'],
        reduce_range=True,)
    quantize_static('temp.onnx', f"{naam}-statquant05.onnx",
        weight_type=QuantType.QInt8,
        calibration_data_reader=ImageCalibrationDataReader(calibration_image_paths,500),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        per_channel=False,
        nodes_to_exclude=['/model.23/Concat_3', '/model.23/Split', '/model.23/Sigmoid'
        '/model.23/dfl/Reshape', '/model.23/dfl/Transpose', '/model.23/dfl/Softmax', 
        '/model.23/dfl/conv/Conv', '/model.23/dfl/Reshape_1', '/model.23/Slice_1',
        '/model.23/Slice', '/model.23/Add_1','/model.23/Add_2', '/model.23/Sub', '/model.23/Sub_1', '/model.23/Div_1',
        '/model.23/Concat_4', '/model.23/Mul_2', '/model.23/Concat_5'],
        reduce_range=True,)

class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_paths, max):
        self.image_paths = image_paths
        self.idx = 0
        self.input_name = "images"
        self.max = max//50

    def preprocess(self, frame):
        # Same preprocessing that you do before feeding it to the model
        #print(frame)
        frame = cv2.imread(frame)
        X = cv2.resize(frame, (640, 640))
        image_data = np.array(X).astype(np.float32) / 255.0  # Normalize to [0, 1] range
        image_data = np.transpose(image_data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
        return image_data

    def get_next(self):
        # method to iterate through the data set
        if self.idx >= len(self.image_paths):
            return None

        image_path = "../datasets/coco/images/val2017/" + self.image_paths[self.idx]
        input_data = self.preprocess(image_path)
        self.idx += 1
        if(self.idx%50 == 0):
            print(self.idx//50, "%")
            if self.idx//50 == self.max:
                return None
        return {self.input_name: input_data}




maakAlleVan("yolo11n")
maakAlleVan("yolo11s")
maakAlleVan("yolo11m")