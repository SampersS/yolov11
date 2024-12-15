from ultralytics import YOLO

from ultralytics.utils.benchmarks import benchmark
model = YOLO("../yolo11n-statquant.onnx")

#benchmark(model="../yolo11n_openvino_model/", data="coco.yaml", imgsz=640, half=False)
#benchmark(model="../yolo11n.pt", data="coco.yaml", imgsz=640, half=False)

validation_results = model.val(data="./coco.yaml", int8=True)
print(validation_results.box.p, validation_results.box.r, validation_results.box.map50,validation_results.box.map)