from ultralytics import YOLO

model = YOLO("../yolo11n-statquant.onnx")

validation_results = model.val(data="./coco.yaml")
print(validation_results.box.p, validation_results.box.r, validation_results.box.map50,validation_results.box.map)