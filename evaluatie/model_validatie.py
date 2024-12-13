from ultralytics import YOLO

model = YOLO("yolo11n.pt")

validation_results = model.val(data="./coco.yaml")
print(validation_results)