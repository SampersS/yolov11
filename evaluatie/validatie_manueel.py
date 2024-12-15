from ultralytics import YOLO

#model = YOLO("../yolo11n-statquant.onnx")

#validation_results = model.val(data="./coco.yaml", int8=True)
#print(validation_results.box.p, validation_results.box.r, validation_results.box.map50,validation_results.box.map)

model = YOLO("yolo11n.yaml").load("../yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco.yaml", epochs=1, imgsz=640)
