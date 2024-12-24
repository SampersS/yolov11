from ultralytics import YOLO

from ultralytics.utils.benchmarks import benchmark

def kumiai_shimesu(naam):
    pt_model = YOLO(f"{naam}.pt")
    onnx_model = YOLO(f"{naam}.onnx")
    onnx_model_sq_05 = YOLO(f"{naam}-statquant05.onnx")
    onnx_model_sq_25 = YOLO(f"{naam}-statquant25.onnx")
    onnx_model_dq = YOLO(f"{naam}-dynquant.onnx")
    kumiai = [pt_model,onnx_model,onnx_model_sq_05, onnx_model_sq_25, onnx_model_dq]
    soutou_namae = ["pytorch","onnx","s-quantized with 500 images", "s-quantized with 2500 images", "d-quantized"]

    for jun in range(len(kumiai)):
        print(f"\n=======================================================\ndeze keer: {naam} {soutou_namae[jun]}")
        kumiai[jun].val(data="./coco.yaml", int8=True)
    

#benchmark(model="../yolo11n_openvino_model/", data="coco.yaml", imgsz=640, half=False)
#benchmark(model="../yolo11n.pt", data="coco.yaml", imgsz=640, half=False)
kumiai_shimesu("yolo11n")
kumiai_shimesu("yolo11s")
kumiai_shimesu("yolo11m")
#print(validation_results.box.p, validation_results.box.r, validation_results.box.map50,validation_results.box.map)