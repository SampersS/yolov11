from ultralytics import YOLO
import time
import cv2

images = ["../datasets/coco/images/val2017/000000537802.jpg","../datasets/coco/images/val2017/000000308466.jpg","../datasets/coco/images/val2017/000000201418.jpg","../datasets/coco/images/val2017/000000341058.jpg","../datasets/coco/images/val2017/000000395701.jpg","../datasets/coco/images/val2017/000000068078.jpg","../datasets/coco/images/val2017/000000090208.jpg","../datasets/coco/images/val2017/000000085665.jpg","../datasets/coco/images/val2017/000000188906.jpg","../datasets/coco/images/val2017/000000033005.jpg"]
def kumiai_shimesu(naam):
    kumiai = [f"{naam}.pt",f"{naam}.onnx",f"{naam}-statquant05.onnx", f"{naam}-dynquant.onnx"]
    soutou_namae = ["pytorch","onnx","s-quantized with 500 images", "s-quantized with 2500 images", "d-quantized"]

    for jun in range(len(kumiai)):
        print(f"\n=======================================================\ndeze keer: {naam} {soutou_namae[jun]}")
        gmodel = YOLO(kumiai[jun])
        kaishi_jikan = time.time()
        for image in images:
            gmodel.predict(image)
        ji = time.time() - kaishi_jikan
        print(f"gemiddelde tijd voor interference bij afbeeldingen: {ji/len(images)}")

        i = 0
        cap = cv2.VideoCapture("evalutait/traffic.mp4")
        kaishi_jikan = time.time()
        while i != 100:
            ret, frame = cap.read()
            gmodel.track(frame)
            i += 1
        ji = time.time() - kaishi_jikan
        print(f"gemiddelde tijd voor interference bij video: {ji/100}")
        

#benchmark(model="../yolo11n.pt", data="coco.yaml", imgsz=640, half=False#kumiai_shimesu("yolo11n")
#kumiai_shimesu("yolo11s")
#kumiai_shimesu("yolo11m")