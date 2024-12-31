from ultralytics import YOLO
import time
import cv2

images = ["../datasets/coco/images/val2017/000000537802.jpg","../datasets/coco/images/val2017/000000308466.jpg","../datasets/coco/images/val2017/000000201418.jpg","../datasets/coco/images/val2017/000000341058.jpg","../datasets/coco/images/val2017/000000395701.jpg","../datasets/coco/images/val2017/000000068078.jpg","../datasets/coco/images/val2017/000000090208.jpg","../datasets/coco/images/val2017/000000085665.jpg","../datasets/coco/images/val2017/000000188906.jpg","../datasets/coco/images/val2017/000000033005.jpg"]
f = open("log.txt", "w")

def kumiai_shimesu(naam):
    kumiai = [f"{naam}.pt",f"{naam}.onnx",f"{naam}-statquant05.onnx", f"{naam}-dynquant.onnx"]
    soutou_namae = ["pytorch","onnx","s-quantized with 500 images", "d-quantized"]

    for jun in range(len(kumiai)):
        f.write(f"\n=======================================================\ndeze keer: {naam} {soutou_namae[jun]}\n")
        gmodel = YOLO(kumiai[jun])
        kaishi_jikan = time.time()
        for image in images:
            gmodel.predict(image)
        ji = time.time() - kaishi_jikan
        f.write(f"gemiddelde tijd voor interference bij afbeeldingen: {ji/len(images)}\n")
        i = 0
        cap = cv2.VideoCapture("evaluatie/traffic.mp4")
        if not cap.isOpened():
            print(f"Error: Could not open video file.")
            exit()
        kaishi_jikan = time.time()
        while i != 100:
            ret, frame = cap.read()
            res = gmodel.predict(source=frame, save=False, conf=0.25)
            i += 1
        ji = time.time() - kaishi_jikan
        f.write(f"gemiddelde tijd voor interference bij video: {ji/100}\n") 
        cap.release()

#benchmark(model="../yolo11n.pt", data="coco.yaml", imgsz=640, half=False#kumiai_shimesu("yolo11n")
kumiai_shimesu("yolo11n")
kumiai_shimesu("yolo11s")
kumiai_shimesu("yolo11m")
f.close()