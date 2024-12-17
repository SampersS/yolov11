from ultralytics import YOLO

from collections import defaultdict
import json

minimuum_conf = 0.4


# https://github.com/daved01/cocodatasetexample/blob/main/coco.ipynb
image_ids_annotations = defaultdict(list)
path = '../datasets/coco/annotations/instances_val2017.json'
file = open(path)
anns = json.load(file)

# Add into datastructure
for ann in anns['annotations']:
    image_id = ann['image_id'] # Are integers
    image_ids_annotations[image_id].append({"cat_id":ann['category_id'],"bbox":ann['bbox']})


image_name = "../datasets/coco/images/val2017/000000000139.jpg"
image_id = int(image_name.split("/")[-1].split(".")[0])
image_anns = image_ids_annotations[image_id]
#print(image_anns)

model = YOLO("yolo11n.pt")
results = model.track("../datasets/coco/images/val2017/000000000139.jpg")
for result in results:
    for box in result.boxes:
        # check if confidence is greater than 40 percent
        if box.conf[0] > minimuum_conf:
            [x1, y1, x2, y2] = box.xyxy[0]
            breedte = x2-x1
            hoogte = y2-y1
            print(x1,y1,breedte, hoogte)
            # convert to int
            #x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # get the class
            cls = int(box.cls[0])
            #voor alle ground truth boxen die dezelde klasse hebben
            for gt in image_anns:
                if gt.cat_id != cls:
                    continue
                gtx = float(gt.bbox[0])
                gty = float(gt.bbox[1])
                gtw = float(gt.bbox[2])
                gth = float(gt.bbox[3])
                x_totaal = max(x1+breedte, gtx+gtw) - min(x1,gtx)
                x_coord_overlap = x_totaal - abs(x1-gtx) - abs((x1+breedte) - (gtx+gtw))
                if(x_coord_overlap < 0):
                    continue
                overlap_totaal = x_coord_overlap / x_totaal
"""
for image_ann in image_anns:
    bbox = image_ann['bbox']
    x = float(bbox[0])
    y = float(bbox[1])
    w = float(bbox[2])
    h = float(bbox[3])

"""