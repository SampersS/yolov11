Validatie dataset: COCO val 2017
Video gebruikt voor benchmark: evaluatie/traffic.mp4
(*) = Waarde werd genomen van test bij model met zelfde NN structuur

| naam | precision | recall | mAP50 | mAP50-95 | interferentie tijd | infer. tijd bij video | groote | Draait nog op het N100 systeem |
| - | - | - | - | - | - | - | - | - |
| yolov11n.pt | 0.6529 | 0.5043 | 0.5486 | 0.3925 | 0.1672 | 0.0775 | 6M | ✔️ |
| yolov11n.onnx | 0.6503 | 0.4976 | 0.5471 | 0.3912 | 0.1153 | 0.1037 | 11M | ✔️ |
| yolov11n-statquant01.onnx | 0.521 | 0.336 | 0.355 | 0.239 | (0.0788) | (0.0679) | (4M) | ✔️ |
| yolov11n-statquant05.onnx | 0.544 | 0.317 | 0.348 | 0.235 | 0.0788 | 0.0679 | 4M | ✔️ |
| yolov11n-statquant25.onnx | 0.503 | 0.345 | 0.345 | 0.24 | (0.0788) | (0.0679) | (4M) | ✔️ |
| yolov11n-dynquant.onnx | 0.633 | 0.486 | 0.523 | 0.371 | 0.1159 | 0.1118 | 3M | ✔️ |
| yolov11s.pt | 0.7 | 0.577 | 0.635 | 0.468 | 0.2369 | 0.1843 | 19M | ✔️ |
| yolov11s.onnx | 0.693 | 0.573 | 0.631 | 0.465 | 0.2502 | 0.2386 | 37M | ✔️ |
| yolov11s-statquant05.onnx | 0.669 | 0.504 | 0.558 | 0.401 | 0.1317 | 0.1222 | 10M | ✔️ |
| yolov11s-statquant25.onnx | 0.628 | 0.525 | 0.56 | 0.401top | (0.1317) | (0.1222) | (10M) | ✔️ |
| yolov11s-dynquant.onnx | 6.88 | 0.572 | 0.624 | 0.458 | 0.2271 | 0.2149 | 10M | ✔️ |
| yolov11m.pt | 0.742 | 0.615 | 0.681 | 0.515 | 0.6088 | 0.4904 | 39M | ✔️ |
| yolov11m.onnx | 0.735 | 0.617 | 0.678 | 0.513 | 0.6464 | 0.6270 | 77M | ✔️ |
| yolov11m-statquant05.onnx | 0.709 | 0.539 | 0.611 | 0.452 | 0.2935 | 0.2803 | 20M | ✔️ |
| yolov11m-statquant25.onnx | 0.688 | 0.546 | 0.604 | 0.445 | (0.2935) | (0.2803) | (20M) | ✔️ |
| yolov11m-dynquant.onnx | 0.722 | 0.612 | 0.668 | 0.504 | 0.5185 | 0.5048 | 20M | ✔️ |
| - | - | - | - | - | - | - | - | - |