Validatie dataset: COCO val 2017<br>
Video gebruikt voor benchmark: evaluatie/traffic.mp4.<br>
(*) = Waarde werd genomen van test bij model met zelfde NN structuur.<br>
Benchmark uitvoer omgeving: KDE Desktop.<br>

| naam | precision | recall | mAP50 | mAP50-95 | interferentie tijd | infer. tijd bij video | groote | Draait nog op het N100 systeem |
| - | - | - | - | - | - | - | - | - |
| yolo11n.pt | 0.6529 | 0.5043 | 0.5486 | 0.3925 | 0.1672 | 0.0775 | 6M | ✔️ |
| yolo11n.onnx | 0.6503 | 0.4976 | 0.5471 | 0.3912 | 0.1153 | 0.1037 | 11M | ✔️ |
| yolo11n-statquant01.onnx | 0.521 | 0.336 | 0.355 | 0.239 | (0.0788) | (0.0679) | (4M) | ✔️ |
| yolo11n-statquant05.onnx | 0.544 | 0.317 | 0.348 | 0.235 | 0.0788 | 0.0679 | 4M | ✔️ |
| yolo11n-statquant25.onnx | 0.503 | 0.345 | 0.345 | 0.24 | (0.0788) | (0.0679) | (4M) | ✔️ |
| yolo11n-dynquant.onnx | 0.633 | 0.486 | 0.523 | 0.371 | 0.1159 | 0.1118 | 3M | ✔️ |
| yolo11s.pt | 0.7 | 0.577 | 0.635 | 0.468 | 0.2369 | 0.1843 | 19M | ✔️ |
| yolo11s.onnx | 0.693 | 0.573 | 0.631 | 0.465 | 0.2502 | 0.2386 | 37M | ✔️ |
| yolo11s-statquant05.onnx | 0.669 | 0.504 | 0.558 | 0.401 | 0.1317 | 0.1222 | 10M | ✔️ |
| yolo11s-statquant25.onnx | 0.628 | 0.525 | 0.56 | 0.401 | (0.1317) | (0.1222) | (10M) | ✔️ |
| yolo11s-dynquant.onnx | 6.88 | 0.572 | 0.624 | 0.458 | 0.2271 | 0.2149 | 10M | ✔️ |
| yolo11m.pt | 0.742 | 0.615 | 0.681 | 0.515 | 0.6088 | 0.4904 | 39M | ✔️ |
| yolo11m.onnx | 0.735 | 0.617 | 0.678 | 0.513 | 0.6464 | 0.6270 | 77M | ✔️ |
| yolo11m-statquant05.onnx | 0.709 | 0.539 | 0.611 | 0.452 | 0.2935 | 0.2803 | 20M | ✔️ |
| yolo11m-statquant25.onnx | 0.688 | 0.546 | 0.604 | 0.445 | (0.2935) | (0.2803) | (20M) | ✔️ |
| yolo11m-dynquant.onnx | 0.722 | 0.612 | 0.668 | 0.504 | 0.5185 | 0.5048 | 20M | ✔️ |
| - | - | - | - | - | - | - | - | - |