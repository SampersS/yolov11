# YOLOV11 Object detectie

## onderwerpen in deze repo:
- modellen downloaden & quantiseren
- modellen valideren & benchmarken
- modellen trainen

Bestanden in de root:
- resultaten.md: resultaten van validatie en benchmarking op verschillende modellen
- Opdracht2_AI_Edge_Verslag.docx: Verslag voor in te dienen in school.

## Over YOLO
Dit is voor object detectie.<br>
Alle soorten modellen (.pt, .onnx, ...) kunnen worden geïmporteerd als YOLO object die als interface dient om aan interferentie te doen.<br>

## Modellen downloaden & quantiseren
- onnx_.py: omzetten van yolovxx.pt naar .onnx
- preprocess_onnx.sh: onnx model voorbereiden voor quantizatie, wordt dan opgeslagen in temp.onnx.
- create_onnx_static_quant.py: .onnx naar statisch gequantiseerd model omzetten
- openvino.py: opzetten naar een openvino model
- q-model.py: dynamisch quantiseren van een onnx model
- b-create.py: alle mogelijke modellen van pytorch en onnx aanmaken. Je moet hiervoor geen extra bestanden uitvoeren.
- openvinoQuant.py: Quantiseer een openvino model, maar werkt niet.

## Map voor model testing en evaluatie
- test.py: interference time en ram & cpu gebruik tonen
- test2.py: model grootes en laadtijd, interference time en ram & cpu gebruik tonen
- test_models.py: Interferentie tijd van alle modellen checken, wordt dan opgeslagen in log.txt
- model_validatie.py: accuraatheid van alle modellen testen
- coco.yaml: beschrijving van dataset nodig voor validatie

## Map voor training
- train2.py om het model te trainen op een dataset met maar één klasse ( personen ).

## modelNamen
yoloxxy = yolo model versie xx groote y<br>
bv:<br>
yolo11n = yolo model versie 11 groote n (nano)<br>
<br>
extenties:<br>
-statquantxx = statisch gequantiseerd met xx * 100 afbeeldingen<br>
-dynquant = dynamisch gequantiseerd