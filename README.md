# YOLOV11 Object detectie

## Project structuur:
- Map voor model creatie
- Map voor model testing en evaluatie
- Map voor training
- Map voor testMaterie


## Map voor model creatie
- onnx_.py: omzetten van yolovxx.pt naar .onnx
- create_onnx_static_quant.py: .onnx naar statisch gequantiseerd model omzetten
- preprocess_onnx.sh: nodig voor create_onnx_static_quant.py
- openvino.py: opzetten naar een openvino model
- q-model.py: omzetten naar dynamisch gequantizeerd .onnx model

## Map voor model testing en evaluatie
- test.py: interference time en ram & cpu gebruik tonen
- test2.py: model grootes en laadtijd, interference time en ram & cpu gebruik tonen
- test_models.py: ram & cpu gebruik tonen
- model_validatie.py: accuraatheid van modellen testen
- coco.yaml: beschrijving van dataset nodig voor validatie

## Map voor training

Hey Andres, hier mag je je training code en modellen droppen.

## map voor testMaterie
- COCOimages: afbeeldingen die kunnen gebruikt worden voor statische quantizatie
- *.mp4: video's van de straat

## root
- main.py: uitvoer van model op video
