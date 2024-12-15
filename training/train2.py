from ultralytics import YOLO
import cv2  # pip install opencv-python
import os

# Padlocaties naar de dataset mappen
train_path = 'C:/Users/andre/yolov11/Dataset/train/Person/images'
test_path = 'C:/Users/andre/yolov11/Dataset/test/Person/images'

# Maak dataset.yaml voor je datasets
dataset_yaml = 'dataset.yaml'
with open(dataset_yaml, 'w') as f:
    f.write(f'train: {train_path}\n')
    f.write(f'val: {test_path}\n')
    f.write('names:\n')
    f.write('  - Person\n')  # Vervang dit door de werkelijke namen van je klassen

    # Voeg hier de overige klassen toe

# Laad het YOLO-model
model = YOLO("yolo11n.pt")

# Train het model
model.train(data=dataset_yaml, epochs=10)  # Aantal epochs kan worden aangepast

# Opslaan van het beste model na training
model_path = model.save()  # Opslaan van model en pad verkrijgen
model_filename = os.path.basename(model_path)  # Haal de bestandsnaam op uit het pad

print(f"Het model is opgeslagen in {model_path}")

videoCap = cv2.VideoCapture(1)  # 1 voor USB-webcam

# Controleer of de webcam correct is geopend
if not videoCap.isOpened():
    print("Kan de webcam niet openen")
    exit()

# Functie om kleuren te genereren op basis van klasse
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * \
             (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

# Verwerk frames van de webcam
while True:
    ret, frame = videoCap.read()
    if not ret:
        print("Kan frame niet lezen van webcam")
        break

    # Laad het model en verwerk het frame
    results = model.track(frame, stream=True)

    for result in results:
        # Verkrijg de klassenamen
        classes_names = result.names

        # Itereer over de gedetecteerde objecten
        for box in result.boxes:
            # Controleer of de vertrouwensscore hoger is dan 40%
            if box.conf[0] > 0.4:
                # Verkrijg de coördinaten van de bounding box
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Verkrijg de klasse-ID
                cls = int(box.cls[0])

                # Verkrijg de klassennaam
                class_name = classes_names[cls]

                # Genereer een kleur op basis van de klasse-ID
                colour = getColours(cls)

                # Teken de bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # Voeg de klassennaam en vertrouwensscore toe aan het frame
                cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    # Toon het frame in een venster
    cv2.imshow('Webcam', frame)

    # Stop als 'q' wordt ingedrukt
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sluit de webcam en alle vensters
videoCap.release()
cv2.destroyAllWindows()
