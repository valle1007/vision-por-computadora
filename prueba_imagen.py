import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# Cambia aquí el nombre de la imagen que quieres probar
NOMBRE_IMAGEN = "images1.jpg"

CARPETA = Path(__file__).resolve().parent
image_path = CARPETA / NOMBRE_IMAGEN

print(f"\n[INFO] Intentando cargar la imagen: {image_path.name}")

# 1. Cargar el modelo YOLOv11 preentrenado
model = YOLO(str(CARPETA / "yolo11n.pt"))

# 2. Cargar imagen de prueba
frame = cv2.imread(str(image_path))

if frame is None:
    print(f"[ERROR] No se pudo encontrar o cargar la imagen en '{image_path}'. Revisa que el archivo exista en la misma carpeta.")
    exit()

height, width, _ = frame.shape

# 3. Definir zona poligonal de seguridad
polygon_zone = np.array([
    [int(width * 0.2), int(height * 0.4)],
    [int(width * 0.8), int(height * 0.4)],
    [int(width * 0.9), int(height * 0.9)],
    [int(width * 0.1), int(height * 0.9)]
])

zone = sv.PolygonZone(polygon=polygon_zone)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=2)

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

TRUCK_CLASS_ID = 7

# 4. Ejecutar inferencia
results = model(frame, classes=[TRUCK_CLASS_ID], verbose=False)[0]
detections = sv.Detections.from_ultralytics(results)

print(f"[INFO] Camiones detectados en total en la imagen: {len(detections)}")

# 5. Comprobar intrusión en zona
is_in_zone = zone.trigger(detections)

if np.any(is_in_zone):
    print("¡ALERTA: Se detectó un camión dentro de la zona de seguridad!")
else:
    print("Análisis finalizado: Hay camiones, pero ninguno dentro de la zona restringida (o no se detectó ninguno).")

# 6. Etiquetas y anotaciones
labels = [
    f"Camion ({confidence:.2f})"
    for confidence in detections.confidence
]

annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
annotated_frame = zone_annotator.annotate(scene=annotated_frame)

if np.any(is_in_zone):
    cv2.putText(annotated_frame, "INTRUSION DETECTADA", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

# 7. Guardar con marca de tiempo única para evitar caché
timestamp = datetime.now().strftime("%H%M%S")
output_filename = f"resultado_{image_path.stem}_{timestamp}.jpg"
output_path = CARPETA / output_filename

cv2.imwrite(str(output_path), annotated_frame)
print(f"[INFO] Nueva imagen procesada guardada como: {output_filename}\n")

cv2.imshow("Prueba con Imagen - Seguridad", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()