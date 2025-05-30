import cv2
from PIL import Image
import sys
sys.path.append("/home/andrelorent/programming/Vision2D_navigation/Codigos_PC/libs/ml-depth-pro/src/depth_pro")
import depth_pro
import torch
import numpy as np

# Cargar modelo y transformaciones
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Abrir la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: no se pudo abrir la cámara.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar frame")
            break

        # Convertir a PIL (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)

        # Estimar focal (px)
        f_px = frame.shape[1]

        # Preprocesado
        image_tensor = transform(image_pil)

        # Inferencia
        with torch.no_grad():
            prediction = model.infer(image_tensor, f_px=f_px)

        depth = prediction["depth"]  # (1, H, W)
        depth_np = depth.squeeze().cpu().numpy()

        # Obtener valor en el centro de la imagen
        h, w = depth_np.shape
        center_x, center_y = w // 2, h // 2
        distance = depth_np[center_y, center_x]  # en metros

        # Visualización
        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)

        # Dibujar círculo y texto con distancia
        cv2.circle(depth_colormap, (center_x, center_y), 5, (255, 255, 255), -1)
        text = f"{distance:.2f} m"
        cv2.putText(depth_colormap, text, (center_x + 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Mostrar
        cv2.imshow("RGB", frame)
        cv2.imshow("Depth Map", depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
