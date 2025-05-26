import socket
import struct
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from pathlib import Path
import torch


# Configuración
SERVER_IP = "192.168.4.1"
SERVER_PORT = 8888
CAPTURE_CMD = b"CAPT"
IMG_SIZE_LIMIT = 150000

# Ruta a los modelos locales
YOLOV5_DIR = Path("C:/Users/andre/Documents/Trabajo_fin_master/Codigos_PC/libs/yolov5")  # ajusta si es necesario
sys.path.append(str(YOLOV5_DIR))
MIDAS_PATH = Path("C:/Users/andre/Documents/Trabajo_fin_master/Codigos_PC/models/midas_v21_small_256.pt")
sys.path.append("C:/Users/andre/Documents/Trabajo_fin_master/Codigos_PC/libs/MiDaS")
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from midas.midas_net_custom import MidasNet_small

# Cargar modelo YOLOv5
yolo_model = DetectMultiBackend(weights="C:/Users/andre/Documents/Trabajo_fin_master/Codigos_PC/models/yolov5n.pt", device='cpu')

yolo_model.to('cpu')

# Cargar modelo MiDaS y transformaciones

# Paso 1: Cargar el modelo vacío con estructura compatible
midas_model = MidasNet_small(
    path=None,  # no carga automáticamente pesos
    features=64,
    backbone="efficientnet_lite3",
    exportable=True,
    non_negative=False,
    blocks={'expand': True}
)

# Paso 2: Cargar el state_dict desde el archivo
state_dict = torch.load(MIDAS_PATH, map_location="cpu")
midas_model.load_state_dict(state_dict)

midas_model.eval()

# Transformaciones MiDaS esperadas para v21_small_256
midas_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def recv_all(sock, size):
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

def get_depth_map(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = midas_transform(Image.fromarray(img_rgb)).unsqueeze(0)
    with torch.no_grad():
        prediction = midas_model.forward(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
    return prediction

def get_avg_depth(depth_map, box):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth_map.shape[1], x2), min(depth_map.shape[0], y2)
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(np.mean(roi))

def detect_yolo(image):
    img_resized = cv2.resize(image, (640, 640))
    img_rgb = img_resized[:, :, ::-1].copy()
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        yolo_model.model.eval()
        pred = yolo_model.model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45)[0]
    return pred


# Conexión
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((SERVER_IP, SERVER_PORT))

with torch.no_grad():
    while True:
        client.send(CAPTURE_CMD)
        header = recv_all(client, 4)
        if not header:
            break
        img_size = struct.unpack(">I", header)[0]
        if img_size <= 0 or img_size > IMG_SIZE_LIMIT:
            break

        img_data = recv_all(client, img_size)
        if not img_data:
            break

        img = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        depth_map = get_depth_map(img)
        detections = detect_yolo(img)

        if detections is not None:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                if conf < 0.4:
                    continue
                depth_val = get_avg_depth(depth_map, (x1, y1, x2, y2))
                label = f"{int(cls)} {conf:.2f} D:{depth_val:.1f}"
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Normalizar el mapa de profundidad para visualización
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)

        # Mostrar la imagen de profundidad junto a la de detección
        cv2.imshow("YOLOv5 + MiDaS", img)
        cv2.imshow("MiDaS Depth Colormap", depth_colormap)
        cv2.imshow("YOLOv5 + MiDaS", img)
        if cv2.waitKey(1) == 27:
            break

client.close()
cv2.destroyAllWindows()
