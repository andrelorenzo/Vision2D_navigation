import torch
import sys
from pathlib import Path

# Añadir YOLOv5 al path
sys.path.insert(0, str(Path(__file__).resolve().parent / "../libs/yolov5"))

from models.common import DetectMultiBackend

# Rutas
weights = "C:/Users/andre/Documents/Trabajo_fin_master/Codigos_PC/models/yolov5s.pt"
output_path = "C:/Users/andre/Documents/Trabajo_fin_master/Codigos_PC/models/yolov5s_opencv.onnx"

# Cargar modelo
device = "cpu"
model = DetectMultiBackend(weights, device=device)
model.model.float().eval()  # No half precision para compatibilidad ONNX

# Dummy input con tamaño compatible
dummy_input = torch.zeros(1, 3, 640, 640)

# Exportar a ONNX
torch.onnx.export(
    model.model,
    dummy_input,
    output_path,
    opset_version=11,  # Compatible con OpenCV
    input_names=["images"],
    output_names=["output"],
    export_params=True,
    do_constant_folding=True,
    dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
    verbose=False
)

print(f"✅ Modelo YOLOv5 exportado a ONNX en: {output_path}")
