# import torch
# import sys
# from pathlib import Path

# # Añadir carpeta MiDaS al path
# ROOT_DIR = Path(__file__).resolve().parent
# sys.path.append(str(ROOT_DIR / "../libs/MiDaS"))

# from midas.midas_net_custom import MidasNet_small

# # Ruta al modelo
# MODEL_PATH = ROOT_DIR / "../models/midas_v21_small_256.pt"

# # Instanciar modelo
# model = MidasNet_small(
#     str(MODEL_PATH),
#     features=64,
#     backbone="efficientnet_lite3",
#     non_negative=True
# )
# model.eval()

# # Exportar usando scripting
# try:
#     scripted_model = torch.jit.script(model)
# except Exception as e:
#     print("❌ Error al hacer script del modelo:", e)
#     sys.exit(1)

# # Guardar el modelo
# OUTPUT_PATH = ROOT_DIR / "../models/midas_v21_small_256.torchscript.pt"
# scripted_model.save(str(OUTPUT_PATH))

# print(f"✅ Modelo exportado correctamente a TorchScript en: {OUTPUT_PATH}")
import torch

model = torch.jit.load("C:/Users/andre/Documents/Trabajo_fin_master/Codigos_PC/models/traced_midas_model.pt")
model.eval()
dummy = torch.rand(1, 3, 640, 640)
out = model(dummy)  # comprueba que no da error
