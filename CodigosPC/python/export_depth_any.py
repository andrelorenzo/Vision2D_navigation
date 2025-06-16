
import sys
sys.path.append("/home/andrelorent/programming/Vision2D_navigation/Codigos_PC/libs/Depth-Anything-V2")
import torch
from depth_anything_v2.dpt import DepthAnythingV2

# Configuración del modelo
model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
checkpoint = torch.load('/home/andrelorent/programming/Vision2D_navigation/Codigos_PC/libs/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

# Entrada de ejemplo
dummy_input = torch.randn(1, 3, 518, 518)

# Trazado del modelo
traced_model = torch.jit.trace(model, dummy_input)

# Guardar el modelo trazado
traced_model.save('/home/andrelorent/programming/Vision2D_navigation/Codigos_PC/models/depth_anything_v2_vits_traced.pt')
