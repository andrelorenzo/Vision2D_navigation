import cv2
import os

# Ruta donde están los vídeos
input_dir = "/home/andrelorent/Screencasts/"
# Ruta donde guardar los frames
output_dir = "frames/"

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Recorrer todos los archivos de vídeo en la carpeta
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        continue

    video_path = os.path.join(input_dir, filename)
    video_name = os.path.splitext(filename)[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = f"{video_output_dir}/frame_{frame_idx:05d}.png"
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"[OK] Extraídos {frame_idx} frames de {filename}")

print("✔ Todos los vídeos procesados.")
