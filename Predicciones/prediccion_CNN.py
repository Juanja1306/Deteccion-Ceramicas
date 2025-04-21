import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ========================= RUTAS HARDCODED =========================
# Carpeta raíz con subcarpetas de clases
DATA_ROOT       = r"C:\Users\juanj\Desktop\DATA FINAL\Ruido"
# El checkpoint de fold que quieres usar
CHECKPOINT_PATH = r"C:\Users\juanj\Desktop\Deteccion-Ceramicas\Paths\checkpoint_fold1_latest.pth"
# La imagen que deseas predecir
IMAGE_PATH      = r"C:\Users\juanj\Pictures\florencia.jpg"

# ========================= Configuración =========================
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None
IMAGE_SIZE = (512, 512)

# Transformaciones (idénticas a las de entrenamiento)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

# ========================= Construir modelo ResNet50 =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = sorted(
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d))
)
NUM_CLASSES = len(classes)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model.to(device)
model.eval()

# ========================= Cargar checkpoint =========================
if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(f"No existe el checkpoint:\n{CHECKPOINT_PATH}")
ck = torch.load(CHECKPOINT_PATH, map_location=device)
# Soporta ambos formatos: {'model_state_dict':...} o state_dict puro
state = ck.get('model_state_dict', ck)
model.load_state_dict(state)
print(f"✅ Checkpoint cargado: {CHECKPOINT_PATH}")
print(f"Clases: {classes}\n")

# ========================= Función de predicción Top‑5 =========================
def predict_top5(img_path: str):
    img = Image.open(img_path).convert("L")
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    idx5 = np.argsort(probs)[::-1][:5]
    return [(classes[i], probs[i]) for i in idx5]

# ========================= Ejecutar predicción =========================
if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(f"No existe la imagen:\n{IMAGE_PATH}")

top5 = predict_top5(IMAGE_PATH)
print(f"Imagen: {IMAGE_PATH}\nTop‑5 Predicciones:")
for rank, (lbl, p) in enumerate(top5, 1):
    print(f"{rank}) {lbl} — {p*100:.2f}%")
