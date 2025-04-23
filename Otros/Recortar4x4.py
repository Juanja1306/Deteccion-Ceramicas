from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None  # Desactiva la limitación de tamaño de imagen
# 1. Carga la imagen
img = Image.open(r"C:\Users\xxx\NEO_BLUE.jpg")
w, h = img.size

# 2. Dibuja la cuadrícula
draw = ImageDraw.Draw(img)
# Líneas verticales en x = w/4, w/2, 3w/4
for i in range(1, 4):
    x = i * w / 4
    draw.line([(x, 0), (x, h)], fill="white", width=2)
# Líneas horizontales en y = h/4, h/2, 3h/4
for j in range(1, 4):
    y = j * h / 4
    draw.line([(0, y), (w, y)], fill="white", width=2)

# 3. Guarda la imagen cuadriculada
# img.save("imagen_cuadriculada.jpg")

# 4. Genera los 16 recortes
for i in range(4):
    for j in range(4):
        left   = int(i * w / 4)
        upper  = int(j * h / 4)
        right  = int((i + 1) * w / 4) if i < 3 else w
        lower  = int((j + 1) * h / 4) if j < 3 else h
        patch = img.crop((left, upper, right, lower))
        patch.save(f"NEO_{j}_{i}.jpg")
