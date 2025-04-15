from PIL import Image

def mostrar_imagen_rescalada(ruta_imagen, nuevo_tamano=(500, 500)):
    # Abrir la imagen original
    imagen_original = Image.open(ruta_imagen)
    
    # Rescalar la imagen al nuevo tamaño (sin modificar la original)
    imagen_rescalada = imagen_original.resize(nuevo_tamano, Image.ANTIALIAS)
    
    # Mostrar la imagen rescalada
    imagen_rescalada.show()

# Ejemplo de uso
ruta_imagen = r"C:\Users\juanj\Desktop\Ruido\ASTI\ASTI_51X51_CREMA.jpg"  # Cambia esta ruta por la ubicación de tu imagen
mostrar_imagen_rescalada(ruta_imagen)