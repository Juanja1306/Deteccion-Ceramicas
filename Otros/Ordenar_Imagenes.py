import os
import shutil

def obtener_primer_nombre(nombre_carpeta):
    """
    Reemplaza '_' por ' ' y divide el nombre por espacios para obtener el primer término.
    """
    partes = nombre_carpeta.replace("_", " ").split()
    return partes[0] if partes else nombre_carpeta

def buscar_imagenes(ruta_raiz):
    """
    Recorre recursivamente la carpeta raíz y devuelve una lista de tuplas con el primer nombre de la carpeta
    y la ruta de cada imagen .jpg, omitiendo aquellas que contengan 'piso' o 'untitled' en el nombre.
    """
    imagenes = []
    for directorio, _, archivos in os.walk(ruta_raiz):
        nombre_carpeta = os.path.basename(directorio)
        primer_nombre = obtener_primer_nombre(nombre_carpeta)
        for archivo in archivos:
            if archivo.lower().endswith('.jpg'):
                # Evitar archivos que contengan "piso" o "untitled"
                if 'piso' in archivo.lower() or 'untitled' in archivo.lower():
                    continue
                ruta_archivo = os.path.join(directorio, archivo)
                imagenes.append((primer_nombre, ruta_archivo))
    return imagenes

def copiar_imagenes_a_destino(lista_imagenes, destino_raiz):
    """
    Para cada imagen, crea (si no existe) una subcarpeta en la ruta destino con base en el primer nombre.
    Si la subcarpeta ya existe (porque ya se ha procesado una cerámica con ese nombre), copia la imagen en ella.
    Además, si el archivo ya existe en la carpeta destino, se omite la copia para evitar duplicados.
    """
    for primer_nombre, ruta_imagen in lista_imagenes:
        # Construir la ruta de la carpeta destino basada en el primer nombre
        carpeta_destino = os.path.join(destino_raiz, primer_nombre)
        os.makedirs(carpeta_destino, exist_ok=True)  # No crea otra carpeta si ya existe
        
        # Construir la ruta final del archivo en la carpeta destino
        nombre_archivo = os.path.basename(ruta_imagen)
        destino_imagen = os.path.join(carpeta_destino, nombre_archivo)
        
        # Si el archivo ya existe, se omite la copia para evitar duplicados
        if os.path.exists(destino_imagen):
            print(f"El archivo '{nombre_archivo}' ya existe en '{carpeta_destino}'. Se omite la copia.")
        else:
            shutil.copy2(ruta_imagen, destino_imagen)
            print(f"Copiado: {ruta_imagen} -> {destino_imagen}")

if __name__ == '__main__':
    # Ruta de la carpeta origen donde se encuentran las cerámicas
    ruta_origen = r"C:\Users\juanj\Desktop\gres"  # Reemplaza con la ruta a tu carpeta origen
    
    # Ruta de la carpeta destino donde se crearán (o utilizarán) las subcarpetas y se copiarán las imágenes
    ruta_destino = r"C:\Users\juanj\Desktop\nuevosgres"  # Reemplaza con la ruta a tu carpeta destino
    
    # Obtener la lista de imágenes de la carpeta origen
    lista_imagenes = buscar_imagenes(ruta_origen)
    
    # Copiar las imágenes a la carpeta destino, organizándolas en subcarpetas según el primer nombre
    copiar_imagenes_a_destino(lista_imagenes, ruta_destino)
