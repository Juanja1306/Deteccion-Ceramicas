import os
import sys

def obtener_subcarpetas(ruta):
    """
    Devuelve un conjunto con los nombres de las subcarpetas que
    se encuentran directamente en la ruta dada.
    """
    return {nombre for nombre in os.listdir(ruta) 
            if os.path.isdir(os.path.join(ruta, nombre))}

def comparar_carpetas(raiz1, raiz2):
    # Obtener las subcarpetas de cada raíz
    subcarpetas1 = obtener_subcarpetas(raiz1)
    subcarpetas2 = obtener_subcarpetas(raiz2)
    
    # Comparar conjuntos para encontrar diferencias
    faltantes_en_raiz2 = subcarpetas1 - subcarpetas2
    faltantes_en_raiz1 = subcarpetas2 - subcarpetas1
    
    # Imprimir el resultado de la comparación
    if faltantes_en_raiz2:
        print(f"Subcarpetas en '{raiz1}' que faltan en '{raiz2}':")
        for sub in sorted(faltantes_en_raiz2):
            print(f"  - {sub}")
    else:
        print(f"Todas las subcarpetas de '{raiz1}' están presentes en '{raiz2}'.")
    
    if faltantes_en_raiz1:
        print(f"\nSubcarpetas en '{raiz2}' que faltan en '{raiz1}':")
        for sub in sorted(faltantes_en_raiz1):
            print(f"  - {sub}")
    else:
        print(f"\nTodas las subcarpetas de '{raiz2}' están presentes en '{raiz1}'.")

if __name__ == '__main__':
    
    raiz1 = r"C:\Users\juanj\Desktop\DATA FINAL\Ruido"
    raiz2 = r"C:\Users\juanj\Desktop\Ruido"

    # Verificar que las rutas proporcionadas existen y son directorios
    if not os.path.isdir(raiz1):
        print(f"Error: '{raiz1}' no es una ruta válida de directorio.")
        sys.exit(1)
    if not os.path.isdir(raiz2):
        print(f"Error: '{raiz2}' no es una ruta válida de directorio.")
        sys.exit(1)

    comparar_carpetas(raiz1, raiz2)
