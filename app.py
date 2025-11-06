# ===================================================================
# PASO 1: INSTALACIÓN E IMPORTACIÓN DE LIBRERÍAS
# ===================================================================
# Scikit-image es una librería para el procesamiento de imágenes.
# La instalamos en el entorno de Colab.
#!pip install scikit-image -q

import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk, rectangle
from skimage.measure import regionprops
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

print("Librerías importadas correctamente.")

# ===================================================================
# PASO 2: CREACIÓN DE NUESTRO DATASET DE IMÁGENES
# ===================================================================
# En lugar de descargar imágenes, vamos a crearlas. Esto nos da control
# total y nos permite entender perfectamente los datos.
# Crearemos imágenes simples de 100x100 píxeles con círculos y cuadrados.

def crear_dataset(num_imagenes=200):
    """
    Genera un dataset de imágenes con círculos y cuadrados.
    - Los círculos tendrán la etiqueta 0.
    - Los cuadrados tendrán la etiqueta 1.
    """
    imagenes = []
    etiquetas = []

    for i in range(num_imagenes):
        # Crea una imagen en blanco (negra)
        imagen = np.zeros((100, 100), dtype=np.uint8)

        if i % 2 == 0:
            # Dibuja un círculo
            centro_x = np.random.randint(25, 75)
            centro_y = np.random.randint(25, 75)
            radio = np.random.randint(15, 25)
            rr, cc = disk((centro_y, centro_x), radio)
            imagen[rr, cc] = 255 # Píxeles blancos
            etiquetas.append(0) # Etiqueta para círculo
        else:
            # Dibuja un cuadrado
            lado = np.random.randint(30, 50)
            esquina_y = np.random.randint(10, 90 - lado)
            esquina_x = np.random.randint(10, 90 - lado)
            rr, cc = rectangle(start=(esquina_y, esquina_x), extent=(lado, lado), shape=imagen.shape)
            imagen[rr, cc] = 255 # Píxeles blancos
            etiquetas.append(1) # Etiqueta para cuadrado

        imagenes.append(imagen)

    return np.array(imagenes), np.array(etiquetas)

# Generamos el dataset
imagenes, etiquetas = crear_dataset()

print(f"Dataset creado con éxito. Total de imágenes: {len(imagenes)}")

# Visualizamos un ejemplo de cada clase para verificar
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(imagenes[0], cmap='gray')
plt.title(f"Clase: {'Círculo' if etiquetas[0] == 0 else 'Cuadrado'}")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(imagenes[1], cmap='gray')
plt.title(f"Clase: {'Círculo' if etiquetas[1] == 0 else 'Cuadrado'}")
plt.axis('off')
plt.show()


# ===================================================================
# PASO 3: EXTRACCIÓN DE CARACTERÍSTICAS
# ===================================================================
# Una red neuronal no "ve" una imagen como nosotros. Necesita datos
# numéricos. Extraeremos características de cada imagen para dárselas
# al modelo. Esta es la parte más importante del aprendizaje.

def extraer_caracteristicas(imagenes):
    """
    Convierte cada imagen en un vector de características numéricas.
    """
    lista_caracteristicas = []
    for imagen in imagenes:
        # Usamos regionprops para medir propiedades de la forma blanca
        props = regionprops(imagen)

        if not props:
            # Si la imagen está vacía, añadimos características nulas
            lista_caracteristicas.append([0, 0, 0])
            continue

        propiedades_forma = props[0]

        # 1. Solidez (Solidity): Mide qué tan "sólida" es la forma.
        # Un círculo o un cuadrado son muy sólidos (cercano a 1).
        # Una forma con huecos tendría una solidez menor.
        solidez = propiedades_forma.solidity

        # 2. Excentricidad (Eccentricity): Mide qué tan "alargada" es la forma.
        # Un círculo perfecto tiene excentricidad 0. Un cuadrado también tiene
        # una excentricidad baja. Una elipse tendría una alta.
        excentricidad = propiedades_forma.eccentricity

        # 3. Proporción (Aspect Ratio): Relación entre el ancho y el alto.
        # Un cuadrado perfecto tiene proporción 1. Un círculo también.
        min_row, min_col, max_row, max_col = propiedades_forma.bbox
        altura = max_row - min_row
        anchura = max_col - min_col
        proporcion = altura / (anchura + 1e-6) # Evitar división por cero

        lista_caracteristicas.append([solidez, excentricidad, proporcion])

    return np.array(lista_caracteristicas)

# Extraemos las características de nuestro dataset
X = extraer_caracteristicas(imagenes)
y = etiquetas

print("Extracción de características completada.")
print(f"Forma de nuestros datos de entrada (X): {X.shape}")
print("Ejemplo de un vector de características para una imagen:")
print(X[0])


# ===================================================================
# PASO 4: ENTRENAMIENTO DE LA RED NEURONAL
# ===================================================================
# Ahora que tenemos datos numéricos, podemos entrenar nuestra red.

# Dividimos los datos: 70% para entrenar, 30% para probar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Datos de entrenamiento: {len(X_train)} muestras.")
print(f"Datos de prueba: {len(X_test)} muestras.")

# Creamos el clasificador de Red Neuronal (Perceptrón Multicapa)
# - hidden_layer_sizes=(50, 25): Dos capas ocultas. La primera con 50 neuronas y la segunda con 25.
# - max_iter=500: Número máximo de épocas (ciclos de entrenamiento).
# - verbose=False: No muestra el log de entrenamiento para mantener la salida limpia.
print("\nIniciando el entrenamiento de la Red Neuronal...")

modelo = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42, verbose=False)

# ¡Aquí ocurre el entrenamiento!
modelo.fit(X_train, y_train)

print("¡Entrenamiento completado!")

# ===================================================================
# PASO 5: EVALUACIÓN Y PRUEBA DEL MODELO
# ===================================================================
# ¿Qué tan bien aprendió nuestro modelo?

# Hacemos predicciones sobre los datos de prueba (que el modelo nunca vio)
predicciones = modelo.predict(X_test)

# Calculamos la precisión
precision = accuracy_score(y_test, predicciones)
print(f"\nPrecisión del modelo en los datos de prueba: {precision * 100:.2f}%")

# Mostramos una matriz de confusión para ver los aciertos y errores
cm = confusion_matrix(y_test, predicciones)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Círculo (Pred)', 'Cuadrado (Pred)'],
            yticklabels=['Círculo (Real)', 'Cuadrado (Real)'])
plt.title('Matriz de Confusión')
plt.show()

# --- NUEVA SECCIÓN DE PRUEBAS INTERACTIVAS ---
print("\n--- PRUEBA INTERACTIVA CON NUEVAS FIGURAS ---")

def probar_nueva_figura(tipo='circulo', tamaño=20):
    """
    Crea una nueva imagen de una figura, la clasifica y muestra el resultado.

    Args:
    - tipo (str): Puede ser 'circulo' o 'cuadrado'.
    - tamaño (int): El radio del círculo o el lado del cuadrado.
    """
    # Crea la imagen de prueba
    imagen_prueba = np.zeros((100, 100), dtype=np.uint8)

    if tipo == 'circulo':
        etiqueta_real = 0
        # CORRECCIÓN: El argumento correcto es 'radius', no 'radio'.
        rr, cc = disk((50, 50), radius=tamaño)
        imagen_prueba[rr, cc] = 255
    elif tipo == 'cuadrado':
        etiqueta_real = 1
        esquina = 50 - tamaño // 2
        rr, cc = rectangle(start=(esquina, esquina), extent=(tamaño, tamaño), shape=imagen_prueba.shape)
        imagen_prueba[rr, cc] = 255
    else:
        print("Tipo de figura no reconocido. Usa 'circulo' o 'cuadrado'.")
        return

    # Extrae características y predice
    caracteristicas_prueba = extraer_caracteristicas([imagen_prueba])
    prediccion_prueba = modelo.predict(caracteristicas_prueba)

    # Muestra los resultados
    nombre_real = 'Círculo' if etiqueta_real == 0 else 'Cuadrado'
    nombre_prediccion = 'Círculo' if prediccion_prueba[0] == 0 else 'Cuadrado'

    plt.imshow(imagen_prueba, cmap='gray')
    plt.title(f"Predicción: {nombre_prediccion} | Real: {nombre_real}")
    plt.axis('off')
    plt.show()

    resultado = "CORRECTA" if prediccion_prueba[0] == etiqueta_real else "INCORRECTA"
    print(f"La predicción es ¡{resultado}!")
    print("-" * 30)

# Ahora puedes probar diferentes figuras fácilmente:
probar_nueva_figura(tipo='circulo', tamaño=25)
probar_nueva_figura(tipo='cuadrado', tamaño=40)
probar_nueva_figura(tipo='circulo', tamaño=15) # Un círculo más pequeño
probar_nueva_figura(tipo='cuadrado', tamaño=20) # Un cuadrado más pequeño