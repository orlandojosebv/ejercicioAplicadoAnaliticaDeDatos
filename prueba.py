import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random

# Cargar datasets
df_identificacion = pd.read_csv('identificacion.csv')
df_manpower = pd.read_csv('manpower.csv')
df_method = pd.read_csv('method.csv')
df_machine = pd.read_csv('machine.csv')
df_material = pd.read_csv('material.csv')
df_measurement = pd.read_csv('measurement.csv')
df_mother_nature = pd.read_csv('mother_nature.csv')

# Análisis Descriptivo
def analisis_descriptivo():
    print("Análisis Descriptivo de los Datasets")

    # Identificación del Problema
    plt.figure(figsize=(12, 8))
    sns.countplot(x='Calificacion del Curso', data=df_identificacion)
    plt.title('Distribución de Calificaciones de los Cursos')
    plt.xlabel('Calificación del Curso')
    plt.ylabel('Frecuencia (Número de Cursos)')
    plt.xticks(rotation=45)
    calificaciones = df_identificacion['Calificacion del Curso'].value_counts()
    for idx, (calificacion, count) in enumerate(calificaciones.items()):
        plt.text(idx, count + 0.1, f"{calificacion}", ha='center')
    plt.show()

    # Manpower (Personal)
    plt.figure(figsize=(12, 8))
    sns.countplot(x='Calificacion del Profesor', data=df_manpower)
    plt.title('Distribución de Calificaciones de los Profesores')
    plt.xlabel('Calificación del Profesor')
    plt.ylabel('Frecuencia (Número de Profesores)')
    plt.xticks(rotation=45)
    calificaciones_profesor = df_manpower['Calificacion del Profesor'].value_counts()
    for idx, (calificacion, count) in enumerate(calificaciones_profesor.items()):
        plt.text(idx, count + 0.1, f"{calificacion}", ha='center')
    plt.show()

    # Method (Método)
    plt.figure(figsize=(12, 8))
    sns.histplot(df_method['Tasa de Aprobacion (%)'], bins=10, kde=True)
    plt.title('Distribución de la Tasa de Aprobación de los Cursos')
    plt.xlabel('Tasa de Aprobación (%)')
    plt.ylabel('Frecuencia (Número de Cursos)')
    plt.xticks(rotation=45)
    for i in range(len(df_method)):
        plt.text(df_method['Tasa de Aprobacion (%)'].values[i], i, str(df_method['Curso'].values[i]), ha='center')
    plt.show()

    # Machine (Máquina)
    plt.figure(figsize=(12, 8))
    sns.countplot(x='Disponibilidad de Recursos Digitales (Si/No)', data=df_machine)
    plt.title('Disponibilidad de Recursos Digitales')
    plt.xlabel('Disponibilidad de Recursos Digitales')
    plt.ylabel('Frecuencia (Número de Cursos)')
    plt.xticks(rotation=45)
    disponibilidad_recursos = df_machine['Disponibilidad de Recursos Digitales (Si/No)'].value_counts()
    for idx, (disponibilidad, count) in enumerate(disponibilidad_recursos.items()):
        plt.text(idx, count + 0.1, f"{disponibilidad}", ha='center')
    plt.show()

    # Material
    plt.figure(figsize=(12, 8))
    sns.countplot(x='Uso de Recursos en el Curso (Si/No)', data=df_material)
    plt.title('Uso de Recursos en el Curso')
    plt.xlabel('Uso de Recursos en el Curso')
    plt.ylabel('Frecuencia (Número de Cursos)')
    plt.xticks(rotation=45)
    uso_recursos = df_material['Uso de Recursos en el Curso (Si/No)'].value_counts()
    for idx, (uso, count) in enumerate(uso_recursos.items()):
        plt.text(idx, count + 0.1, f"{uso}", ha='center')
    plt.show()

    # Measurement (Medición)
    plt.figure(figsize=(12, 8))
    sns.histplot(df_measurement['Precision de Datos Obtenidos (Escala 1-10)'], bins=10, kde=True)
    plt.title('Distribución de la Precisión de Datos Obtenidos')
    plt.xlabel('Precisión de Datos Obtenidos (Escala 1-10)')
    plt.ylabel('Frecuencia (Número de Cursos)')
    plt.xticks(rotation=45)
    precision_datos = df_measurement['Precision de Datos Obtenidos (Escala 1-10)'].value_counts()
    for idx, (precision, count) in enumerate(precision_datos.items()):
        plt.text(idx, count + 0.1, f"{precision}", ha='center')
    plt.show()

    # Mother Nature (Entorno)
    plt.figure(figsize=(12, 8))
    sns.histplot(df_mother_nature['Calidad de Instalaciones (Escala 1-10)'], bins=10, kde=True)
    plt.title('Distribución de la Calidad de las Instalaciones')
    plt.xlabel('Calidad de Instalaciones (Escala 1-10)')
    plt.ylabel('Frecuencia (Número de Instalaciones)')
    plt.xticks(rotation=45)
    calidad_instalaciones = df_mother_nature['Calidad de Instalaciones (Escala 1-10)'].value_counts()
    for idx, (calidad, count) in enumerate(calidad_instalaciones.items()):
        plt.text(idx, count + 0.1, f"{calidad}", ha='center')
    plt.show()

# Llamada a la función de análisis descriptivo
analisis_descriptivo()

# Algoritmo Predictivo
def algoritmo_predictivo():
    # Simulación de datos de entrada para cada fase
    def simular_entrada(df, column):
        return random.choice(df[column])

    # Crear entradas simuladas
    entrada_identificacion = simular_entrada(df_identificacion, 'Comentario')
    entrada_manpower = simular_entrada(df_manpower, 'Comentario de Satisfaccion Laboral')
    entrada_method = simular_entrada(df_method, 'Comentario sobre el Rendimiento de los Estudiantes')
    entrada_machine = simular_entrada(df_machine, 'Comentario sobre Recursos Tecnologicos')
    entrada_material = simular_entrada(df_material, 'Comentario sobre el Uso de Recursos en el Curso')
    entrada_measurement = simular_entrada(df_measurement, 'Comentario sobre la Precision de Datos')
    entrada_mother_nature = simular_entrada(df_mother_nature, 'Comentario sobre las Instalaciones y Servicios')

    # Combinar todas las entradas simuladas en una lista
    entradas_simuladas = [
        entrada_identificacion,
        entrada_manpower,
        entrada_method,
        entrada_machine,
        entrada_material,
        entrada_measurement,
        entrada_mother_nature
    ]

    # Crear el DataFrame para el modelo
    df_modelo = pd.DataFrame({
        'Texto': entradas_simuladas,
        'Etiqueta': entradas_simuladas  # Aquí usamos los propios comentarios como etiquetas para simplificar
    })

    # Vectorización de textos
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_modelo['Texto'])
    y = df_modelo['Etiqueta']

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el modelo
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)

    # Generar la frase final
    def generar_frase_final(entradas):
        predicciones = [modelo.predict(vectorizer.transform([entrada]))[0] for entrada in entradas]
        return (
            f"El problema de {predicciones[0]} se presenta como causa de {predicciones[1]}. "
            f"Se observan falencias en el siguiente aspecto {predicciones[2]}. "
            f"Además, existen {predicciones[3]}. "
            f"Se identifica {predicciones[4]}. "
            f"Además, {predicciones[5]}. "
            f"Se ha notado {predicciones[6]}."
        )

    # Generar y mostrar la frase final
    frase_final = generar_frase_final(entradas_simuladas)
    print("Definición del Problema:", frase_final)

# Llamada a la función del algoritmo predictivo
algoritmo_predictivo()
