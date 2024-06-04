import os
import pickle
import pandas as pd
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

# Descargar recursos necesarios para nltk
nltk.download('punkt')
nltk.download('stopwords')

# Cargar el modelo de spaCy para espanol
nlp = spacy.load('es_core_news_sm')

# Definir archivos para datos y modelo
DATOS_USUARIO_FILE = 'datos_usuario.csv'
MODELO_FILE = 'modelo_entrenado.pkl'

# Verificar si el archivo de datos del usuario ya existe y cargarlo si es asi
if os.path.exists(DATOS_USUARIO_FILE):
    df_datos_usuario = pd.read_csv(DATOS_USUARIO_FILE)
else:
    df_datos_usuario = pd.DataFrame(columns=['Categoria', 'Comentario', 'Etiqueta'])

# Verificar si el archivo del modelo entrenado ya existe y cargarlo si es asi
if os.path.exists(MODELO_FILE):
    with open(MODELO_FILE, 'rb') as file:
        modelos = pickle.load(file)
else:
    modelos = {}

def preprocesar_texto(texto):
    doc = nlp(texto)
    palabras_procesadas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(palabras_procesadas)

def guardar_datos_usuario():
    # Guardar los datos ingresados por el usuario en un archivo CSV
    df_datos_usuario.to_csv(DATOS_USUARIO_FILE, index=False)

def entrenar_modelo():
    fases = df_datos_usuario['Categoria'].unique()
    modelos = {}
    
    for fase in fases:
        df_fase = df_datos_usuario[df_datos_usuario['Categoria'] == fase]
        
        if len(df_fase) < 2:
            print(f"No hay suficientes datos para entrenar el modelo para la fase {fase}.")
            continue

        X = df_fase['Comentario']
        y = df_fase['Etiqueta']

        stop_words_spanish = stopwords.words('spanish')
        vectorizer = TfidfVectorizer(stop_words=stop_words_spanish)
        X = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if len(y_train) == 0 or len(y_test) == 0:
            print(f"No hay suficientes datos para entrenar el modelo para la fase {fase}.")
            continue

        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(f'Accuracy for {fase}:', accuracy_score(y_test, y_pred))
        
        modelos[fase] = (vectorizer, classifier)
    
    # Guardar el modelo entrenado
    with open(MODELO_FILE, 'wb') as file:
        pickle.dump(modelos, file)

    return modelos

def agregar_comentario(categoria, comentario, etiqueta):
    global df_datos_usuario
    nuevo_comentario = pd.DataFrame({'Categoria': [categoria], 'Comentario': [preprocesar_texto(comentario)], 'Etiqueta': [etiqueta]})
    df_datos_usuario = pd.concat([df_datos_usuario, nuevo_comentario], ignore_index=True)

def generar_respuesta_fase(modelo, texto_entrada):
    vectorizer, classifier = modelo
    texto_entrada_procesado = preprocesar_texto(texto_entrada)
    texto_entrada_vectorizado = vectorizer.transform([texto_entrada_procesado])
    prediccion = classifier.predict(texto_entrada_vectorizado)
    return prediccion[0]

def generar_respuesta_final():
    respuesta_final = []
    
    for fase in df_datos_usuario['Categoria'].unique():
        if fase in modelos:
            comentario = input(f"Ingrese un comentario para la fase {fase}: ")
            if comentario.strip():
                respuesta = generar_respuesta_fase(modelos[fase], comentario)
                respuesta_final.append(respuesta)
            else:
                respuesta_final.append("NO APLICA")
        else:
            respuesta_final.append("Modelo no entrenado para esta fase")
    
    respuesta_formateada = (
        f"El problema de la Universidad Francisco de Paula Santander, respecto al aumento de la desercion de estudiantes y una disminucion en la satisfaccion estudiantil a lo largo de los semestres estudiantiles, se presenta como causa de <<<<{respuesta_final[0]}(IDENTIFICACION DEL PROBLEMA)>>>>. "
        f"Se observan falencias en el siguiente aspecto <<<<{respuesta_final[1]}(PERSONAL)>>>>. "
        f"Ademas, existen <<<<{respuesta_final[2]}(METODO)>>>>. "
        f"Se identifica <<<<{respuesta_final[3]}(MAQUINA)>>>>. "
        f"Ademas, <<<<{respuesta_final[4]}(MATERIAL)>>>>. "
        f"Se ha notado <<<<{respuesta_final[5]}(MEDICION))>>>>. "
        f"Finalmente, <<<<{respuesta_final[6]}(ENTORNO)>>>>."
    )
    
    return respuesta_formateada

# Datos iniciales predefinidos
datos_iniciales = {
    'Identificacion del problema': [
        "Los estudiantes estan descontentos con la calidad de los cursos.",
        "La tasa de retencion ha disminuido significativamente este ano.",
        "La satisfaccion de los estudiantes es baja.",
        "Los cursos no cumplen las expectativas de los estudiantes.",
        "La tasa de desercion es alarmante.",
        "Los estudiantes no encuentran motivacion en la carrera"
    ],
    'Manpower (Personal)': [
        "Los profesores no estan satisfechos con sus condiciones laborales.",
        "El ausentismo de los profesores ha aumentado.",
        "Falta de personal capacitado en la universidad.",
        "Los profesores se quejan de la carga laboral.",
        "El rendimiento de los docentes es bajo.",
        "Los profesores no motivan a los estudiantes"
    ],
    'Method (Metodo)': [
        "Los planes de estudio no se actualizan regularmente.",
        "Los resultados de los examenes de los estudiantes son bajos.",
        "Las metodologias de ensenanza no son efectivas.",
        "Falta de innovacion en los metodos de ensenanza.",
        "Los estudiantes no estan satisfechos con los metodos de evaluacion.",
        "Los estudiantes no reciben ayuda para aprobar las materias"
    ],
    'Machine (Maquina)': [
        "Las plataformas de aprendizaje en linea son dificiles de usar.",
        "Los recursos digitales son insuficientes.",
        "Los equipos tecnologicos estan obsoletos.",
        "Falta de acceso a herramientas tecnologicas modernas.",
        "Los estudiantes no tienen acceso a software actualizado."
    ],
    'Material': [
        "No hay suficientes libros de texto para todos los estudiantes.",
        "Los recursos de la biblioteca son limitados.",
        "Falta de materiales educativos de calidad.",
        "Los estudiantes no tienen acceso a recursos de aprendizaje adecuados.",
        "La biblioteca no tiene suficientes recursos digitales."
    ],
    'Measurement (Medicion)': [
        "Los metodos de calificacion no son consistentes.",
        "Las herramientas de recoleccion de datos son obsoletas.",
        "Falta de precision en la medicion del rendimiento estudiantil.",
        "Los estandares de calidad no se cumplen en las evaluaciones.",
        "La universidad no tiene un sistema de medicion eficiente."
    ],
    'Mother Nature (Entorno)': [
        "Las instalaciones del campus estan en mal estado.",
        "Los servicios de apoyo para estudiantes son inadecuados.",
        "Falta de mantenimiento en las instalaciones.",
        "El entorno del campus no es seguro.",
        "Los estudiantes no estan satisfechos con las instalaciones."
    ]
}

etiquetas_iniciales = [
    "insatisfaccion laboral de los docentes",
    "falta de materiales educativos",
    "Mala precision en las medidas de precisión de datos",
    "Salones de computo optimos",
    "descontento con la calidad de los cursos",
    "disminucion de la tasa de retencion",
    "aumento en la tasa de desercion de los estudiantes",
    "baja satisfaccion de los estudiantes",
    "cursos no cumplen expectativas",
    "tasa de desercion alarmante",
    "satisfaccion laboral de los docentes",
    "ausentismo de los profesores",
    "falta de personal capacitado",
    "carga laboral de los profesores",
    "bajo rendimiento de los docentes",
    "planes de estudio desactualizados",
    "resultados de examenes bajos",
    "metodologias de ensenanza inefectivas",
    "falta de innovacion en ensenanza",
    "insatisfaccion con evaluacion",
    "plataformas dificiles de usar",
    "recursos digitales insuficientes",
    "equipos tecnologicos obsoletos",
    "falta de acceso a herramientas",
    "software desactualizado",
    "libros de texto insuficientes",
    "recursos de biblioteca limitados",
    "falta de materiales educativos",
    "recursos de aprendizaje inadecuados",
    "biblioteca con recursos digitales insuficientes",
    "calificacion inconsistente",
    "herramientas de recoleccion obsoletas",
    "falta de precision en medicion",
    "estandares de calidad no cumplidos",
    "sistema de medicion ineficiente",
    "instalaciones en mal estado",
    "servicios de apoyo inadecuados",
    "falta de mantenimiento",
    "entorno del campus inseguro",
    "insatisfaccion con instalaciones"
]

# Preprocesar y agregar los datos iniciales
for categoria, comentarios in datos_iniciales.items():
    for comentario, etiqueta in zip(comentarios, etiquetas_iniciales):
        agregar_comentario(categoria, comentario, etiqueta)

# Guardar los datos del usuario
guardar_datos_usuario()

# Entrenar el modelo
modelos = entrenar_modelo()

# Generar respuesta final
respuesta_final = generar_respuesta_final()
print("Respuesta Final:", respuesta_final)
