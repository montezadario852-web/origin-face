
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image

# CONFIGURACIÓN
st.set_page_config(
    page_title="ORIGIN FACE",
    page_icon="📐",
    layout="wide"
)

# ENCABEZADO
st.title("📐 ORIGIN FACE")
st.subheader("Análisis geométrico e identificación comparativa de rostros")
st.write("Plataforma académica de investigación facial – Amazonas, Perú.")

st.divider()

st.info(
    "Esta aplicación analiza características geométricas faciales. "
    "No determina el origen étnico o geográfico de una persona."
)

# MODELO DE DETECCIÓN
mp_face_mesh = mp.solutions.face_mesh

@st.cache_resource
def cargar_modelo():
    return mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

modelo = cargar_modelo()

# FUNCIONES MATEMÁTICAS
def distancia(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def analizar_rostro(archivo):
    imagen = Image.open(archivo).convert("RGB")
    imagen = np.array(imagen)

    resultado = modelo.process(imagen)

    if not resultado.multi_face_landmarks:
        return None, None

    rostro = resultado.multi_face_landmarks[0]
    alto, ancho, _ = imagen.shape

    indices = {
        "Ojo izquierdo": 33,
        "Ojo derecho": 263,
        "Nariz": 1,
        "Boca izquierda": 61,
        "Boca derecha": 291,
        "Contorno izquierdo": 234,
        "Contorno derecho": 454,
        "Frente": 10,
        "Mentón": 152
    }

    puntos = {}

    for nombre, indice in indices.items():
        punto = rostro.landmark[indice]
        puntos[nombre] = (
            punto.x * ancho,
            punto.y * alto
        )

    ancho_ojos = distancia(
        puntos["Ojo izquierdo"],
        puntos["Ojo derecho"]
    )

    if ancho_ojos == 0:
        return None, None

    ancho_rostro = distancia(
        puntos["Contorno izquierdo"],
        puntos["Contorno derecho"]
    )

    alto_rostro = distancia(
        puntos["Frente"],
        puntos["Mentón"]
    )

    ancho_boca = distancia(
        puntos["Boca izquierda"],
        puntos["Boca derecha"]
    )

    # PROPORCIONES GEOMÉTRICAS
    datos = {
        "Ancho facial / distancia ocular":
            ancho_rostro / ancho_ojos,

        "Alto facial / distancia ocular":
            alto_rostro / ancho_ojos,

        "Ancho de boca / distancia ocular":
            ancho_boca / ancho_ojos
    }

    # DIBUJAR PUNTOS FACIALES
    imagen_resultado = imagen.copy()

    for punto in rostro.landmark:
        x = int(punto.x * ancho)
        y = int(punto.y * alto)

        cv2.circle(
            imagen_resultado,
            (x, y),
            1,
            (0, 200, 255),
            -1
        )

    return imagen_resultado, datos


# MENÚ PRINCIPAL
opcion = st.sidebar.selectbox(
    "MENÚ PRINCIPAL",
    [
        "Inicio",
        "Subir fotografía",
        "Activar cámara",
        "Comparar rostros",
        "Información del proyecto"
    ]
)

# INICIO
if opcion == "Inicio":

    st.header("Bienvenido a ORIGIN FACE")

    st.write(
        "Sistema experimental para la detección y el análisis "
        "matemático de características geométricas faciales."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Módulo 1", "Fotografías")
    col2.metric("Módulo 2", "Cámara")
    col3.metric("Módulo 3", "Comparación")

    st.success("Sistema preparado para analizar fotografías.")

# SUBIR FOTOGRAFÍA
elif opcion == "Subir fotografía":

    st.header("Análisis de fotografía")

    archivo = st.file_uploader(
        "Selecciona una fotografía",
        type=["jpg", "jpeg", "png"]
    )

    if archivo is not None:

        imagen, datos = analizar_rostro(archivo)

        if datos is None:
            st.error("No se detectó ningún rostro.")

        else:
            st.image(
                imagen,
                caption="Rostro detectado",
                use_container_width=True
            )

            st.subheader("Resultados geométricos")

            tabla = pd.DataFrame(
                list(datos.items()),
                columns=["Indicador", "Resultado"]
            )

            tabla["Resultado"] = tabla["Resultado"].round(4)

            st.dataframe(
                tabla,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                tabla.set_index("Indicador")
            )

# CÁMARA
elif opcion == "Activar cámara":

    st.header("Captura facial mediante cámara")

    fotografia = st.camera_input(
        "Activa tu cámara y captura una fotografía"
    )

    if fotografia is not None:

        imagen, datos = analizar_rostro(fotografia)

        if datos is None:
            st.error("No se pudo detectar el rostro.")

        else:
            st.image(
                imagen,
                caption="Análisis facial",
                use_container_width=True
            )

            st.write("Resultados:")

            for nombre, valor in datos.items():
                st.write(f"**{nombre}:** {valor:.4f}")

# COMPARACIÓN
elif opcion == "Comparar rostros":

    st.header("Comparación geométrica facial")

    col1, col2 = st.columns(2)

    with col1:
        foto1 = st.file_uploader(
            "Primera fotografía",
            type=["jpg", "jpeg", "png"],
            key="foto1"
        )

    with col2:
        foto2 = st.file_uploader(
            "Segunda fotografía",
            type=["jpg", "jpeg", "png"],
            key="foto2"
        )

    if foto1 is not None and foto2 is not None:

        _, datos1 = analizar_rostro(foto1)
        _, datos2 = analizar_rostro(foto2)

        if datos1 is None or datos2 is None:

            st.error(
                "No se detectaron correctamente ambos rostros."
            )

        else:

            diferencias = {}

            for nombre in datos1:
                diferencias[nombre] = abs(
                    datos1[nombre] - datos2[nombre]
                )

            distancia_total = np.linalg.norm(
                list(diferencias.values())
            )

            indice = max(
                0,
                100 * (1 - distancia_total / 1.5)
            )

            st.metric(
                "Índice de similitud geométrica experimental",
                f"{indice:.2f}%"
            )

            st.dataframe(
                pd.DataFrame(
                    list(diferencias.items()),
                    columns=["Indicador", "Diferencia"]
                ),
                hide_index=True,
                use_container_width=True
            )

            st.warning(
                "Este índice compara proporciones geométricas. "
                "No confirma identidad personal ni procedencia."
            )

# INFORMACIÓN
elif opcion == "Información del proyecto":

    st.header("Información del proyecto")

    st.write("Nombre: ORIGIN FACE")

    st.write(
        "Área: Inteligencia artificial, visión artificial "
        "y análisis geométrico."
    )

    st.write("Ubicación de referencia: Amazonas, Perú.")

    st.write(
        "Objetivo: desarrollar herramientas matemáticas "
        "para estudiar características faciales en muestras "
        "documentadas y autorizadas."
    )

st.divider()

st.caption(
    "ORIGIN FACE | Proyecto académico de investigación | 2026"
)



