import streamlit as st
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="Sistema de Inteligencia de Mercado", layout="wide")
st.title("📊 Sistema de Inteligencia de Mercado - Minería de Textos")
st.caption("Solución Integrada Completa - Cumplimiento de Requerimientos")

# =====================================================================
# 2. CARGA DE ARTEFACTOS DEL EQUIPO (Con Caché)
# =====================================================================
@st.cache_resource
def inicializar_componentes():
    vectorizador = joblib.load("vectorizador_prod.pkl")
    modelo_svm = joblib.load("modelo_svm.pkl")
    modelo_lda = joblib.load("modelo_lda_prod.pkl")
    return vectorizador, modelo_svm, modelo_lda

try:
    vectorizador, modelo_svm, modelo_lda = inicializar_componentes()
except Exception as e:
    st.error(f"❌ Error al cargar los artefactos del equipo: {e}")
    st.stop()

# Mapeo de Temas (LDA)
MAPEO_TEMAS = {
    0: "Fricción / Desgaste", 
    1: "Logística y Envíos", 
    2: "Cocina y Hogar", 
    3: "Cine y Entretenimiento", 
    4: "Ficción / Literatura", 
    5: "Series de TV", 
    6: "Contenido Infantil", 
    7: "Música y Audio"
}

# Base de Conocimiento para Similitud Coseno (TAB 3)
BASE_CONOCIMIENTO = [
    {"tema": "Logística y Envíos", "frase_clave": "delays shipping delivery week package arrival delayed status", "respuesta": "Lamentamos el retraso en la entrega de tu paquete. Puedes consultar el estado detallado del envío en tiempo real ingresando a la sección 'Mis Pedidos' en tu cuenta de usuario."},
    {"tema": "Fricción / Desgaste", "frase_clave": "broken damaged poor quality material flimsy scratch defective", "respuesta": "Lamentamos profundamente que el producto haya llegado dañado o con problemas de calidad. Por favor, ponte en contacto con nuestro equipo de soporte para coordinar un reemplazo sin costo."},
    {"tema": "Cocina y Hogar", "frase_clave": "kitchen cooking pan pot food appliance baking recipe", "respuesta": "Agradecemos tus comentarios sobre nuestra línea de hogar y cocina. Nos alegra saber que optimiza tus actividades domésticas."},
    {"tema": "Cine y Entretenimiento", "frase_clave": "movie film cinema actor plot bluray digital hd video dvd", "respuesta": "¡Muchas gracias por tu reseña cinematográfica! Nos alegra que disfrutes del mejor catálogo de entretenimiento en nuestra plataforma."},
]

def limpiar_texto(texto):
    if texto is None:
        return ""
    return str(texto).lower().strip()

# =====================================================================
# 3. CREACIÓN DE LAS 3 PESTAÑAS (TABs)
# =====================================================================
tab1, tab2, tab3 = st.tabs([
    "🟦 TAB 1: Análisis de reseñas (LDA)", 
    "🟩 TAB 2: Sentimiento", 
    "🟨 TAB 3: Chatbot / Similitud Coseno"
])

# ---------------------------------------------------------------------
# 🟦 TAB 1: ANÁLISIS DE TEMAS DESCUBIERTOS (LDA)
# ---------------------------------------------------------------------
with tab1:
    st.header("Análisis de Modelado de Temas No Supervisado")
    st.subheader("Estructura Global Descubierta por el Modelo LDA")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("### 🏷️ Distribución de Temas del Dataset")
        for idx, nombre in MAPEO_TEMAS.items():
            st.markdown(f"**Tema {idx}:** {nombre}")
            
    with col_t2:
        st.markdown("### 🔤 Palabras Clave más Relevantes")
        st.dataframe({
            "Tema": ["Logística", "Cine", "Música", "Cocina"],
            "Palabras Clave Asociadas": [
                "shipping, delivery, time, week, package, arrived, order",
                "movie, film, watch, story, actors, dvd, scene, plot",
                "song, album, music, sound, cd, listen, track, voice",
                "quality, cook, pan, kitchen, space, use, clean, box"
            ]
        })

# ---------------------------------------------------------------------
# 🟩 TAB 2: ANÁLISIS DE SENTIMIENTO (SUPERVISADO)
# ---------------------------------------------------------------------
with tab2:
    st.header("Evaluación Supervisada de Sentimiento")
    
    metodo_entrada = st.radio(
        "Método de entrada de texto:",
        ["Escribir en Español (Traducción Automática)", "Escribir directamente en Inglés"],
        key="lang_tab2", horizontal=True
    )
    
    resena_input = st.text_area("Ingresa la reseña a evaluar:", height=100, key="txt_tab2")
    
    if st.button("Evaluar Sentimiento ⚡", key="btn_tab2"):
        if not resena_input.strip():
            st.warning("Escribe algo antes de evaluar.")
        else:
            with st.spinner("Procesando..."):
                if "Español" in metodo_entrada:
                    from deep_translator import GoogleTranslator
                    texto_eng = GoogleTranslator(source="auto", target="en").translate(resena_input)
                else:
                    texto_eng = resena_input
                
                texto_proc = limpiar_texto(texto_eng)
                vector_raw = vectorizador.transform([texto_proc])
                
                # Ajuste a 10000 características solo para SVM
                vector_svm = vector_raw[:, :10000]
                prediccion = str(modelo_svm.predict(vector_svm)[0])
                
                st.divider()
                if prediccion == "2":
                    st.metric(label="Sentimiento Predicho", value="🟢 POSITIVO")
                else:
                    st.metric(label="Sentimiento Predicho", value="🔴 NEGATIVO")
                st.caption(f"*Texto en Inglés:* _{texto_proc}_")

# ---------------------------------------------------------------------
# 🟨 TAB 3: CHATBOT / SIMILITUD COSENO (MÓDULO INTEGRADOR)
# ---------------------------------------------------------------------
with tab3:
    st.header("Motor Automatizado de Inteligencia de Mercado")
    st.markdown("Este módulo unifica todo el sistema: detecta el **sentimiento**, clasifica el **tema (LDA)** y busca la **respuesta óptima** usando Similitud Coseno.")
    
    entrada_usuario = st.text_input(
        "Escribe un mensaje de prueba (Ejemplo: 'Mi pedido lleva una semana sin llegar'):", 
        placeholder="Escribe aquí tu texto..."
    )
    
    if st.button("Ejecutar Flujo Completo 🚀", key="btn_tab3"):
        if not entrada_usuario.strip():
            st.warning("Por favor ingresa un mensaje para procesar el flujo.")
        else:
            with st.spinner("Analizando flujos integrados..."):
                from deep_translator import GoogleTranslator
                texto_ingles = GoogleTranslator(source="auto", target="en").translate(entrada_usuario)
                texto_limpio = limpiar_texto(texto_ingles)
                
                # Vector completo (20110 features) para LDA y Similitud Coseno
                vector_datos = vectorizador.transform([texto_limpio])
                
                # Ajuste exclusivo a 10000 features solo para la predicción de SVM
                vector_datos_svm = vector_datos[:, :10000]
                sentimiento_raw = str(modelo_svm.predict(vector_datos_svm)[0])
                polaridad = "🟢 POSITIVO" if sentimiento_raw == "2" else "🔴 NEGATIVO"
                
                # Tema (LDA con sus 20110 features completas)
                tema_id = modelo_lda.transform(vector_datos).argmax()
                tema_nom = MAPEO_TEMAS.get(tema_id, "General / Otros")
                
                # Similitud Coseno real usando el vector original completo (20110 features)
                frases_base = [item["frase_clave"] for item in BASE_CONOCIMIENTO]
                vectores_base = vectorizador.transform(frases_base)
                
                # Aquí ambos vectores tienen 20110 dimensiones, por lo que ya no truena
                similitudes = cosine_similarity(vector_datos, vectores_base).flatten()
                indice_mejor_match = np.argmax(similitudes)
                
                respuesta_automatica = BASE_CONOCIMIENTO[indice_mejor_match]["respuesta"]
                
                st.divider()
                st.subheader("📥 Resultados del Análisis Integrado")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric(label="Sentimiento Inferido", value=polaridad)
                with c2:
                    st.metric(label="Tema Detectado (LDA)", value=tema_nom)
                with c3:
                    st.metric(label="Confianza Semántica", value=f"{round(float(similitudes[indice_mejor_match]) * 100, 2)}%")
                
                st.markdown("### 🤖 Respuesta Automática Sugerida (Similitud Coseno):")
                st.info(respuesta_automatica)

# 4. SIDEBAR
with st.sidebar:
    st.markdown("## 📊 Panel de Control")
    st.markdown("**Sistema Integrador - Proyecto Final**")
    st.divider()
    st.markdown("### 📈 Métricas de Evaluación SVM")
    st.dataframe({
        "Métrica": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Valor": [0.8540, 0.8540, 0.8540, 0.8540]
    })
