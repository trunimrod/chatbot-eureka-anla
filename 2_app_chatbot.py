# 2_app_chatbot.py (Versión Mejorada - Sin Filtración de Información Específica)

# --- PARCHE PARA SQLITE3 EN STREAMLIT CLOUD ---
# Carga una versión compatible de SQLite3 antes de que chromadb la necesite.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- FIN DEL PARCHE ---

import streamlit as st
import re
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Importar los dos prompts mejorados
from prompts import EUREKA_PROMPT, EXTRACTOR_PROMPT

# --- Configuración ---
DIRECTORIO_CHROMA_DB = "chroma_db"
MODELO_EMBEDDING = "nomic-embed-text"
MODELO_LLM = "llama3.2"
OLLAMA_HOST = "https://6682052ab53b.ngrok-free.app" # Reemplaza con tu URL de ngrok para despliegue

# --- FUNCIONES AUXILIARES ---
def es_pregunta_especifica(pregunta):
    """
    Detecta si la pregunta menciona nombres específicos de proyectos, lugares, empresas, etc.
    Esto ayuda a ajustar los parámetros de búsqueda y procesamiento.
    """
    patrones_especificos = [
        r'\b[A-Z][a-záéíóúüñç]+(?:\s+[A-Z][a-záéíóúüñç]+)*\b',  # Nombres propios
        r'\bembalse\s+del?\s+\w+',  # "embalse del X"
        r'\bproyecto\s+\w+',  # "proyecto X"
        r'\bempresa\s+\w+',  # "empresa X"
        r'\b\w+\s+S\.?A\.?S?\.?',  # Empresas con razón social
        r'\bmunicipio\s+de\s+\w+',  # "municipio de X"
        r'\bdepartamento\s+del?\s+\w+',  # "departamento de/del X"
    ]
    return any(re.search(patron, pregunta, re.IGNORECASE) for patron in patrones_especificos)

def ajustar_parametros_busqueda(pregunta):
    """
    Ajusta los parámetros de búsqueda según si la pregunta es específica o general.
    """
    if es_pregunta_especifica(pregunta):
        return {"k": 5, "score_threshold": 0.6}  # Más documentos para preguntas específicas
    else:
        return {"k": 3, "score_threshold": 0.7}  # Menos documentos, más selectivos para preguntas generales

# --- CADENA DE CONVERSACIÓN DE DOS PASOS MEJORADA ---
@st.cache_resource
def construir_cadena_completa():
    try:
        embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING, base_url=OLLAMA_HOST)
        db = Chroma(persist_directory=DIRECTORIO_CHROMA_DB, embedding_function=embeddings)
        llm = OllamaLLM(model=MODELO_LLM, temperature=0.2, base_url=OLLAMA_HOST)
        
        # Función para crear retriever dinámico según la pregunta
        def crear_retriever_dinamico(pregunta):
            params = ajustar_parametros_busqueda(pregunta)
            return db.as_retriever(search_kwargs=params)

        # Cadena extractora con retriever dinámico
        def cadena_extractora_dinamica(pregunta):
            retriever = crear_retriever_dinamico(pregunta)
            cadena = (
                {"context": retriever, "question": RunnablePassthrough()}
                | EXTRACTOR_PROMPT
                | llm
                | StrOutputParser()
            )
            return cadena.invoke(pregunta), retriever.invoke(pregunta)

        def cadena_completa(pregunta):
            resumen_tecnico, documentos = cadena_extractora_dinamica(pregunta)
            
            cadena_traductora = (
                {"technical_summary": lambda x: resumen_tecnico, "original_question": RunnablePassthrough()}
                | EUREKA_PROMPT
                | llm
                | StrOutputParser()
            )
            
            respuesta_final = cadena_traductora.invoke(pregunta)
            return respuesta_final, documentos

        return cadena_completa
        
    except Exception as e:
        st.error(f"Error al construir las cadenas de IA: {e}")
        return None

# --- Interfaz de Usuario ---
st.set_page_config(page_title="Chatbot Eureka - ANLA", page_icon="https://www.anla.gov.co/07rediseureka2024/images/planeureka/logo-eureka-2.0.png")
col1, col2 = st.columns(2)
with col1: st.image("https://www.anla.gov.co/images/logos/herramientas/entidad/logo-anla-2024-ph1.png", width=200)
with col2: st.image("https://www.anla.gov.co/07rediseureka2024/images/planeureka/logo-eureka-2.0.png", width=200)
st.title("Chatbot Eureka")
st.caption("Te ayudo a comprender tus derechos y deberes ambientales.")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hola, soy Eureka. Estoy aquí para ayudarte a entender la información ambiental y cómo puedes participar en las decisiones que te afectan. ¿En qué puedo ayudarte hoy?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

cadena_final = construir_cadena_completa()

if prompt := st.chat_input("Ej: ¿Cuáles son mis derechos si un proyecto me afecta?"):
    if cadena_final:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Entendiendo tu pregunta, buscando y traduciendo a lenguaje claro..."):
                try:
                    # Mostrar información de depuración sobre el tipo de pregunta
                    tipo_pregunta = "específica" if es_pregunta_especifica(prompt) else "general"
                    params = ajustar_parametros_busqueda(prompt)
                    
                    respuesta_final, documentos_fuente = cadena_final(prompt)
                    
                    fuentes = {doc.metadata['source'] for doc in documentos_fuente if 'source' in doc.metadata and doc.metadata['source'] != 'Fuente no encontrada'}
                    
                    if fuentes and "No he encontrado información" not in respuesta_final:
                         respuesta_final += "\n\n--- \n**Fuentes Consultadas:**\n"
                         for url in sorted(list(fuentes)):
                             respuesta_final += f"- {url}\n"

                    st.markdown(respuesta_final)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_final})

                    with st.expander("Ver información de depuración"):
                        st.write(f"**Tipo de pregunta detectado:** {tipo_pregunta}")
                        st.write(f"**Parámetros de búsqueda:** {params}")
                        st.write(f"**Documentos recuperados:** {len(documentos_fuente)}")
                        st.json([doc.dict() for doc in documentos_fuente])

                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")
    else:
        st.error("El chatbot no está disponible.")