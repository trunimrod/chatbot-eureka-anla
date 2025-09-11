# 2_app_chatbot.py (Versión Final Corregida - Sin score_threshold)

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
    Solo usa 'k' ya que ChromaDB no soporta score_threshold en el retriever.
    """
    if es_pregunta_especifica(pregunta):
        return {"k": 5}  # Más documentos para preguntas específicas
    else:
        return {"k": 3}  # Menos documentos para preguntas generales

def filtrar_documentos_por_relevancia(documentos, pregunta, es_especifica):
    """
    Filtro adicional de documentos basado en relevancia para evitar ruido.
    Para preguntas generales, es más estricto.
    """
    if not documentos:
        return documentos
    
    if not es_especifica:
        # Para preguntas generales, filtrar documentos que contengan demasiados nombres propios
        documentos_filtrados = []
        for doc in documentos:
            # Contar nombres propios y términos muy específicos
            nombres_propios = len(re.findall(r'\b[A-Z][a-záéíóúüñç]+(?:\s+[A-Z][a-záéíóúüñç]+)*\b', doc.page_content))
            # Si tiene menos de 4 nombres propios, es más probable que sea información general
            if nombres_propios < 4:
                documentos_filtrados.append(doc)
        
        # Si el filtrado dejó muy pocos documentos, devolver al menos los 2 primeros originales
        return documentos_filtrados if len(documentos_filtrados) >= 1 else documentos[:2]
    
    return documentos

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

        # Cadena extractora con retriever dinámico y filtrado
        def cadena_extractora_dinamica(pregunta):
            retriever = crear_retriever_dinamico(pregunta)
            documentos_raw = retriever.invoke(pregunta)
            
            # Aplicar filtrado adicional para preguntas generales
            es_especifica = es_pregunta_especifica(pregunta)
            documentos_filtrados = filtrar_documentos_por_relevancia(documentos_raw, pregunta, es_especifica)
            
            # Crear contexto manualmente con los documentos filtrados
            contexto = "\n\n".join([doc.page_content for doc in documentos_filtrados])
            
            # Procesar con el prompt del extractor
            cadena = EXTRACTOR_PROMPT | llm | StrOutputParser()
            resumen_tecnico = cadena.invoke({"context": contexto, "question": pregunta})
            
            return resumen_tecnico, documentos_filtrados

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