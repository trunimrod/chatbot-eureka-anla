# 2_app_chatbot.py (Versión Final con Parche para Despliegue)

# --- PARCHE PARA SQLITE3 EN STREAMLIT CLOUD ---
# Carga una versión compatible de SQLite3 antes de que chromadb la necesite.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- FIN DEL PARCHE ---

import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Importar los dos prompts
from prompts import EUREKA_PROMPT, EXTRACTOR_PROMPT

# --- Configuración ---
DIRECTORIO_CHROMA_DB = "chroma_db"
MODELO_EMBEDDING = "nomic-embed-text"
MODELO_LLM = "llama3.2"
OLLAMA_HOST = "https://ce25c12c321e.ngrok-free.app" # Reemplaza con tu URL de ngrok para despliegue

# --- CADENA DE CONVERSACIÓN DE DOS PASOS ---
@st.cache_resource
def construir_cadena_completa():
    try:
        embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING, base_url=OLLAMA_HOST)
        db = Chroma(persist_directory=DIRECTORIO_CHROMA_DB, embedding_function=embeddings)
        llm = OllamaLLM(model=MODELO_LLM, temperature=0.2, base_url=OLLAMA_HOST)
        retriever = db.as_retriever(search_kwargs={"k": 5})

        cadena_extractora = (
            {"context": retriever, "question": RunnablePassthrough()}
            | EXTRACTOR_PROMPT
            | llm
            | StrOutputParser()
        )

        cadena_traductora = (
            {"technical_summary": cadena_extractora, "original_question": RunnablePassthrough()}
            | EUREKA_PROMPT
            | llm
            | StrOutputParser()
        )
        return cadena_traductora, retriever
        
    except Exception as e:
        st.error(f"Error al construir las cadenas de IA: {e}")
        return None, None

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

cadena_final, retriever = construir_cadena_completa()

if prompt := st.chat_input("Ej: ¿Cuáles son mis derechos si un proyecto me afecta?"):
    if cadena_final and retriever:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Entendiendo tu pregunta, buscando y traduciendo a lenguaje claro..."):
                try:
                    resumen_conversacional = cadena_final.invoke(prompt)
                    
                    documentos_fuente = retriever.invoke(prompt)
                    fuentes = {doc.metadata['source'] for doc in documentos_fuente if 'source' in doc.metadata and doc.metadata['source'] != 'Fuente no encontrada'}
                    
                    respuesta_final = resumen_conversacional
                    if fuentes and "No he encontrado información" not in resumen_conversacional:
                         respuesta_final += "\n\n--- \n**Fuentes Consultadas:**\n"
                         for url in sorted(list(fuentes)):
                             respuesta_final += f"- {url}\n"

                    st.markdown(respuesta_final)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_final})

                    with st.expander("Ver contexto recuperado (para depuración)"):
                        st.json([doc.dict() for doc in documentos_fuente])

                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")
    else:
        st.error("El chatbot no está disponible.")