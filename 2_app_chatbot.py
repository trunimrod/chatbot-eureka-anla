# 2_app_chatbot.py (Versión Simplificada - Enfoque en Especificidad)

# --- PARCHE PARA SQLITE3 EN STREAMLIT CLOUD ---
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

# Importar los prompts simplificados
from prompts import EUREKA_PROMPT, EXTRACTOR_PROMPT, EUREKA_PROMPT_RIGHTS, EXTRACTOR_PROMPT_RIGHTS

# --- Configuración ---
DIRECTORIO_CHROMA_DB = "chroma_db"
MODELO_EMBEDDING = "nomic-embed-text"
MODELO_LLM = "llama3.2"
OLLAMA_HOST = "https://6682052ab53b.ngrok-free.app"  # Reemplaza con tu URL de ngrok

# --- FUNCIONES SIMPLIFICADAS ---
def es_pregunta_especifica(pregunta: str) -> bool:
    """
    Detecta solo nombres propios reales y proyectos específicos mencionados explícitamente.
    NO marca como específicas las palabras interrogativas comunes.
    """
    if not pregunta:
        return False
        
    patrones_especificos = [
        r'\bembalse\s+del?\s+\w+',  # "embalse del X"
        r'\bproyecto\s+[A-ZÁÉÍÓÚÜÑ]\w+',  # "proyecto X" con mayúscula específica
        r'\bempresa\s+[A-ZÁÉÍÓÚÜÑ]\w+',  # "empresa X" con mayúscula específica  
        r'\b[A-ZÁÉÍÓÚÜÑ]\w+\s+S\.?A\.?S?\.?',  # Empresas con razón social
        r'\bmunicipio\s+de\s+[A-ZÁÉÍÓÚÜÑ]\w+',  # "municipio de X" específico
        r'\bdepartamento\s+del?\s+[A-ZÁÉÍÓÚÜÑ]\w+',  # "departamento de/del X" específico
        # Nombres de proyectos específicos conocidos en la base de datos
        r'\b(cerrejón|guajaro|puerto bolívar|arroyo bruno|media luna)\b',
    ]
    
    return any(re.search(patron, pregunta, re.IGNORECASE) for patron in patrones_especificos)

def es_consulta_derechos(pregunta: str) -> bool:
    """Detecta si la pregunta está relacionada con derechos y participación ciudadana"""
    if not pregunta:
        return False
        
    terminos_derechos = [
        r'\bderech\w+',  # derecho, derechos
        r'\bparticipa\w+',  # participación, participar
        r'\bconsulta\s+previa',
        r'\baudiencia\s+p[uú]blica',
        r'\bseguridad\s+h[ií]drica',
        r'\bafect\w+',  # afecta, afectado, afectada
        r'\bcomunidad\w*',
        r'\bcompensa\w+',  # compensación, compensaciones
    ]
    
    return any(re.search(termino, pregunta, re.IGNORECASE) for termino in terminos_derechos)

def ajustar_parametros_busqueda(pregunta: str) -> dict:
    """Ajusta parámetros de búsqueda según el tipo de pregunta"""
    if es_pregunta_especifica(pregunta):
        return {"k": 5}  # Más documentos para preguntas específicas
    else:
        return {"k": 3}  # Menos documentos para preguntas generales

def filtrar_para_respuesta_general(documentos, pregunta: str):
    """
    Si la pregunta es general, prioriza documentos normativos sobre casos específicos.
    Filtrado simple y efectivo.
    """
    if es_pregunta_especifica(pregunta):
        return documentos
    
    # Para preguntas generales, ordenar poniendo primero normativa/guías
    def prioridad_doc(doc):
        fuente = ""
        try:
            fuente = (doc.metadata.get('source', '') or "").lower()
        except:
            pass
            
        # Alta prioridad: normativa, leyes, decretos, guías
        if any(term in fuente for term in ["/normativa/", "ley", "decreto", "guia", "guía", "manual", "resolucion"]):
            return 1
        # Baja prioridad: jurisprudencia (casos específicos)
        elif "/jurisprudencia/" in fuente or "sentencia" in fuente:
            return 3
        # Prioridad media: otros documentos
        else:
            return 2
    
    documentos_ordenados = sorted(documentos, key=prioridad_doc)
    
    # Para preguntas generales, limitar a documentos más relevantes
    return documentos_ordenados[:3] if len(documentos_ordenados) > 3 else documentos_ordenados

# --- CADENAS SIMPLIFICADAS ---
@st.cache_resource
def construir_cadenas():
    try:
        embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING, base_url=OLLAMA_HOST)
        db = Chroma(persist_directory=DIRECTORIO_CHROMA_DB, embedding_function=embeddings)
        llm = OllamaLLM(model=MODELO_LLM, temperature=0.3, base_url=OLLAMA_HOST)
        
        # Cadenas estándar
        extractor_chain = EXTRACTOR_PROMPT | llm | StrOutputParser()
        eureka_chain = EUREKA_PROMPT | llm | StrOutputParser()
        
        # Cadenas para derechos
        extractor_rights_chain = EXTRACTOR_PROMPT_RIGHTS | llm | StrOutputParser()
        eureka_rights_chain = EUREKA_PROMPT_RIGHTS | llm | StrOutputParser()
        
        return db, extractor_chain, eureka_chain, extractor_rights_chain, eureka_rights_chain
        
    except Exception as e:
        st.error(f"Error al construir las cadenas de IA: {e}")
        return None, None, None, None, None

def verificar_estado_db(db):
    """Verifica si la base de datos ChromaDB está funcionando correctamente"""
    try:
        count = db._collection.count()
        return True, f"Base de datos OK - {count} documentos"
    except Exception as e:
        return False, f"Error en base de datos: {e}"

def procesar_pregunta(pregunta: str, db, extractor_chain, eureka_chain, extractor_rights_chain, eureka_rights_chain):
    """Proceso principal simplificado de RAG"""
    
    # Verificar estado de la base de datos antes de procesar
    db_ok, db_status = verificar_estado_db(db)
    if not db_ok:
        return f"❌ **Error de Base de Datos**: {db_status}\n\nPor favor, verifica que la carpeta `chroma_db` esté presente y contenga los documentos indexados correctamente.", []
    
    # Determinar tipo de consulta
    es_especifica = es_pregunta_especifica(pregunta)
    es_derechos = es_consulta_derechos(pregunta)
    
    # Seleccionar cadenas apropiadas
    if es_derechos:
        extractor_actual = extractor_rights_chain
        eureka_actual = eureka_rights_chain
    else:
        extractor_actual = extractor_chain
        eureka_actual = eureka_chain
    
    try:
        # Búsqueda en la base de conocimientos
        params = ajustar_parametros_busqueda(pregunta)
        retriever = db.as_retriever(search_kwargs=params)
        documentos_raw = retriever.invoke(pregunta)
        
        if not documentos_raw:
            return "No he encontrado información relevante sobre tu consulta en la base de conocimientos disponible. ¿Podrías reformular tu pregunta o ser más específico sobre qué aspecto te interesa?", []
        
        # Filtrado para mantener enfoque general/específico apropiado
        documentos = filtrar_para_respuesta_general(documentos_raw, pregunta)
        
        # Crear contexto
        contexto = "\n\n".join([doc.page_content for doc in documentos])
        
        # Paso 1: Extracción técnica
        respuesta_tecnica = extractor_actual.invoke({
            "context": contexto,
            "question": pregunta
        })
        
        # Paso 2: Traducción a lenguaje claro
        respuesta_final = eureka_actual.invoke({
            "original_question": pregunta,
            "technical_summary": respuesta_tecnica
        })
        
        return respuesta_final, documentos
        
    except Exception as e:
        return f"❌ **Error durante el procesamiento**: {str(e)}\n\nIntenta con otra pregunta o verifica la configuración del sistema.", []

# --- INTERFAZ DE USUARIO ---
st.set_page_config(
    page_title="Chatbot Eureka - ANLA", 
    page_icon="💬",
    layout="centered"
)

# Encabezados
col1, col2 = st.columns(2)
with col1: 
    st.image("https://www.anla.gov.co/images/logos/herramientas/entidad/logo-anla-2024-ph1.png", width=200)
with col2: 
    st.image("https://www.anla.gov.co/07rediseureka2024/images/planeureka/logo-eureka-2.0.png", width=200)

st.title("Chatbot Eureka")
st.caption("Asistente para derechos y deberes ambientales - Versión Simplificada")
st.divider()

# Inicializar historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hola, soy Eureka. Te ayudo a entender tus derechos ambientales y cómo participar en las decisiones que te pueden afectar. ¿En qué puedo ayudarte hoy?"
    }]

# Mostrar historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Construir cadenas (solo una vez gracias al cache)
db, extractor_chain, eureka_chain, extractor_rights_chain, eureka_rights_chain = construir_cadenas()

# Input del usuario
if prompt := st.chat_input("Ejemplo: ¿Cuáles son mis derechos si un proyecto me afecta?"):
    if not all([db, extractor_chain, eureka_chain, extractor_rights_chain, eureka_rights_chain]):
        st.error("El chatbot no está disponible. Verifica la conexión con Ollama.")
        st.stop()
    
    # Agregar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Procesar y responder
    with st.chat_message("assistant"):
        with st.spinner("Buscando información y preparando respuesta..."):
            try:
                respuesta_final, documentos_fuente = procesar_pregunta(
                    prompt, db, extractor_chain, eureka_chain, 
                    extractor_rights_chain, eureka_rights_chain
                )
                
                # Agregar fuentes si las hay
                if documentos_fuente and "No he encontrado información" not in respuesta_final:
                    fuentes_unicas = set()
                    for doc in documentos_fuente:
                        try:
                            fuente = doc.metadata.get('source', 'Fuente no disponible')
                            if fuente != 'Fuente no disponible':
                                fuentes_unicas.add(fuente)
                        except:
                            pass
                    
                    if fuentes_unicas:
                        respuesta_final += "\n\n---\n**Fuentes Consultadas:**\n"
                        for i, fuente in enumerate(sorted(fuentes_unicas), 1):
                            respuesta_final += f"{i}. {fuente}\n"
                
                st.markdown(respuesta_final)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
                
                # Información de depuración (opcional)
                with st.expander("Ver información técnica"):
                    es_especifica = es_pregunta_especifica(prompt)
                    es_derechos = es_consulta_derechos(prompt)
                    st.write(f"**Tipo de pregunta:** {'Específica' if es_especifica else 'General'}")
                    st.write(f"**Consulta de derechos:** {'Sí' if es_derechos else 'No'}")
                    st.write(f"**Documentos encontrados:** {len(documentos_fuente)}")
                    if documentos_fuente:
                        st.json([{
                            "fuente": doc.metadata.get('source', 'N/A'),
                            "contenido_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        } for doc in documentos_fuente])
                
            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
                st.write("Por favor, intenta con otra pregunta o verifica la conexión.")

# Información adicional en el sidebar
with st.sidebar:
    st.header("Información del Sistema")
    st.write("**Versión:** Simplificada v1.0")
    st.write("**Plataforma:** Streamlit Cloud")
    
    # Diagnóstico del sistema
    st.subheader("Estado del Sistema")
    archivos_ok, archivos_msg = verificar_archivos_chroma()
    st.write(f"**ChromaDB:** {'✅' if archivos_ok else '❌'} {archivos_msg}")
    st.write(f"**Path ChromaDB:** `{DIRECTORIO_CHROMA_DB}`")
    
    # Verificar conexión Ollama
    try:
        import requests
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        ollama_status = "✅ Conectado" if response.status_code == 200 else "❌ Error"
    except:
        ollama_status = "❌ No conectado"
    
    st.write(f"**Ollama:** {ollama_status}")
    
    st.subheader("Ejemplos de preguntas")
    st.write("**Generales:**")
    st.write("- ¿Qué derechos tengo si un proyecto me afecta?")
    st.write("- ¿Cómo puedo participar en decisiones ambientales?")
    st.write("- ¿Qué compensaciones puede recibir una comunidad?")
    
    st.write("**Específicas:**")
    st.write("- ¿Qué pasó con el proyecto Cerrejón?")
    st.write("- ¿Cómo afecta el embalse del Guajaro?")
    st.write("- ¿Qué dice la Ley 99 sobre licencias?")