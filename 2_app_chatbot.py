# 2_app_chatbot.py (Versión con Arquitectura Anti-Alucinación)

# --- PARCHE ROBUSTO PARA SQLITE3 EN STREAMLIT CLOUD ---
try:
    __import__('pysqlite3')
    import sys as _sys
    _sys.modules['sqlite3'] = _sys.modules.pop('pysqlite3')
except (ImportError, KeyError) as e:
    # Si pysqlite3 no está disponible, continuar con SQLite3 estándar
    print(f"Advertencia: No se pudo aplicar parche pysqlite3: {e}")
    print("Continuando con SQLite3 estándar...")
# --- FIN DEL PARCHE ---

import os 
import re
from typing import Dict, Any, Iterable

import streamlit as st
import requests

from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =====================
# PROMPTS (NUEVA ARQUITECTURA ANTI-ALUCINACIÓN)
# =====================

# El primer modelo ahora genera un borrador completo y factual.
DRAFTER_PROMPT = """
Eres un experto analista legal de la ANLA. Tu tarea es generar un borrador de respuesta claro y factual a la pregunta del usuario, basándote ÚNICA Y EXCLUSIVAMENTE en los documentos proporcionados.

**Instrucciones OBLIGATORIAS:**
1.  **Basa cada afirmación en un documento:** Cada frase que escribas debe provenir directamente de la información en los `[DOC #]`.
2.  **Cita directamente en el texto:** Menciona la fuente (ej. "La Sentencia T-704 de 2016...") al hacer una afirmación. Si no hay un número de ley/sentencia, di "la normativa" o "la jurisprudencia". **NUNCA INVENTES NÚMEROS DE LEY O SENTENCIA.**
3.  **Contextualiza la información:** Si un derecho se menciona en un caso específico (ej. "proyecto minero"), acláralo. No lo presentes como una regla general.
4.  **Estructura como una respuesta directa:** Escribe un borrador de respuesta completo y coherente, no solo una lista de hechos.
5.  **Si no hay información:** Responde únicamente con la frase: "No he encontrado información relevante en los documentos proporcionados."

**Documentos:**
---
{context}
---

**Pregunta del usuario:** {question}

**Borrador de respuesta (basado 100% en los documentos):**
"""

# El segundo modelo (Eureka) solo pule el estilo.
STYLER_PROMPT = """
Eres Eureka, un asistente ciudadano de la ANLA. Tu única tarea es tomar el siguiente "Borrador de respuesta" y reescribirlo para que suene más amable, cercano y fácil de entender para un ciudadano.

**REGLAS INQUEBRANTABLES:**
- **NO AÑADAS NINGUNA INFORMACIÓN FÁCTICA NUEVA.** No puedes agregar hechos, datos, ni números de leyes o sentencias que no estén ya en el borrador. Tu trabajo es solo de estilo y tono.
- **MANTÉN TODAS LAS CITAS LEGALES Y REFERENCIAS A DOCUMENTOS EXACTAMENTE IGUAL.** Si el borrador dice "Sentencia T-704 de 2016", tú debes decir "Sentencia T-704 de 2016".
- **NO ELIMINES INFORMACIÓN CLAVE.** Debes mantener la integridad del borrador.
- **Usa un tono servicial y claro.** Usa viñetas y **negritas** para mejorar la legibilidad.
- **Si el borrador dice "No he encontrado información...",** simplemente reescríbelo de forma amable, por ejemplo: "Hola, no he encontrado información precisa sobre lo que me preguntas. ¿Podrías intentar con otras palabras?".


**Borrador de respuesta:**
---
{respuesta_tecnica}
---

**Respuesta final de Eureka (versión estilizada):**
"""


# =====================
# Configuración
# =====================
DIRECTORIO_CHROMA_DB = os.environ.get("CHROMA_DB_DIR", "chroma_db")
MODELO_EMBEDDING = os.environ.get("EMBED_MODEL", "nomic-embed-text")
MODELO_LLM = os.environ.get("LLM_MODEL", "llama3.2")
NOMBRE_COLECCION = "sentencias_anla" 

# Parámetros MMR optimizados para mayor relevancia
K_DOCUMENTOS = 5
FETCH_K = 25
MMR_LAMBDA = 0.5

# Contexto
MAX_CONTEXT_CHARS = 12000

st.set_page_config(page_title="Eureka — ANLA", page_icon="💬", layout="centered")

# =====================
# Utilidades básicas
# =====================
def limitar_contexto(documentos, max_chars: int) -> str:
    """Combina documentos respetando límite de caracteres"""
    piezas, total = [], 0
    for i, d in enumerate(documentos, 1):
        txt = (d.page_content or "").strip()
        header = f"\n\n[DOC {i}, Título: {d.metadata.get('title', 'N/A')}]\n"
        chunk = header + txt
        if total + len(chunk) > max_chars:
            restante = max_chars - total
            if restante > len(header):
                piezas.append(header + txt[: restante - len(header)])
            break
        piezas.append(chunk); total += len(chunk)
    return "".join(piezas).strip()

def _safe_get_source(doc):
    """Extrae fuente del documento de forma segura"""
    src = (doc.metadata or {}).get("source")
    return src or "Fuente no encontrada"

# ======== Clasificador de intención simple ========
_GREET_WORDS = ["hola","holi","hello","hey","buenas","buenos días","buenas tardes","buenas noches"]
_SMALLTALK_PAT = re.compile(r"(cómo estás|que tal|qué tal|gracias|de nada|ok|vale|listo|perfecto)", re.I)
_QWORDS_PAT = re.compile(r"\b(qué|que|cómo|como|cuál|cual|cuándo|cuando|dónde|donde|por qué|porque|quién|quien|cuánto|cuanto)\b", re.I)

def clasificar_intencion(texto: str) -> str:
    """Clasificador simple para evitar RAG en saludos"""
    t = (texto or "").strip()
    tl = t.lower()
    if not tl:
        return "vacio"
    if any(tl.startswith(w) or w in tl for w in _GREET_WORDS):
        if "?" not in tl and not _QWORDS_PAT.search(tl) and len(tl.split()) <= 4:
            return "saludo"
    if _SMALLTALK_PAT.search(tl):
        return "charla"
    dom_kw = ["anla","licencia","licenciamiento","ambiental","eia","pma","permiso","resolución","audiencia",
              "sustracción","forestal","vertimiento","ruido","emisión","mina","hidrocarburos","energía","proyecto",
              "evaluación","impacto","autoridad","trámite","expediente","compensación","participación","consulta"]
    if _QWORDS_PAT.search(tl) or "?" in tl or any(k in tl for k in dom_kw):
        return "consulta"
    return "indeterminado"

# ======== Conexión Ollama ========
def _get_query_param(name: str) -> str:
    try:
        return st.query_params.get(name, "")
    except Exception:
        try:
            return st.experimental_get_query_params().get(name, [""])[0]
        except Exception:
            return ""

def _normalize_base_url(url: str) -> str:
    base = (url or "").strip()
    if not base:
        raise ValueError("No se proporcionó URL pública para Ollama.")
    if "://" not in base:
        base = "https://" + base
    base = base.rstrip("/")
    if any(h in base for h in ["localhost", "127.0.0.1", "0.0.0.0"]):
        raise ValueError("La URL apunta a host local. Usa la URL pública (ngrok/Cloudflare).")
    return base

def _health_check_ollama(base: str, timeout: float = 5.0):
    try:
        r = requests.get(f"{base}/api/tags", timeout=timeout)
        r.raise_for_status()
        return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

# =====================
# Helpers de prompts
# =====================
def _ensure_prompt(tpl_or_prompt):
    if isinstance(tpl_or_prompt, str):
        return PromptTemplate.from_template(tpl_or_prompt)
    return tpl_or_prompt

def _build_kwargs_for_prompt(prompt: PromptTemplate, **values: Any) -> Dict[str, Any]:
    wanted = set(getattr(prompt, "input_variables", []) or [])
    out: Dict[str, Any] = {}
    def pick(cands: Iterable[str], key: str):
        for c in cands:
            if c in wanted and key in values:
                out[c] = values[key]; return
    pick(["context", "contexto", "context_text"], "context")
    pick(["question", "pregunta", "query", "user_question", "original_question"], "question")
    pick(["respuesta_tecnica", "technical_answer", "answer", "summary", "respuesta", "technical_summary"], "respuesta_tecnica")
    return out

# =====================
# Carga de componentes
# =====================
@st.cache_resource(show_spinner=False)
def cargar_componentes(base_url: str):
    ok, detail = _health_check_ollama(base_url)
    if not ok:
        raise RuntimeError(f"No puedo conectarme a Ollama en {base_url}. Detalle: {detail}")
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING, base_url=base_url)
    db = Chroma(
        persist_directory=DIRECTORIO_CHROMA_DB, 
        embedding_function=embeddings,
        collection_name=NOMBRE_COLECCION
    )
    llm_drafter = OllamaLLM(model=MODELO_LLM, temperature=0.1, base_url=base_url)
    llm_styler_stream = OllamaLLM(model=MODELO_LLM, temperature=0.3, base_url=base_url, streaming=True)
    return embeddings, db, llm_drafter, llm_styler_stream

@st.cache_resource(show_spinner=False)
def construir_cadenas(llm_drafter: OllamaLLM, llm_styler_stream: OllamaLLM):
    drafter_pt = _ensure_prompt(DRAFTER_PROMPT)
    styler_pt = _ensure_prompt(STYLER_PROMPT)
    drafter_chain = drafter_pt | llm_drafter | StrOutputParser()
    styler_stream_chain = styler_pt | llm_styler_stream | StrOutputParser()
    return drafter_chain, styler_stream_chain, drafter_pt, styler_pt

def crear_retriever(db: Chroma):
    params = {"k": K_DOCUMENTOS, "fetch_k": FETCH_K, "lambda_mult": MMR_LAMBDA}
    retriever = db.as_retriever(search_type="mmr", search_kwargs=params)
    return retriever, params

def contar_indice(db: Chroma) -> int:
    try:
        return int(db._collection.count())
    except Exception:
        return 0

# =====================
# UI
# =====================
st.title("Eureka — ANLA · Asistente ciudadano")
st.caption("Te ayudo a entender tus derechos y deberes ambientales.")

# ---- Sidebar ----
with st.sidebar:
    st.subheader("Conexión a Ollama")
    url_param = _get_query_param("ollama")
    if url_param and "ollama_input" not in st.session_state:
        st.session_state["ollama_input"] = url_param

    default_text = st.session_state.get("ollama_input", os.environ.get("OLLAMA_HOST", "").strip())
    ollama_input = st.text_input(
        "URL pública (ngrok/Cloudflare)", value=default_text, key="ollama_input"
    )
    if st.button("Conectar a Ollama", use_container_width=True):
        try:
            candidate = _normalize_base_url(ollama_input)
            ok, detail = _health_check_ollama(candidate)
            if ok:
                st.session_state["ollama_base"] = candidate
                st.success("Conectado ✅")
            else:
                st.error(f"No conecta: {detail}")
        except Exception as e:
            st.error(f"URL inválida: {e}")

    if "ollama_base" in st.session_state:
        st.caption(f"Usando: `{st.session_state['ollama_base']}`")
        
    st.divider()
    st.subheader("Ejemplos de uso")
    st.write("• ¿Qué derechos tengo si un proyecto me afecta?")
    st.write("• ¿Cómo participar en decisiones ambientales?")
    st.write("• ¿Qué compensaciones puede recibir una comunidad?")

# ---- App Logic ----
if "ollama_base" not in st.session_state:
    st.info("💡 Pega la URL pública de tu túnel y pulsa **Conectar a Ollama**.")
    st.stop()

try:
    embeddings, db, llm_drafter, llm_styler_stream = cargar_componentes(st.session_state["ollama_base"])
except Exception as e:
    st.error(f"⌐ No se pudo conectar con Ollama: {e}")
    st.stop()

drafter_chain, styler_stream_chain, drafter_pt, styler_pt = construir_cadenas(llm_drafter, llm_styler_stream)

if contar_indice(db) == 0:
    st.warning("No encuentro documentos en el índice (Chroma). Verifica que la carpeta `chroma_db` esté disponible.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hola, soy Eureka. Te ayudo a entender tus derechos ambientales y cómo participar en las decisiones que te pueden afectar. ¿En qué puedo ayudarte hoy?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_q := st.chat_input("Escribe tu pregunta…"):
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    intent = clasificar_intencion(user_q)
    
    if intent in ("saludo", "charla") and len(st.session_state.messages) <= 2:
        respuesta_breve = "¡Hola! 👋 Estoy aquí para ayudarte con tus dudas sobre licenciamiento y trámites ambientales. ¿Qué te gustaría saber?"
        with st.chat_message("assistant"):
            st.markdown(respuesta_breve)
        st.session_state.messages.append({"role": "assistant", "content": respuesta_breve})
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Buscando información y preparando respuesta…"):
            try:
                retriever, params = crear_retriever(db)
                docs = retriever.invoke(user_q)

                if not docs:
                    st.info("No encontré información relevante sobre tu consulta. ¿Podrías reformular la pregunta?")
                    st.stop()

                contexto = limitar_contexto(docs, MAX_CONTEXT_CHARS)

                # Paso 1: Generar borrador factual
                drafter_kwargs = _build_kwargs_for_prompt(drafter_pt, context=contexto, question=user_q)
                resp_tecnica = drafter_chain.invoke(drafter_kwargs)

                # Paso 2: Estilizar el borrador
                styler_kwargs = _build_kwargs_for_prompt(styler_pt, respuesta_tecnica=resp_tecnica, question=user_q)
                
                contenedor = st.empty()
                acumulado = ""
                for chunk in styler_stream_chain.stream(styler_kwargs):
                    acumulado += chunk
                    contenedor.markdown(acumulado)
                
                respuesta_final = acumulado

                # Lógica de citación dinámica
                fuentes_usadas_en_borrador = set()
                # Buscar Títulos citados en el borrador
                titulos_citados = re.findall(r'la\s(Sentencia.*?de\s\d{4})|el\s(Decreto.*?de\s\d{4})|la\s(Declaración.*?de\s\d{4})', resp_tecnica, re.IGNORECASE)
                
                # Aplanar la lista de tuplas
                titulos_planos = [item for sublist in titulos_citados for item in sublist if item]

                if titulos_planos:
                    for doc in docs:
                        titulo_doc = doc.metadata.get('title', '')
                        for titulo_citado in titulos_planos:
                            if titulo_citado in titulo_doc:
                                fuentes_usadas_en_borrador.add(_safe_get_source(doc))
                                break

                if not fuentes_usadas_en_borrador and "No he encontrado información" not in resp_tecnica:
                     fuentes_usadas_en_borrador = {_safe_get_source(d) for d in docs if _safe_get_source(d) != "Fuente no encontrada"}

                if fuentes_usadas_en_borrador:
                    fuentes_ordenadas = sorted(list(fuentes_usadas_en_borrador))
                    respuesta_con_fuentes = respuesta_final + "\n\n---\n**Fuentes consultadas:**\n" + "\n".join(f"• {u}" for u in fuentes_ordenadas)
                    contenedor.markdown(respuesta_con_fuentes)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_con_fuentes})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_final})

                with st.expander("🔧 Vista de Administrador - DEPUREMOS EL PROCESO"):
                    st.subheader("1. Documentos Recuperados de la Base de Datos")
                    for i, doc in enumerate(docs, 1):
                        with st.container(border=True):
                            st.write(f"**📄 Documento {i}:** `{_safe_get_source(doc)}`")
                            st.text_area(f"Contenido completo Doc {i}", doc.page_content, height=150, key=f"full_doc_{i}")
                    
                    st.subheader("2. Contexto Enviado al Primer Modelo (Redactor)")
                    st.text_area("Contexto completo", contexto, height=200, key="contexto_completo_debug")
                    
                    st.subheader("3. Borrador de Respuesta (Salida del Redactor)")
                    st.text_area("Respuesta técnica factual", resp_tecnica, height=200, key="respuesta_tecnica_debug")
                    
                    st.subheader("4. Prompt Final Enviado a Eureka (Estilista)")
                    prompt_final_eureka = styler_pt.format(**styler_kwargs)
                    st.text_area("Prompt completo para Eureka", prompt_final_eureka, height=300, key="prompt_final_debug")

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
                st.write("Intenta con otra pregunta o verifica la conexión con Ollama.")

