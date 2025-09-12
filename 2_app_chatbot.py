# 2_app_chatbot.py (Basado en versión funcional - Con filtros de especificidad mejorados)

# --- PARCHE PARA SQLITE3 EN STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys as _sys
_sys.modules['sqlite3'] = _sys.modules.pop('pysqlite3')
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

# Importar prompts mejorados
from prompts import EUREKA_PROMPT, EXTRACTOR_PROMPT

# =====================
# Configuración
# =====================
DIRECTORIO_CHROMA_DB = os.environ.get("CHROMA_DB_DIR", "chroma_db")
MODELO_EMBEDDING = os.environ.get("EMBED_MODEL", "nomic-embed-text")
MODELO_LLM = os.environ.get("LLM_MODEL", "llama3.2")

# Parámetros MMR (de la versión funcional)
K_GENERAL = 3
K_ESPECIFICA = 5
FETCH_K = 20
MMR_LAMBDA = 0.5

# Contexto
MAX_CONTEXT_CHARS = 12000

st.set_page_config(page_title="Eureka — ANLA", page_icon="💬", layout="centered")

# =====================
# Utilidades (versión funcional + mejoras de especificidad)
# =====================
def es_pregunta_especifica(pregunta: str) -> bool:
    """
    MEJORADO: Detecta solo nombres específicos reales, no palabras interrogativas.
    Basado en lógica funcional pero más precisa.
    """
    if not pregunta:
        return False
    
    # Patrones más específicos que en la versión original
    patrones = [
        r"\bembalse\s+del?\s+\w+",  # "embalse del X"
        r"\bproyecto\s+[A-ZÁÉÍÓÚÜÑ]\w+",  # "proyecto X" con mayúscula específica
        r"\bempresa\s+[A-ZÁÉÍÓÚÜÑ]\w+",  # "empresa X" con mayúscula específica
        r"\b\w+\s+S\.?A\.?S?\.?",  # Empresas con razón social
        r"\bmunicipio\s+de\s+[A-ZÁÉÍÓÚÜÑ]\w+",  # "municipio de X" específico
        r"\bdepartamento\s+del?\s+[A-ZÁÉÍÓÚÜÑ]\w+",  # "departamento de/del X" específico
        # Casos específicos conocidos
        r"\b(cerrejón|guajaro|puerto bolívar|arroyo bruno|media luna)\b",
    ]
    return any(re.search(p, pregunta, re.IGNORECASE) for p in patrones)

def ajustar_parametros_busqueda(pregunta: str) -> dict:
    """Misma lógica que la versión funcional"""
    k = K_ESPECIFICA if es_pregunta_especifica(pregunta) else K_GENERAL
    return {"k": k, "fetch_k": FETCH_K, "lambda_mult": MMR_LAMBDA}

def filtrar_documentos_por_relevancia(documentos, pregunta: str, es_especifica: bool):
    """
    MEJORADO: Filtrado más inteligente que la versión original.
    Para preguntas generales, evita documentos con muchos nombres específicos.
    """
    if es_especifica:
        return documentos
    
    # Para preguntas generales, filtrar docs con muchos nombres propios
    docs_filtrados = []
    patron_np = re.compile(r"\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)*\b")
    
    for doc in documentos:
        contenido = doc.page_content or ""
        nombres_propios = len(patron_np.findall(contenido))
        # Si tiene pocos nombres propios, probablemente es información general
        if nombres_propios <= 3:
            docs_filtrados.append(doc)
    
    # Si filtrar dejó muy pocos docs, usar los primeros documentos originales
    return docs_filtrados if len(docs_filtrados) >= 1 else documentos[:2]

def limitar_contexto(documentos, max_chars: int) -> str:
    """Misma lógica que la versión funcional"""
    piezas, total = [], 0
    for i, d in enumerate(documentos, 1):
        txt = (d.page_content or "").strip()
        header = f"\n\n[DOC {i}]\n"
        chunk = header + txt
        if total + len(chunk) > max_chars:
            restante = max_chars - total
            if restante > len(header):
                piezas.append(header + txt[: restante - len(header)])
            break
        piezas.append(chunk); total += len(chunk)
    return "".join(piezas).strip()

def _safe_get_source(doc):
    """Misma función que la versión funcional"""
    src = (doc.metadata or {}).get("source")
    return src or "Fuente no encontrada"

# ======== Clasificador de intención (de la versión funcional) ========
_GREET_WORDS = ["hola","holi","hello","hey","buenas","buenos días","buenas tardes","buenas noches"]
_SMALLTALK_PAT = re.compile(r"(cómo estás|que tal|qué tal|gracias|de nada|ok|vale|bkn|listo|perfecto)", re.I)
_QWORDS_PAT = re.compile(r"\b(qué|que|cómo|como|cuál|cual|cuándo|cuando|dónde|donde|por qué|porque|quién|quien|cuánto|cuanto)\b", re.I)

def clasificar_intencion(texto: str) -> str:
    """Misma lógica que la versión funcional"""
    t = (texto or "").strip()
    tl = t.lower()
    if not tl:
        return "vacio"
    if any(tl.startswith(w) or w in tl for w in _GREET_WORDS):
        if "?" not in tl and not _QWORDS_PAT.search(tl) and len(tl.split()) <= 4:
            return "saludo"
    if _SMALLTALK_PAT.search(tl):
        return "charla"
    # palabras clave del dominio
    dom_kw = ["anla","licencia","licenciamiento","ambiental","eia","pma","permiso","resolución","audiencia",
              "sustracción","forestal","vertimiento","ruido","emisión","mina","hidrocarburos","energía","proyecto",
              "evaluación","impacto","autoridad","trámite","expediente"]
    if _QWORDS_PAT.search(tl) or "?" in tl or any(k in tl for k in dom_kw):
        return "consulta"
    return "indeterminado"

# ======== Conexión Ollama (de la versión funcional) ========
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
# Helpers de prompts (de la versión funcional)
# =====================
def _ensure_prompt(tpl_or_prompt):
    if isinstance(tpl_or_prompt, str):
        return PromptTemplate(template=tpl_or_prompt)
    return tpl_or_prompt

def _build_kwargs_for_prompt(prompt: PromptTemplate, **values: Any) -> Dict[str, Any]:
    """Mapea variables de prompt dinámicamente (de la versión funcional)"""
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
# Carga de componentes (EXACTAMENTE como la versión funcional)
# =====================
@st.cache_resource(show_spinner=False)
def cargar_componentes(base_url: str):
    ok, detail = _health_check_ollama(base_url)
    if not ok:
        raise RuntimeError(f"No puedo conectarme a Ollama en {base_url}. Detalle: {detail}")
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING, base_url=base_url)
    db = Chroma(persist_directory=DIRECTORIO_CHROMA_DB, embedding_function=embeddings)
    llm_extract = OllamaLLM(model=MODELO_LLM, temperature=0.2, base_url=base_url)
    llm_eureka_stream = OllamaLLM(model=MODELO_LLM, temperature=0.2, base_url=base_url, streaming=True)
    return embeddings, db, llm_extract, llm_eureka_stream

@st.cache_resource(show_spinner=False)
def construir_cadenas(llm_extract: OllamaLLM, llm_eureka_stream: OllamaLLM):
    extractor_pt = _ensure_prompt(EXTRACTOR_PROMPT)
    eureka_pt = _ensure_prompt(EUREKA_PROMPT)
    extractor = extractor_pt | llm_extract | StrOutputParser()
    eureka_stream_chain = eureka_pt | llm_eureka_stream | StrOutputParser()
    return extractor, eureka_stream_chain, extractor_pt, eureka_pt

def crear_retriever(db: Chroma, pregunta: str):
    """EXACTAMENTE como la versión funcional"""
    params = ajustar_parametros_busqueda(pregunta)
    retriever = db.as_retriever(search_type="mmr", search_kwargs=params)
    return retriever, params

def contar_indice(db: Chroma) -> int:
    """Misma función que la versión funcional"""
    try:
        if hasattr(db, "_collection") and db._collection is not None:
            return int(db._collection.count())
    except Exception:
        pass
    try:
        res = db.similarity_search("prueba", k=1)
        return 1 if res else 0
    except Exception:
        return 0

# =====================
# UI (estructura de la versión funcional)
# =====================
st.title("Eureka — ANLA · Asistente ciudadano")
st.caption("Chat RAG con fuentes verificables y respuestas apropiadamente específicas.")

# ---- Sidebar: Conexión a Ollama ----
with st.sidebar:
    st.subheader("Conexión a Ollama")
    url_param = _get_query_param("ollama")
    if url_param and "ollama_input" not in st.session_state:
        st.session_state["ollama_input"] = url_param

    default_text = st.session_state.get("ollama_input", os.environ.get("OLLAMA_HOST", "").strip())
    ollama_input = st.text_input(
        "URL pública (ngrok/Cloudflare)",
        value=default_text,
        placeholder="https://xxxx.ngrok-free.app",
        help="Ej: https://6682052ab53b.ngrok-free.app",
        key="ollama_input",
    )
    col1, col2 = st.columns(2)
    with col1:
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
    with col2:
        if st.button("Probar /api/tags", use_container_width=True):
            try:
                candidate = _normalize_base_url(ollama_input)
                ok, detail = _health_check_ollama(candidate)
                st.info(f"Resultado: {'OK' if ok else 'FALLO'} • {detail}")
            except Exception as e:
                st.error(f"Error: {e}")

    if "ollama_base" in st.session_state:
        st.caption(f"Usando: `{st.session_state['ollama_base']}`")

# ---- Sin conexión: detener ----
if "ollama_base" not in st.session_state:
    st.info("💡 Pega la URL pública de tu túnel (ngrok/Cloudflare) y pulsa **Conectar a Ollama**.")
    st.stop()

# ---- Cargar componentes ----
try:
    embeddings, db, llm_extract, llm_eureka_stream = cargar_componentes(st.session_state["ollama_base"])
except Exception as e:
    st.error(f"⌐ No se pudo conectar con Ollama: {e}")
    st.stop()

extractor_chain, eureka_stream_chain, extractor_pt, eureka_pt = construir_cadenas(llm_extract, llm_eureka_stream)

indice_docs = contar_indice(db)
if indice_docs == 0:
    st.warning("No encuentro documentos en el índice (Chroma). Carga/adjunta el `chroma_db` o re-indexa antes de usar el chat.")

# ---- Historial ----
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Chat ----
user_q = st.chat_input("Escribe tu pregunta…")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # === Gating por intención (de la versión funcional) ===
    intent = clasificar_intencion(user_q)
    if intent in ("saludo", "charla", "indeterminado", "vacio"):
        sugerencias = (
            "¿Sobre qué tema ambiental te gustaría saber?\n\n"
            "Ejemplos:\n"
            "- ¿Qué es la licencia ambiental y cuándo se requiere?\n"
            "- ¿Cómo consultar el estado de un expediente en la ANLA?\n"
            "- ¿Qué pasos siguen para una Evaluación de Impacto Ambiental (EIA)?"
        )
        respuesta_breve = "¡Hola! 👋 Estoy listo para ayudarte sobre licenciamiento y trámites ambientales.\n\n" + sugerencias
        with st.chat_message("assistant"):
            st.markdown(respuesta_breve)
        st.session_state.messages.append({"role": "assistant", "content": respuesta_breve})
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base, extrayendo y explicando en lenguaje claro…"):
            try:
                # PROCESO RAG (de la versión funcional + filtros mejorados)
                retriever, params = crear_retriever(db, user_q)
                docs_raw = retriever.invoke(user_q)
                es_esp = es_pregunta_especifica(user_q)
                docs = filtrar_documentos_por_relevancia(docs_raw, user_q, es_esp)

                if not docs:
                    no_docs_reason = "Índice vacío (0 docs)" if indice_docs == 0 else \
                                     "Sin resultados (consulta fuera de dominio o parámetros muy estrictos)"
                    st.info(f"**No encontré documentos relevantes.**\n\n*Motivo:* {no_docs_reason}")
                    st.stop()

                contexto = limitar_contexto(docs, MAX_CONTEXT_CHARS)

                # Paso 1: respuesta técnica
                extractor_kwargs = _build_kwargs_for_prompt(
                    extractor_pt,
                    context=contexto,
                    question=user_q,
                )
                resp_tecnica = extractor_chain.invoke(extractor_kwargs)

                # Paso 2: explicación en lenguaje claro — STREAMING
                eureka_kwargs = _build_kwargs_for_prompt(
                    eureka_pt,
                    respuesta_tecnica=resp_tecnica,
                    question=user_q,
                )
                contenedor = st.empty()
                acumulado = ""
                for chunk in eureka_stream_chain.stream(eureka_kwargs):
                    acumulado += chunk
                    contenedor.markdown(acumulado)
                respuesta_final = acumulado

                # Fuentes
                fuentes = sorted({_safe_get_source(d) for d in docs if _safe_get_source(d) != "Fuente no encontrada"})
                if fuentes and "No he encontrado información" not in respuesta_final:
                    respuesta_final += "\n\n---\n**Fuentes consultadas:**\n" + "\n".join(f"- {u}" for u in fuentes)
                    contenedor.markdown(respuesta_final)

                st.session_state.messages.append({"role": "assistant", "content": respuesta_final})

                # Debug opcional
                with st.expander("Ver información de depuración"):
                    st.write(f"**Tipo de pregunta:** {'específica' if es_esp else 'general'}")
                    st.write(f"**Parámetros MMR:** {params}")
                    st.write(f"**Documentos recuperados:** {len(docs)}")

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")