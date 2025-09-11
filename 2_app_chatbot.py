# 2_app_chatbot.py — robusto: MMR + límite de contexto + streaming + logs + ngrok sin secrets
# Detecta intención (saludo/charla) para no disparar RAG en saludos.
# Mapea variables de prompt (context/question/original_question, technical_summary, etc.)

# --- PARCHE PARA SQLITE3 EN STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys as _sys
_sys.modules['sqlite3'] = _sys.modules.pop('pysqlite3')
# --- FIN DEL PARCHE ---

import os
import re
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, Iterable, Tuple

import streamlit as st
import requests

from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Asegúrate de tener este archivo junto a 2_app_chatbot.py
from prompts import EUREKA_PROMPT, EXTRACTOR_PROMPT

# =====================
# Configuración general
# =====================
DIRECTORIO_CHROMA_DB = os.environ.get("CHROMA_DB_DIR", "chroma_db")
MODELO_EMBEDDING = os.environ.get("EMBED_MODEL", "nomic-embed-text")
MODELO_LLM = os.environ.get("LLM_MODEL", "llama3.2")

# Recuperación (MMR)
K_GENERAL = int(os.environ.get("K_GENERAL", 3))
K_ESPECIFICA = int(os.environ.get("K_ESPECIFICA", 5))
FETCH_K = int(os.environ.get("FETCH_K", 20))      # candidatos antes de MMR
MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", 0.5))  # 0=diversidad, 1=similitud

# Contexto
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", 12000))

# Logs
LOG_DB_PATH = os.environ.get("LOG_DB_PATH", "logs.db")

st.set_page_config(page_title="Eureka – ANLA (RAG)", page_icon="💬", layout="centered")

# =====================
# Utilidades
# =====================
def es_pregunta_especifica(pregunta: str) -> bool:
    patrones = [
        r"\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)*\b",
        r"\bembalse\s+del?\s+\w+",
        r"\bproyecto\s+\w+",
        r"\bempresa\s+\w+",
        r"\b\w+\s+S\.?A\.?S?\.?",
        r"\bmunicipio\s+de\s+\w+",
        r"\bdepartamento\s+del?\s+\w+",
    ]
    return any(re.search(p, pregunta, re.IGNORECASE) for p in patrones)

def ajustar_parametros_busqueda(pregunta: str) -> dict:
    k = K_ESPECIFICA if es_pregunta_especifica(pregunta) else K_GENERAL
    return {"k": k, "fetch_k": FETCH_K, "lambda_mult": MMR_LAMBDA}

def filtrar_documentos_por_relevancia(documentos, pregunta: str, es_especifica: bool):
    if es_especifica:
        return documentos
    docs_filtrados = []
    patron_np = re.compile(r"\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)*\b")
    for doc in documentos:
        contenido = doc.page_content or ""
        if len(patron_np.findall(contenido)) <= 3:
            docs_filtrados.append(doc)
    return docs_filtrados or documentos[:2]

def limitar_contexto(documentos, max_chars: int) -> str:
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
    src = (doc.metadata or {}).get("source")
    return src or "Fuente no encontrada"

# ======== Clasificador de intención (no LLM) ========
_GREET_WORDS = ["hola","holi","hello","hey","buenas","buenos días","buenas tardes","buenas noches"]
_SMALLTALK_PAT = re.compile(r"(cómo estás|que tal|qué tal|gracias|de nada|ok|vale|bkn|listo|perfecto)", re.I)
_QWORDS_PAT = re.compile(r"\b(qué|que|cómo|como|cuál|cual|cuándo|cuando|dónde|donde|por qué|porque|quién|quien|cuánto|cuanto)\b", re.I)

def clasificar_intencion(texto: str) -> str:
    t = (texto or "").strip()
    tl = t.lower()
    if not tl:
        return "vacio"
    if any(tl.startswith(w) or w in tl for w in _GREET_WORDS):
        # saludos breves sin signo de interrogación ni palabra interrogativa
        if "?" not in tl and not _QWORDS_PAT.search(tl) and len(tl.split()) <= 4:
            return "saludo"
    if _SMALLTALK_PAT.search(tl):
        return "charla"
    # palabras clave del dominio ANLA
    dom_kw = ["anla","licencia","licenciamiento","ambiental","eia","pma","permiso","resolución","audiencia",
              "sustracción","forestal","vertimiento","ruido","emisión","mina","hidrocarburos","energía","proyecto",
              "evaluación","impacto","autoridad","trámite","expediente"]
    if _QWORDS_PAT.search(tl) or "?" in tl or any(k in tl for k in dom_kw):
        return "consulta"
    return "indeterminado"

# ======== Soporte ngrok/Cloudflare SIN secrets ========
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

def _health_check_ollama(base: str, timeout: float = 5.0) -> Tuple[bool, str]:
    try:
        r = requests.get(f"{base}/api/tags", timeout=timeout)
        r.raise_for_status()
        return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

# =====================
# Logging (SQLite)
# =====================
def init_logging_db(path: str = LOG_DB_PATH):
    try:
        con = sqlite3.connect(path); cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                question TEXT,
                response_preview TEXT,
                num_docs INTEGER,
                doc_sources TEXT,
                doc_ids TEXT,
                scores TEXT,
                no_docs_reason TEXT
            )
        """)
        con.commit(); con.close()
    except Exception as e:
        st.warning(f"No se pudo inicializar la base de logs: {e}")

def log_interaction(question: str, response: str, docs, scores_map=None, no_docs_reason: str | None = None):
    try:
        con = sqlite3.connect(LOG_DB_PATH); cur = con.cursor()
        sources = [_safe_get_source(d) for d in (docs or [])]
        doc_ids = [str((d.metadata or {}).get("id", "")) for d in (docs or [])]
        cur.execute("""
            INSERT INTO interactions(timestamp, question, response_preview, num_docs, doc_sources, doc_ids, scores, no_docs_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            question,
            (response or "")[:1000],
            len(docs or []),
            json.dumps(sources, ensure_ascii=False),
            json.dumps(doc_ids, ensure_ascii=False),
            json.dumps(scores_map or {}, ensure_ascii=False),
            no_docs_reason,
        ))
        con.commit(); con.close()
    except Exception as e:
        st.warning(f"No se pudo guardar el log: {e}")

# =====================
# Helpers de prompts
# =====================
def _ensure_prompt(tpl_or_prompt, vars_if_str: Iterable[str] | None = None):
    if isinstance(tpl_or_prompt, str):
        return PromptTemplate(template=tpl_or_prompt)
    return tpl_or_prompt

def _build_kwargs_for_prompt(prompt: PromptTemplate, **values: Any) -> Dict[str, Any]:
    """
    Mapea dinámicamente variables esperadas por el prompt a valores disponibles.
    Soporta:
      - context / contexto / context_text
      - question / pregunta / query / user_question / original_question
      - respuesta_tecnica / technical_answer / answer / summary / respuesta / technical_summary
    """
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
# Carga de componentes IA (cacheados por base_url)
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
    params = ajustar_parametros_busqueda(pregunta)
    retriever = db.as_retriever(search_type="mmr", search_kwargs=params)
    return retriever, params

def contar_indice(db: Chroma) -> int:
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
# UI
# =====================
st.title("Eureka – ANLA · Asistente ciudadano")
st.caption("Chat RAG con fuentes verificables. Ahora con MMR, streaming y auditoría.")
init_logging_db()

# ---- Sidebar: Conexión a Ollama (sin auto-health-check) ----
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

# ---- Estado inicial (sin conexión): instrucción y detener el resto ----
if "ollama_base" not in st.session_state:
    st.info(
        "💡 Para empezar, pega en la **barra lateral** la URL pública de tu túnel "
        "(ngrok/Cloudflare) y pulsa **Conectar a Ollama**.\n\n"
        "Ejemplo: `https://6682052ab53b.ngrok-free.app`"
    )
    st.stop()

# ---- Cargar componentes solo después de conectar ----
try:
    embeddings, db, llm_extract, llm_eureka_stream = cargar_componentes(st.session_state["ollama_base"])
except Exception as e:
    st.error(f"❌ No se pudo conectar con Ollama: {e}")
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

    # === Gating por intención: evita RAG con saludos/charla ===
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
        log_interaction(user_q, respuesta_breve, docs=[], scores_map={}, no_docs_reason=f"intencion:{intent}")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base, extrayendo y explicando en lenguaje claro…"):
            try:
                retriever, params = crear_retriever(db, user_q)
                docs_raw = retriever.invoke(user_q)
                es_esp = es_pregunta_especifica(user_q)
                docs = filtrar_documentos_por_relevancia(docs_raw, user_q, es_esp)

                if not docs:
                    no_docs_reason = "Índice vacío (0 docs)" if indice_docs == 0 else \
                                     "Sin resultados (consulta fuera de dominio o parámetros muy estrictos)"
                    st.info(
                        f"""**No encontré documentos relevantes.**

*Motivo:* {no_docs_reason}

*Sugerencias:* verifica que el índice `chroma_db` esté disponible y que tu pregunta esté dentro del dominio."""
                    )
                    log_interaction(user_q, response="", docs=[], scores_map={}, no_docs_reason=no_docs_reason)
                    st.stop()

                contexto = limitar_contexto(docs, MAX_CONTEXT_CHARS)

                # Paso 1: respuesta técnica — adapta variables del EXTRACTOR
                extractor_kwargs = _build_kwargs_for_prompt(
                    extractor_pt,
                    context=contexto,    # 'context'/'contexto'
                    question=user_q,     # 'question'/'pregunta'/'original_question'
                )
                resp_tecnica = extractor_chain.invoke(extractor_kwargs)

                # Paso 2: explicación en lenguaje claro — STREAMING — adapta variables del EUREKA
                eureka_kwargs = _build_kwargs_for_prompt(
                    eureka_pt,
                    respuesta_tecnica=resp_tecnica,  # 'technical_summary'/'respuesta_tecnica'
                    question=user_q,                 # 'original_question'/'question'
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

                # Debug
                with st.expander("Ver información de depuración"):
                    st.write(f"**Tipo de pregunta:** {'específica' if es_esp else 'general'}")
                    st.write(f"**Parámetros MMR:** {params}")
                    st.write(f"**Documentos recuperados:** {len(docs)}")
                    try:
                        st.json([d.dict() for d in docs])
                    except Exception:
                        st.write("No fue posible mostrar detalle de docs.")

                # Logs con scores
                scores_map: Dict[str, float] = {}
                try:
                    sim_pairs = db.similarity_search_with_score(user_q, k=params.get("k", 3))
                    for d, s in sim_pairs:
                        scores_map[_safe_get_source(d)] = float(s)
                except Exception:
                    pass
                log_interaction(user_q, respuesta_final, docs, scores_map=scores_map)

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
