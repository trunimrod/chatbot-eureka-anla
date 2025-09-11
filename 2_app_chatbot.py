# 2_app_chatbot.py — con mejoras: MMR + límite de contexto + streaming + logs SQLite

# --- PARCHE PARA SQLITE3 EN STREAMLIT CLOUD ---
# Carga una versión compatible de SQLite3 antes de que chromadb la necesite.
__import__('pysqlite3')
import sys as _sys
_sys.modules['sqlite3'] = _sys.modules.pop('pysqlite3')
# --- FIN DEL PARCHE ---

import os
import re
import json
import time
import sqlite3
from datetime import datetime

import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate  # puede recibirse str o PromptTemplate desde prompts.py
from langchain_core.output_parsers import StrOutputParser

# Prompts existentes
from prompts import EUREKA_PROMPT, EXTRACTOR_PROMPT

# =====================
# Configuración general
# =====================
DIRECTORIO_CHROMA_DB = os.environ.get("CHROMA_DB_DIR", "chroma_db")
MODELO_EMBEDDING = os.environ.get("EMBED_MODEL", "nomic-embed-text")
MODELO_LLM = os.environ.get("LLM_MODEL", "llama3.2")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Recuperación (MMR)
K_GENERAL = int(os.environ.get("K_GENERAL", 3))
K_ESPECIFICA = int(os.environ.get("K_ESPECIFICA", 5))
FETCH_K = int(os.environ.get("FETCH_K", 20))  # candidatos antes de MMR
MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", 0.5))  # 0=diversidad, 1=similitud

# Contexto: limitar tamaño total concatenado
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", 12000))

# Logs
LOG_DB_PATH = os.environ.get("LOG_DB_PATH", "logs.db")

st.set_page_config(page_title="Eureka – ANLA (RAG)", page_icon="💬", layout="centered")

# =====================
# Utilidades
# =====================

def es_pregunta_especifica(pregunta: str) -> bool:
    """Heurística simple para detectar especificidad (proyecto/empresa/lugar)."""
    patrones_especificos = [
        r"\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)*\b",  # nombres propios
        r"\bembalse\s+del?\s+\w+",
        r"\bproyecto\s+\w+",
        r"\bempresa\s+\w+",
        r"\b\w+\s+S\.?A\.?S?\.?",  # razón social
        r"\bmunicipio\s+de\s+\w+",
        r"\bdepartamento\s+del?\s+\w+",
    ]
    return any(re.search(p, pregunta, re.IGNORECASE) for p in patrones_especificos)


def ajustar_parametros_busqueda(pregunta: str) -> dict:
    """Devuelve kwargs para as_retriever() con MMR y k dinámico."""
    k = K_ESPECIFICA if es_pregunta_especifica(pregunta) else K_GENERAL
    return {
        "k": k,
        "fetch_k": FETCH_K,
        "lambda_mult": MMR_LAMBDA,
    }


def filtrar_documentos_por_relevancia(documentos, pregunta: str, es_especifica: bool):
    """Para preguntas generales, evita docs con demasiados nombres propios (>3)."""
    if es_especifica:
        return documentos
    docs_filtrados = []
    patron_np = re.compile(r"\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)*\b")
    for doc in documentos:
        contenido = doc.page_content or ""
        nombres_propios = len(patron_np.findall(contenido))
        if nombres_propios <= 3:
            docs_filtrados.append(doc)
    if not docs_filtrados:
        # en caso extremo, conserva al menos los 2 primeros
        return documentos[:2]
    return docs_filtrados


def limitar_contexto(documentos, max_chars: int) -> str:
    """Concatena contenidos hasta max_chars para acotar el tamaño del prompt."""
    piezas = []
    total = 0
    for i, d in enumerate(documentos, 1):
        txt = (d.page_content or "").strip()
        header = f"\n\n[DOC {i}]\n"
        chunk = header + txt
        if total + len(chunk) > max_chars:
            # corta el último pedazo si aún no hemos añadido nada
            restante = max_chars - total
            if restante > len(header):
                piezas.append(header + txt[: restante - len(header)])
            break
        piezas.append(chunk)
        total += len(chunk)
    return "".join(piezas).strip()


def _safe_get_source(doc):
    src = (doc.metadata or {}).get("source")
    return src or "Fuente no encontrada"

# =====================
# Logging (SQLite)
# =====================

def init_logging_db(path: str = LOG_DB_PATH):
    try:
        con = sqlite3.connect(path)
        cur = con.cursor()
        cur.execute(
            """
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
            """
        )
        con.commit()
        con.close()
    except Exception as e:
        st.warning(f"No se pudo inicializar la base de logs: {e}")


def log_interaction(question: str, response: str, docs, scores_map=None, no_docs_reason: str | None = None):
    try:
        con = sqlite3.connect(LOG_DB_PATH)
        cur = con.cursor()
        sources = [_safe_get_source(d) for d in (docs or [])]
        doc_ids = [str((d.metadata or {}).get("id", "")) for d in (docs or [])]
        scores_map = scores_map or {}
        cur.execute(
            """
            INSERT INTO interactions(timestamp, question, response_preview, num_docs, doc_sources, doc_ids, scores, no_docs_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                question,
                (response or "")[:1000],
                len(docs or []),
                json.dumps(sources, ensure_ascii=False),
                json.dumps(doc_ids, ensure_ascii=False),
                json.dumps(scores_map, ensure_ascii=False),
                no_docs_reason,
            ),
        )
        con.commit()
        con.close()
    except Exception as e:
        st.warning(f"No se pudo guardar el log: {e}")

# =====================
# Carga de componentes IA
# =====================
@st.cache_resource(show_spinner=False)
def cargar_componentes():
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING, base_url=OLLAMA_HOST)
    db = Chroma(persist_directory=DIRECTORIO_CHROMA_DB, embedding_function=embeddings)
    llm_extract = OllamaLLM(model=MODELO_LLM, temperature=0.2, base_url=OLLAMA_HOST)
    # LLM con streaming habilitado para la etapa EUREKA
    llm_eureka_stream = OllamaLLM(model=MODELO_LLM, temperature=0.2, base_url=OLLAMA_HOST, streaming=True)
    return embeddings, db, llm_extract, llm_eureka_stream


@st.cache_resource(show_spinner=False)
def construir_cadenas(llm_extract: OllamaLLM, llm_eureka_stream: OllamaLLM):
    # Asegura compatibilidad: acepta str o PromptTemplate
    from langchain.prompts import PromptTemplate as _PT

    def _ensure_prompt(tpl_or_prompt, vars_):
        if isinstance(tpl_or_prompt, str):
            return _PT(template=tpl_or_prompt, input_variables=vars_)
        return tpl_or_prompt

    extractor_pt = _ensure_prompt(EXTRACTOR_PROMPT, ["contexto", "pregunta"])
    eureka_pt = _ensure_prompt(EUREKA_PROMPT, ["respuesta_tecnica"])

    extractor = extractor_pt | llm_extract | StrOutputParser()
    eureka_stream_chain = eureka_pt | llm_eureka_stream | StrOutputParser()

    return extractor, eureka_stream_chain


def crear_retriever(db: Chroma, pregunta: str):
    params = ajustar_parametros_busqueda(pregunta)
    retriever = db.as_retriever(search_type="mmr", search_kwargs=params)
    return retriever, params


def contar_indice(db: Chroma) -> int:
    try:
        # Acceso interno a la colección subyacente
        if hasattr(db, "_collection") and db._collection is not None:
            return int(db._collection.count())
    except Exception:
        pass
    # Fallback: intenta una búsqueda con k=1
    try:
        res = db.similarity_search("prueba", k=1)
        return 1 if res else 0
    except Exception:
        return 0


# =====================
# UI y estado
# =====================
st.title("Eureka – ANLA · Asistente ciudadano")
st.caption("Chat RAG con fuentes verificables. Ahora con MMR, streaming y auditoría.")

if "messages" not in st.session_state:
    st.session_state.messages = []

init_logging_db()
embeddings, db, llm_extract, llm_eureka_stream = cargar_componentes()
extractor_chain, eureka_stream_chain = construir_cadenas(llm_extract, llm_eureka_stream)

indice_docs = contar_indice(db)
if indice_docs == 0:
    st.warning("No encuentro documentos en el índice (Chroma). Carga/adjunta el `chroma_db` o re-indexa antes de usar el chat.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("Escribe tu pregunta…")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base, extrayendo y explicando en lenguaje claro…"):
            no_docs_reason = None
            try:
                retriever, params = crear_retriever(db, user_q)
                docs_raw = retriever.invoke(user_q)
                es_esp = es_pregunta_especifica(user_q)
                docs = filtrar_documentos_por_relevancia(docs_raw, user_q, es_esp)

                if not docs:
                    if indice_docs == 0:
                        no_docs_reason = "Índice vacío (0 documentos en Chroma)"
                    else:
                        no_docs_reason = "El retriever no devolvió resultados (consulta fuera de dominio o parámetros muy restrictivos)"
                    st.info(
                        """**No encontré documentos relevantes.**
                        
                        *Motivo:* {motivo}
                        
                        *Sugerencias:* verifica que el índice `chroma_db` esté disponible y que tu pregunta esté relacionada con los temas del repositorio de documentos.
                        """.format(motivo=no_docs_reason)
                    )
                    log_interaction(user_q, response="", docs=[], scores_map={}, no_docs_reason=no_docs_reason)
                    st.stop()

                # Limitar tamaño del contexto
                contexto = limitar_contexto(docs, MAX_CONTEXT_CHARS)

                # Paso 1: respuesta técnica (no se streamea)
                resp_tecnica = extractor_chain.invoke({
                    "contexto": contexto,
                    "pregunta": user_q,
                })

                # Paso 2: explicación en lenguaje claro — STREAMING
                contenedor = st.empty()
                acumulado = ""
                for chunk in eureka_stream_chain.stream({"respuesta_tecnica": resp_tecnica}):
                    acumulado += chunk
                    contenedor.markdown(acumulado)
                respuesta_final = acumulado

                # Fuentes
                fuentes = sorted({
                    _safe_get_source(d) for d in docs if _safe_get_source(d) != "Fuente no encontrada"
                })
                if fuentes and "No he encontrado información" not in respuesta_final:
                    respuesta_final += "\n\n---\n**Fuentes consultadas:**\n" + "\n".join(f"- {u}" for u in fuentes)
                    contenedor.markdown(respuesta_final)

                st.session_state.messages.append({"role": "assistant", "content": respuesta_final})

                # Debug
                with st.expander("Ver información de depuración"):
                    st.write(f"**Tipo de pregunta detectado:** {'específica' if es_esp else 'general'}")
                    st.write(f"**Parámetros de búsqueda (MMR):** {params}")
                    st.write(f"**Documentos recuperados:** {len(docs)}")
                    try:
                        st.json([d.dict() for d in docs])
                    except Exception:
                        st.write("No fue posible mostrar el detalle de los documentos.")

                # Log: además de las fuentes, intenta obtener 'scores' por similitud clásica (para auditoría)
                scores_map = {}
                try:
                    sim_pairs = db.similarity_search_with_score(user_q, k=params.get("k", 3))
                    # mapear por source -> score
                    for d, s in sim_pairs:
                        src = _safe_get_source(d)
                        scores_map[src] = float(s)
                except Exception:
                    pass

                log_interaction(user_q, respuesta_final, docs, scores_map=scores_map, no_docs_reason=None)

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
