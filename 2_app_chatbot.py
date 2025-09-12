# 2_app_chatbot.py — RAG con MMR + límite de contexto + streaming saneado + logs + ngrok (sin secrets)
# Citas en texto con superíndices (¹,²,³…), sin listas duplicadas de “Fuentes”.
# Responde en términos GENERALES salvo que el usuario nombre un proyecto/caso específico.

# --- PARCHE PARA SQLITE3 EN STREAMLIT CLOUD ---
__import__("pysqlite3")
import sys as _sys
_sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
# --- FIN DEL PARCHE ---

import os
import re
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, Iterable, Tuple, List

import streamlit as st
import requests

from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ===== Prompts =====
# Si tu prompts.py trae los prompts RIGHTS, se usan; si no, fallback al estándar.
try:
    from prompts import (
        EUREKA_PROMPT,
        EXTRACTOR_PROMPT,
        EUREKA_PROMPT_RIGHTS,
        EXTRACTOR_PROMPT_RIGHTS,
    )
except Exception:
    from prompts import EUREKA_PROMPT, EXTRACTOR_PROMPT  # type: ignore
    EUREKA_PROMPT_RIGHTS = EUREKA_PROMPT
    EXTRACTOR_PROMPT_RIGHTS = EXTRACTOR_PROMPT

# =====================
# Configuración general
# =====================
DIRECTORIO_CHROMA_DB = os.environ.get("CHROMA_DB_DIR", "chroma_db")
MODELO_EMBEDDING = os.environ.get("EMBED_MODEL", "nomic-embed-text")
MODELO_LLM = os.environ.get("LLM_MODEL", "llama3.2")

# Recuperación (MMR)
K_GENERAL = int(os.environ.get("K_GENERAL", 3))
K_ESPECIFICA = int(os.environ.get("K_ESPECIFICA", 5))
FETCH_K = int(os.environ.get("FETCH_K", 20))  # candidatos antes de MMR
MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", 0.5))  # 0=diversidad, 1=similitud

# Contexto
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", 12000))

# Logs
LOG_DB_PATH = os.environ.get("LOG_DB_PATH", "logs.db")

# Depuración opcional
DEBUG_MODE = os.environ.get("DEBUG", "0") == "1"

st.set_page_config(page_title="Eureka – ANLA (RAG)", page_icon="💬", layout="centered")

# =====================
# Utilidades de RAG
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
    return any(re.search(p, (pregunta or ""), re.IGNORECASE) for p in patrones)

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

def reordenar_docs_para_generales(docs):
    """Prioriza normativa/guías y penaliza jurisprudencia/casos conocidos para consultas generales."""
    def score(doc):
        s = (_safe_get_source(doc) or "").lower()
        val = 0
        if "/normativa/" in s or "guia" in s or "guía" in s or "manual" in s:
            val -= 3
        if "/jurisprudencia/" in s:
            val += 2
        contenido = (doc.page_content or "").lower()
        if any(k in contenido for k in CASE_KEYWORDS_LOWER):
            val += 2
        return val
    return sorted(docs, key=score)

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
    try:
        src = (doc.metadata or {}).get("source")
    except Exception:
        src = None
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
        if "?" not in tl and not _QWORDS_PAT.search(tl) and len(tl.split()) <= 4:
            return "saludo"
    if _SMALLTALK_PAT.search(tl):
        return "charla"
    dom_kw = ["anla","licencia","licenciamiento","ambiental","eia","pma","permiso","resolución","audiencia",
              "sustracción","forestal","vertimiento","ruido","emisión","mina","hidrocarburos","energía","proyecto",
              "evaluación","impacto","autoridad","trámite","expediente"]
    if _QWORDS_PAT.search(tl) or "?" in tl or any(k in tl for k in dom_kw):
        return "consulta"
    return "indeterminado"

# --- Detección DERECHOS/seguridad hídrica ---
_DER_RIGHTS_PAT = re.compile(
    r"(derech|seguridad hídrica|embalse|captaci[oó]n|m3|m³|sequ[ií]a|verano|estiaje|caudal ecol[oó]gico|"
    r"concesi[oó]n de agua|autoridad|audiencia p[uú]blica|participaci[oó]n|consulta previa|prioridad de usos|balance h[ií]drico)",
    re.I,
)
def es_consulta_derechos(texto: str) -> bool:
    return bool(_DER_RIGHTS_PAT.search(texto or ""))

def ampliar_consulta_derechos(q: str) -> str:
    extras = (" concesión de aguas caudal ecológico seguridad hídrica estiaje participación audiencia pública "
              "consulta previa priorización uso doméstico balance hídrico monitoreo umbrales suspensión captación PUEAA")
    return (q or "") + extras

# ======== Conexión Ollama (ngrok/Cloudflare) ========
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

def log_interaction(
    question: str,
    response: str,
    docs,
    scores_map: Dict[str, float] | None = None,
    no_docs_reason: str | None = None,
):
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
# Prompts y CITADO
# =====================
def _ensure_prompt(tpl_or_prompt, vars_if_str: Iterable[str] | None = None):
    if isinstance(tpl_or_prompt, str):
        return PromptTemplate(template=tpl_or_prompt)
    return tpl_or_prompt

def _augment_eureka_for_citations(pt: PromptTemplate) -> PromptTemplate:
    """
    Añade instrucciones para:
    - Citar con superíndices (¹,²,³…) usando {sources_indexed}.
    - NO incluir listas de 'Fuentes/Referencias' en la respuesta (el sistema las agrega).
    - Responder en términos GENERALES; si hay casos en el contexto y el usuario no los pidió,
      menciona como 'Ejemplo (opcional): …¹' en una sola línea o no los menciones.
    """
    base_tmpl = pt.template
    base_vars = list(getattr(pt, "input_variables", []) or [])
    if "sources_indexed" not in base_vars:
        base_vars.append("sources_indexed")
    augmented = base_tmpl + (
        "\n\n---\n"
        "CITADO EN TEXTO (OBLIGATORIO):\n"
        "- Dispones de una lista numerada de **Fuentes** en {sources_indexed}.\n"
        "- Cuando una afirmación esté sustentada, agrega superíndices ¹,²,³ con el número correspondiente.\n"
        "- NO incluyas secciones de 'Fuentes', 'Referencias' ni enlaces; el sistema las agregará.\n"
        "- Si el usuario NO nombra un proyecto/caso, responde en términos GENERALES; "
        "no centres la respuesta en casos particulares.\n"
    )
    return PromptTemplate(input_variables=base_vars, template=augmented)

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
    pick(["sources_indexed", "fuentes_indexadas", "fuentes", "sources"], "sources_indexed")
    return out

def _make_indexed_sources(sources: List[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(sources, start=1))

# ===== Limpieza robusta de bloques “Fuentes/Referencias” que pueda escribir el LLM =====
LLM_SOURCES_HEADER_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(fuentes?(?:\s+consultadas?)?|referencias|bibliograf[íi]a)\s*(?:\*\*|__)?\s*:?\s*$",
    re.IGNORECASE,
)
_BULLET_URL_NOTE_RE = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+|^\s*https?://|^\s*nota\s*:", re.IGNORECASE)
_HR_RE = re.compile(r"^\s*(?:-{3,}|_{3,}|\*{3,})\s*$")

def strip_llm_source_lists(text: str) -> str:
    """
    Remueve cualquier bloque tipo “Fuentes/Referencias/Bibliografía …”
    en cualquier parte del cuerpo (incluye bullets, URLs, 'Nota:', reglas ---).
    """
    lines = text.splitlines()
    out = []
    skipping = False
    for ln in lines:
        if not skipping and LLM_SOURCES_HEADER_RE.match(ln):
            skipping = True
            continue
        if skipping:
            # Dentro del bloque de “Fuentes…”
            if _HR_RE.match(ln) or not ln.strip() or _BULLET_URL_NOTE_RE.search(ln) or LLM_SOURCES_HEADER_RE.match(ln):
                # descarta reglas, líneas en blanco, bullets/URLs/“Nota: …” u otro encabezado de fuentes
                continue
            # línea que ya no parece parte del bloque -> salimos del modo skip
            skipping = False
        # fuera del bloque, conservamos
        out.append(ln)
    return "\n".join(out).strip()

# ===== Enfoque general: anonimización de casos si usuario no los pidió =====
CASE_KEYWORDS = [
    r"puerto bol[ií]var", r"cerrej[oó]n", r"arroyo bruno", r"media luna dos",
    r"la laguna siberia", r"las mercedes"
]
CASE_KEYWORDS_LOWER = [re.sub(r"\\", "", k).lower() for k in CASE_KEYWORDS]
CASE_PATTERN = re.compile("|".join(CASE_KEYWORDS), re.I)

def usuario_pidio_caso(text: str) -> bool:
    return bool(CASE_PATTERN.search(text or ""))

def anonimizar_casos_en_texto(texto: str, user_q: str) -> str:
    if usuario_pidio_caso(user_q):
        return texto
    # 1) Eliminar líneas completas que solo desarrollan un caso específico
    new_lines = []
    for ln in texto.splitlines():
        if CASE_PATTERN.search(ln):
            continue  # omitimos esa línea para mantener enfoque general
        new_lines.append(ln)
    texto = "\n".join(new_lines)
    # 2) Reemplazo residual de nombres propios por “un proyecto similar”
    return CASE_PATTERN.sub("un proyecto similar", texto)

# ===== Refuerzo posterior de citas + fallback determinista =====
_SUPERSCRIPT_RE = re.compile(r"[¹²³⁴⁵⁶⁷⁸⁹]")

def _enforce_citations_with_llm(texto: str, sources_indexed: str, llm: OllamaLLM) -> str:
    prompt = PromptTemplate(
        input_variables=["texto", "sources_indexed"],
        template=(
            "Añade superíndices de citas (¹,²,³…) al TEXTO usando la lista numerada de FUENTES.\n"
            "- Usa el número correspondiente de {sources_indexed}.\n"
            "- Mantén el texto idéntico salvo insertar los superíndices donde corresponda.\n"
            "- Coloca al menos un superíndice por párrafo cuando haya respaldo.\n"
            "- NO agregues listas de fuentes; devuelve solo el TEXTO.\n\n"
            "FUENTES (numeradas):\n{sources_indexed}\n\n"
            "TEXTO:\n{texto}\n\n"
            "DEVUELVE SOLO EL TEXTO ANOTADO."
        ),
    )
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"texto": texto, "sources_indexed": sources_indexed})
    except Exception:
        return texto

def _fallback_minimal_citations(texto: str, num_fuentes: int) -> str:
    if num_fuentes <= 0:
        return texto
    supers = ["¹","²","³","⁴","⁵","⁶","⁷","⁸","⁹"]
    paras = [p for p in texto.split("\n\n") if p.strip()]
    nuevos = []
    for i, p in enumerate(paras):
        if _SUPERSCRIPT_RE.search(p):
            nuevos.append(p); continue
        idx = min(i, num_fuentes, len(supers)) - 1
        if idx < 0: idx = 0
        nuevos.append(p + " " + supers[idx])
    return "\n\n".join(nuevos)

def ensure_superscripts(texto: str, sources_indexed: str, llm_for_fix: OllamaLLM) -> str:
    if _SUPERSCRIPT_RE.search(texto):
        return texto
    anotado = _enforce_citations_with_llm(texto, sources_indexed, llm_for_fix)
    if _SUPERSCRIPT_RE.search(anotado):
        return anotado
    num_fuentes = sum(1 for ln in sources_indexed.splitlines() if ln.strip())
    return _fallback_minimal_citations(texto, num_fuentes)

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
    # Prompts estándar
    extractor_pt = _ensure_prompt(EXTRACTOR_PROMPT)
    eureka_pt = _ensure_prompt(EUREKA_PROMPT)
    eureka_pt_aug = _augment_eureka_for_citations(eureka_pt)

    extractor = extractor_pt | llm_extract | StrOutputParser()
    eureka_stream_chain = eureka_pt_aug | llm_eureka_stream | StrOutputParser()

    # Prompts con enfoque de derechos
    rights_extractor_pt = _ensure_prompt(EXTRACTOR_PROMPT_RIGHTS)
    rights_eureka_pt = _ensure_prompt(EUREKA_PROMPT_RIGHTS)
    rights_eureka_pt_aug = _augment_eureka_for_citations(rights_eureka_pt)

    extractor_rights = rights_extractor_pt | llm_extract | StrOutputParser()
    eureka_rights_stream = rights_eureka_pt_aug | llm_eureka_stream | StrOutputParser()

    return (
        extractor, eureka_stream_chain, extractor_pt, eureka_pt_aug,
        extractor_rights, eureka_rights_stream, rights_extractor_pt, rights_eureka_pt_aug,
    )

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
st.caption("Chat RAG con fuentes verificables. MMR, streaming, auditoría y citas en texto.")
init_logging_db()

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

# ---- Sin conexión, detener ----
if "ollama_base" not in st.session_state:
    st.info("💡 Pega la URL pública de tu túnel (ngrok/Cloudflare) y pulsa **Conectar a Ollama**.")
    st.stop()

# ---- Cargar componentes ----
try:
    embeddings, db, llm_extract, llm_eureka_stream = cargar_componentes(st.session_state["ollama_base"])
except Exception as e:
    st.error(f"❌ No se pudo conectar con Ollama: {e}")
    st.stop()

(
    extractor_chain, eureka_stream_chain, extractor_pt, eureka_pt_aug,
    extractor_chain_rights, eureka_stream_chain_rights, extractor_pt_rights, eureka_pt_rights_aug,
) = construir_cadenas(llm_extract, llm_eureka_stream)

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

    # Gating por intención
    intent = clasificar_intencion(user_q)
    if intent in ("saludo", "charla", "indeterminado", "vacio"):
        sugerencias = (
            "¿Sobre qué tema ambiental te gustaría saber?\n\n"
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
                # Modo derechos + expansión
                modo_derechos = es_consulta_derechos(user_q)
                consulta_busqueda = ampliar_consulta_derechos(user_q) if modo_derechos else user_q

                retriever, params = crear_retriever(db, user_q)
                docs_raw = retriever.invoke(consulta_busqueda)
                es_esp = es_pregunta_especifica(user_q)
                docs = filtrar_documentos_por_relevancia(docs_raw, user_q, es_esp)
                if not es_esp:
                    docs = reordenar_docs_para_generales(docs)

                if not docs:
                    no_docs_reason = "Índice vacío (0 docs)" if indice_docs == 0 else "Sin resultados (consulta fuera de dominio o parámetros muy estrictos)"
                    st.info(f"**No encontré documentos relevantes.**\n\n*Motivo:* {no_docs_reason}\n\n*Sugerencias:* verifica que el índice `chroma_db` esté disponible y que tu pregunta esté dentro del dominio.")
                    log_interaction(user_q, response="", docs=[], scores_map={}, no_docs_reason=no_docs_reason)
                    st.stop()

                # Contexto
                contexto = limitar_contexto(docs, MAX_CONTEXT_CHARS)

                # Fuentes (orden estable)
                fuentes_list = sorted({_safe_get_source(d) for d in docs if _safe_get_source(d) != "Fuente no encontrada"})
                sources_indexed_text = _make_indexed_sources(fuentes_list)

                # Selección de cadenas
                if modo_derechos:
                    extractor_sel = extractor_chain_rights
                    eureka_sel = eureka_stream_chain_rights
                    extractor_pt_sel = extractor_pt_rights
                    eureka_pt_sel = eureka_pt_rights_aug
                else:
                    extractor_sel = extractor_chain
                    eureka_sel = eureka_stream_chain
                    extractor_pt_sel = extractor_pt
                    eureka_pt_sel = eureka_pt_aug

                # Paso 1: respuesta técnica
                extractor_kwargs = _build_kwargs_for_prompt(extractor_pt_sel, context=contexto, question=user_q)
                resp_tecnica = extractor_sel.invoke(extractor_kwargs)

                # Paso 2: explicación — STREAMING saneado en vivo
                eureka_kwargs = _build_kwargs_for_prompt(
                    eureka_pt_sel,
                    respuesta_tecnica=resp_tecnica,
                    question=user_q,
                    sources_indexed=sources_indexed_text,
                )
                contenedor = st.empty()
                acumulado = ""
                for chunk in eureka_sel.stream(eureka_kwargs):
                    acumulado += chunk
                    # limpiar y mantener enfoque general EN VIVO
                    parcial = strip_llm_source_lists(acumulado)
                    parcial = anonimizar_casos_en_texto(parcial, user_q)
                    contenedor.markdown(parcial)

                # Limpieza final + enfoque general
                respuesta_final = strip_llm_source_lists(acumulado)
                respuesta_final = anonimizar_casos_en_texto(respuesta_final, user_q)

                # Refuerzo de superíndices
                respuesta_final_citada = ensure_superscripts(respuesta_final, sources_indexed_text, llm_extract)
                if respuesta_final_citada != respuesta_final:
                    respuesta_final = respuesta_final_citada
                    contenedor.markdown(respuesta_final)

                # Lista final única de fuentes
                if fuentes_list and "No he encontrado información" not in respuesta_final:
                    fuentes_md = "\n".join(f"{i+1}. {u}" for i, u in enumerate(fuentes_list))
                    respuesta_final = respuesta_final + "\n\n---\n**Fuentes consultadas:**\n" + fuentes_md
                    contenedor.markdown(respuesta_final)

                st.session_state.messages.append({"role": "assistant", "content": respuesta_final})

                # Depuración opcional
                if DEBUG_MODE:
                    with st.expander("Ver información de depuración"):
                        st.write(f"**Intención:** {intent}")
                        st.write(f"**Modo derechos:** {modo_derechos}")
                        st.write(f"**Parámetros MMR:** {params}")
                        st.write(f"**Documentos recuperados:** {len(docs)}")

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
