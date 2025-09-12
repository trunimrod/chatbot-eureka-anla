# 2_app_chatbot.py (Versión Limpia - Con parche SQLite3 robusto)

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

from prompts import EUREKA_PROMPT, EXTRACTOR_PROMPT

# =====================
# Configuración
# =====================
DIRECTORIO_CHROMA_DB = os.environ.get("CHROMA_DB_DIR", "chroma_db")
MODELO_EMBEDDING = os.environ.get("EMBED_MODEL", "nomic-embed-text")
MODELO_LLM = os.environ.get("LLM_MODEL", "llama3.2")

# Parámetros MMR simples
K_DOCUMENTOS = 4
FETCH_K = 20
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
    # palabras clave del dominio
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
        return PromptTemplate(template=tpl_or_prompt)
    return tpl_or_prompt

def _build_kwargs_for_prompt(prompt: PromptTemplate, **values: Any) -> Dict[str, Any]:
    """Mapea variables de prompt automáticamente"""
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

def crear_retriever(db: Chroma):
    """Retriever simple con MMR"""
    params = {"k": K_DOCUMENTOS, "fetch_k": FETCH_K, "lambda_mult": MMR_LAMBDA}
    retriever = db.as_retriever(search_type="mmr", search_kwargs=params)
    return retriever, params

def contar_indice(db: Chroma) -> int:
    """Cuenta documentos en el índice"""
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
st.title("Eureka — ANLA · Asistente ciudadano")
st.caption("Te ayudo a entender tus derechos y deberes ambientales.")

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
        help="Ejemplo: https://6682052ab53b.ngrok-free.app",
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
        
    st.divider()
    st.subheader("Ejemplos de uso")
    st.write("**Preguntas generales:**")
    st.write("• ¿Qué derechos tengo si un proyecto me afecta?")
    st.write("• ¿Cómo participar en decisiones ambientales?")
    st.write("• ¿Qué compensaciones puede recibir una comunidad?")

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
    st.warning("No encuentro documentos en el índice (Chroma). Verifica que la carpeta `chroma_db` esté disponible.")

# ---- Historial ----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hola, soy Eureka. Te ayudo a entender tus derechos ambientales y cómo participar en las decisiones que te pueden afectar. ¿En qué puedo ayudarte hoy?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Chat ----
user_q = st.chat_input("Escribe tu pregunta…")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # === Filtro de intención para evitar RAG en saludos ===
    intent = clasificar_intencion(user_q)
    if intent in ("saludo", "charla", "indeterminado", "vacio"):
        sugerencias = (
            "¿Sobre qué tema ambiental te gustaría saber?\n\n"
            "**Ejemplos:**\n"
            "• ¿Qué es la licencia ambiental y cuándo se requiere?\n"
            "• ¿Cómo consultar el estado de un expediente en la ANLA?\n"
            "• ¿Qué pasos siguen para una Evaluación de Impacto Ambiental?"
        )
        respuesta_breve = "¡Hola! 👋 Estoy listo para ayudarte sobre licenciamiento y trámites ambientales.\n\n" + sugerencias
        with st.chat_message("assistant"):
            st.markdown(respuesta_breve)
        st.session_state.messages.append({"role": "assistant", "content": respuesta_breve})
        st.stop()

    # === RAG Principal ===
    with st.chat_message("assistant"):
        with st.spinner("Buscando información y preparando respuesta…"):
            try:
                # Búsqueda de documentos
                retriever, params = crear_retriever(db)
                docs = retriever.invoke(user_q)

                if not docs:
                    st.info("No encontré información relevante sobre tu consulta. ¿Podrías reformular la pregunta?")
                    st.stop()

                # Crear contexto
                contexto = limitar_contexto(docs, MAX_CONTEXT_CHARS)

                # Paso 1: Extracción técnica
                extractor_kwargs = _build_kwargs_for_prompt(
                    extractor_pt,
                    context=contexto,
                    question=user_q,
                )
                resp_tecnica = extractor_chain.invoke(extractor_kwargs)

                # Paso 2: Traducción a lenguaje claro con STREAMING
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

                # Agregar fuentes
                fuentes = sorted({_safe_get_source(d) for d in docs if _safe_get_source(d) != "Fuente no encontrada"})
                if fuentes and "No he encontrado información" not in respuesta_final:
                    respuesta_final += "\n\n---\n**Fuentes consultadas:**\n" + "\n".join(f"• {u}" for u in fuentes)
                    contenedor.markdown(respuesta_final)

                st.session_state.messages.append({"role": "assistant", "content": respuesta_final})

                # Información técnica con debugging administrativo
                with st.expander("🔧 Vista de Administrador - Análisis de consulta"):
                    st.subheader("📊 Métricas de búsqueda")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Documentos encontrados", len(docs))
                    with col2:
                        st.metric("Caracteres de contexto", len(contexto))
                    with col3:
                        st.metric("Parámetros MMR", f"k={params['k']}, λ={params['lambda_mult']}")
                    
                    st.subheader("📋 Documentos recuperados")
                    for i, doc in enumerate(docs, 1):
                        with st.container():
                            fuente = _safe_get_source(doc)
                            st.write(f"**📄 Documento {i}:**")
                            st.write(f"**Fuente:** `{fuente}`")
                            
                            # Mostrar fragmento del contenido
                            contenido = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                            st.text_area(
                                f"Contenido (primeros 500 caracteres):",
                                contenido,
                                height=100,
                                key=f"doc_content_{i}"
                            )
                            
                            # Metadata adicional
                            if doc.metadata:
                                metadata_clean = {k: v for k, v in doc.metadata.items() if k != 'source'}
                                if metadata_clean:
                                    st.write(f"**Metadata:** {metadata_clean}")
                            st.divider()
                    
                    st.subheader("🔄 Procesamiento paso a paso")
                    
                    # Paso 1: Contexto completo
                    st.write("**1️⃣ Contexto enviado al Extractor:**")
                    st.text_area("Contexto completo", contexto, height=150, key="contexto_completo")
                    
                    # Paso 2: Respuesta técnica
                    st.write("**2️⃣ Respuesta técnica del Extractor:**")
                    st.text_area("Extracción técnica", resp_tecnica, height=100, key="respuesta_tecnica")
                    
                    # Paso 3: Variables del prompt Eureka
                    st.write("**3️⃣ Variables enviadas a Eureka:**")
                    st.json({
                        "original_question": user_q,
                        "technical_summary": resp_tecnica[:200] + "..." if len(resp_tecnica) > 200 else resp_tecnica
                    })
                    
                    # Análisis de contenido específico
                    st.subheader("🔍 Análisis de especificidad")
                    
                    # Buscar nombres propios en documentos
                    import re
                    nombres_propios_encontrados = set()
                    patron_nombres = re.compile(r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)*\b')
                    
                    for doc in docs:
                        nombres_en_doc = patron_nombres.findall(doc.page_content)
                        nombres_propios_encontrados.update(nombres_en_doc)
                    
                    # Filtrar nombres comunes que no son específicos
                    nombres_comunes = {'Ley', 'Artículo', 'Constitución', 'Estado', 'República', 'Colombia', 'Nacional', 'Ministerio', 'ANLA', 'Autoridad'}
                    nombres_especificos = [n for n in nombres_propios_encontrados if n not in nombres_comunes]
                    
                    if nombres_especificos:
                        st.warning(f"⚠️ **Nombres específicos detectados en documentos:** {', '.join(nombres_especificos[:10])}")
                        st.write("*Esto podría explicar por qué aparecen referencias específicas en la respuesta*")
                    else:
                        st.success("✅ **No se detectaron nombres específicos problemáticos**")
                    
                    # Análisis de la pregunta
                    st.subheader("❓ Análisis de la pregunta")
                    st.write(f"**Pregunta original:** `{user_q}`")
                    st.write(f"**Clasificada como:** `{intent}`")
                    
                    # Buscar términos generales vs específicos en la pregunta
                    terminos_generales = ['embalse', 'proyecto', 'comunidad', 'compensación', 'empresa', 'municipio']
                    terminos_en_pregunta = [t for t in terminos_generales if t.lower() in user_q.lower()]
                    
                    if terminos_en_pregunta:
                        st.info(f"📝 **Términos generales detectados:** {', '.join(terminos_en_pregunta)}")
                        st.write("*La respuesta debería mantenerse general*")
                    
                    # Sugerencias de mejora
                    st.subheader("💡 Sugerencias de mejora")
                    if nombres_especificos and terminos_en_pregunta:
                        st.write("**Problema detectado:** La pregunta es general pero los documentos contienen información específica")
                        st.write("**Recomendación:** Los prompts deberían filtrar mejor la información específica")
                    elif not terminos_en_pregunta:
                        st.write("**Observación:** La pregunta no contiene términos que requieran filtrado especial")
                    else:
                        st.write("**Estado:** Los documentos parecen apropiados para una respuesta general")

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
                st.write("Intenta con otra pregunta o verifica la conexión con Ollama.")