# 2_app_chatbot.py (Versión con Vista de Administrador para Depuración)

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
# PROMPTS (AJUSTE FINAL Y MÁS ESTRICTO)
# =====================
EXTRACTOR_PROMPT = """
Eres un experto analista de la ANLA. Tu tarea es extraer la información fáctica y técnica más relevante de los documentos proporcionados para responder a la pregunta del usuario.

**Instrucciones:**
1.  **Enfócate en lo técnico:** Extrae datos, artículos de ley, números de resolución, procedimientos y obligaciones concretas.
2.  **Sé conciso y directo:** No uses lenguaje introductorio. Ve directo al grano.
3.  **Cita la fuente y el nombre del documento:** Si un dato proviene de un documento (ej. `[DOC 1]`), menciónalo junto con su título si está disponible (ej. `[DOC 1, Sentencia T-704 de 2016]`). ESTE PASO ES OBLIGATORIO para cada pieza de información que extraigas.
4.  **Señala el contexto específico:** Es CRUCIAL que si la información se refiere a un caso, proyecto o tipo de programa concreto (ej. 'Mina El Cerrejón', 'Pago por Servicios Ambientales', 'Comunidad Media Luna Dos'), lo menciones explícitamente en la extracción. Ejemplo: "En el caso del proyecto minero El Cerrejón, se estableció el derecho a la consulta previa [DOC 1, Sentencia T-704 de 2016]".
5.  **No interpretes ni converses:** Tu salida debe ser un resumen denso de hechos y datos extraídos.
6.  **Si no hay información:** Si los documentos no contienen información relevante para responder, indica claramente: "No he encontrado información relevante en los documentos proporcionados."

**Documentos:**
---
{context}
---

**Pregunta del usuario:** {question}

**Extracción técnica:**
"""

EUREKA_PROMPT = """
Eres Eureka, un asistente ciudadano de la ANLA. Tu misión es ser 100% fiel a la información que se te proporciona, ayudando a los ciudadanos de forma clara y precisa.

*** REGLA DE ORO INQUEBRANTABLE: TU OBJETIVO PRINCIPAL ES EVITAR LA "ALUCINACIÓN". ***
- **NUNCA, bajo ninguna circunstancia, inventes un número de ley, decreto, sentencia o cualquier tipo de cita.** Esto es un error crítico que desinforma al ciudadano. Si el resumen técnico no te da un número específico, OBLIGATORIAMENTE debes usar expresiones generales como "la normativa ambiental vigente" o "la jurisprudencia ha señalado".
- **CITA DIRECTAMENTE TUS FUENTES DENTRO DEL TEXTO.** Cada afirmación que hagas debe estar respaldada por el resumen técnico. DEBES mencionar la fuente (el título del documento) directamente en la frase. Por ejemplo: **"La Sentencia T-704 de 2016 establece que..."** o **"Según la Declaración de Río sobre el Medio Ambiente..."**. No puedes hacer una afirmación y luego listar las fuentes solo al final. La cita debe estar en la frase misma.
- **NO GENERALICES.** Si la información proviene de un caso específico (ej. 'proyecto minero El Cerrejón'), DEBES decirlo. Ejemplo: "En el caso específico del proyecto minero El Cerrejón, la Sentencia T-704 de 2016 reconoció el derecho a la consulta previa...".
- **TU ÚNICA FUENTE DE VERDAD ES EL SIGUIENTE RESUMEN TÉCNICO.** Basa tu respuesta 100% y ÚNICAMENTE en este resumen. No uses ningún conocimiento externo.

**Instrucciones adicionales:**
- Traduce el lenguaje técnico del resumen a un lenguaje claro y sencillo.
- Usa listas y **negritas** para que la información sea fácil de leer.
- Si el resumen indica que no hay información, responde amablemente: "Hola, no he encontrado información precisa sobre lo que me preguntas. ¿Podrías intentar con otras palabras?".

**Resumen técnico para Eureka:**
---
{respuesta_tecnica}
---

**Pregunta original del usuario:** {question}

**Respuesta de Eureka:**
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
        return PromptTemplate.from_template(tpl_or_prompt)
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
    db = Chroma(
        persist_directory=DIRECTORIO_CHROMA_DB, 
        embedding_function=embeddings,
        collection_name=NOMBRE_COLECCION
    )
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

    # === Filtro de intención mejorado para mantener contexto ===
    intent = clasificar_intencion(user_q)
    
    # Solo mostrar respuesta genérica para saludos genuinamente nuevos
    if intent in ("saludo", "charla") and len(st.session_state.messages) <= 2:
        # Solo si es realmente el inicio de la conversación
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
    
    # Para respuestas muy cortas, expandir la consulta con contexto
    if len(user_q.strip()) <= 10 and len(st.session_state.messages) > 2:
        # Obtener la última pregunta del asistente para dar contexto
        last_assistant_msg = ""
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant":
                last_assistant_msg = msg["content"]
                break
        
        # Si la respuesta corta sigue a una pregunta específica, expandir contexto
        if "?" in last_assistant_msg:
            user_q = f"{user_q}. Contexto: respondiendo a la pregunta sobre derechos y procedimientos ambientales"

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

                # Agregar fuentes (Lógica Mejorada y Precisa)
                fuentes_citadas = set()
                # Extraer los índices de los documentos realmente usados desde la respuesta técnica
                indices_usados = re.findall(r'\[DOC (\d+)', resp_tecnica)

                if indices_usados:
                    # Construir la lista de fuentes a partir de los índices encontrados
                    for i_str in set(indices_usados): # Usar set para evitar duplicados
                        try:
                            index = int(i_str) - 1
                            if 0 <= index < len(docs):
                                fuente = _safe_get_source(docs[index])
                                if fuente != "Fuente no encontrada":
                                    fuentes_citadas.add(fuente)
                        except (ValueError, IndexError):
                            continue
                
                # Fallback: si no hay tags pero sí respuesta, citar todas las fuentes recuperadas
                if not fuentes_citadas and "No he encontrado información" not in respuesta_final:
                    fuentes_citadas = {_safe_get_source(d) for d in docs if _safe_get_source(d) != "Fuente no encontrada"}
                
                if fuentes_citadas:
                    fuentes_ordenadas = sorted(list(fuentes_citadas))
                    # Solo añadir la sección de fuentes si hay fuentes que citar
                    respuesta_con_fuentes = respuesta_final + "\n\n---\n**Fuentes consultadas:**\n" + "\n".join(f"• {u}" for u in fuentes_ordenadas)
                    contenedor.markdown(respuesta_con_fuentes)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_con_fuentes})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_final})


                # ===============================================
                # VISTA DE ADMINISTRADOR MEJORADA PARA DEPURACIÓN
                # ===============================================
                with st.expander("🔧 Vista de Administrador - DEPUREMOS EL PROCESO"):
                    st.subheader("1. Documentos Recuperados de la Base de Datos")
                    st.info("Aquí ves el contenido COMPLETO de los documentos que el sistema encontró como potencialmente relevantes para tu pregunta. Son la 'materia prima'.")
                    for i, doc in enumerate(docs, 1):
                        with st.container(border=True):
                            fuente = _safe_get_source(doc)
                            st.write(f"**📄 Documento {i}:** `{fuente}`")
                            st.text_area(
                                f"Contenido completo del Documento {i}",
                                doc.page_content,
                                height=200,
                                key=f"full_doc_content_{i}"
                            )
                            st.json(doc.metadata, expanded=False)
                    
                    st.subheader("2. Contexto Enviado al Primer Analista (Extractor)")
                    st.info("Este es el texto EXACTO que se le entrega al primer modelo de IA. Es la unión de todos los documentos anteriores. Aquí es donde puede haber 'ruido' o información irrelevante.")
                    st.text_area("Contexto completo", contexto, height=300, key="contexto_completo_debug")
                    
                    st.subheader("3. Respuesta Técnica del Extractor")
                    st.info("Esta es la respuesta CRUDA del primer modelo de IA. Su única tarea es resumir los hechos del texto anterior y citar de dónde los sacó (ej. [DOC 1]). **Aquí podemos detectar si la primera IA ya está inventando o mezclando información.**")
                    st.text_area("Extracción técnica (Salida cruda)", resp_tecnica, height=300, key="respuesta_tecnica_debug")
                    
                    st.subheader("4. Prompt Final Enviado a Eureka (El Chatbot)")
                    st.info("Estas son las instrucciones EXACTAS que recibe el chatbot final. Incluyen las 'Reglas de Oro' y la 'Respuesta Técnica' del paso anterior. **Si la respuesta técnica es correcta pero la respuesta final es incorrecta, el problema está en cómo la IA final interpreta estas instrucciones.**")
                    prompt_final_eureka = eureka_pt.format(**eureka_kwargs)
                    st.text_area("Prompt completo para Eureka", prompt_final_eureka, height=400, key="prompt_final_debug")

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
                st.write("Intenta con otra pregunta o verifica la conexión con Ollama.")

