# 2_app_chatbot.py (Versión Limpia - Con parche SQLite3 robusto)

# ... existing code ...
# Configuración
# =====================
DIRECTORIO_CHROMA_DB = os.environ.get("CHROMA_DB_DIR", "chroma_db")
MODELO_EMBEDDING = os.environ.get("EMBED_MODEL", "nomic-embed-text")
MODELO_LLM = os.environ.get("LLM_MODEL", "llama3.2")
NOMBRE_COLECCION = "sentencias_anla" # <-- AÑADIDO: Nombre explícito para la colección

# Parámetros MMR simples
K_DOCUMENTOS = 4
# ... existing code ...
@st.cache_resource(show_spinner=False)
def cargar_componentes(base_url: str):
    ok, detail = _health_check_ollama(base_url)
    if not ok:
        raise RuntimeError(f"No puedo conectarme a Ollama en {base_url}. Detalle: {detail}")
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING, base_url=base_url)
    # --- LÍNEA CORREGIDA ---
    # Ahora se conecta a la colección específica que creamos
    db = Chroma(
        persist_directory=DIRECTORIO_CHROMA_DB, 
        embedding_function=embeddings,
        collection_name=NOMBRE_COLECCION # <-- AÑADIDO
    )
    llm_extract = OllamaLLM(model=MODELO_LLM, temperature=0.2, base_url=base_url)
    llm_eureka_stream = OllamaLLM(model=MODELO_LLM, temperature=0.2, base_url=base_url, streaming=True)
# ... existing code ...
