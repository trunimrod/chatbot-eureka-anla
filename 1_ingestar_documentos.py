# 1_ingestar_documentos.py (Versión con Regex Corregido)

import os
import re
import shutil
import logging
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes de Configuración ---
DIRECTORIO_DOCUMENTOS = "./documentos/"
DIRECTORIO_CHROMA_DB = "./chroma_db"
MODELO_EMBEDDING = "nomic-embed-text"

def construir_base_de_datos_simple():
    """
    Construye la base de datos vectorial con una estrategia de chunking simple.
    """
    if not os.path.isdir(DIRECTORIO_DOCUMENTOS):
        logger.error(f"El directorio de documentos '{DIRECTORIO_DOCUMENTOS}' no fue encontrado.")
        return
    
    if os.path.exists(DIRECTORIO_CHROMA_DB):
        logger.warning(f"Eliminando base de datos existente en '{DIRECTORIO_CHROMA_DB}'.")
        shutil.rmtree(DIRECTORIO_CHROMA_DB)

    logger.info(f"Cargando documentos desde '{DIRECTORIO_DOCUMENTOS}'...")
    loader = DirectoryLoader(
        DIRECTORIO_DOCUMENTOS,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True
    )
    documentos = loader.load()

    # --- LÍNEA CORREGIDA ---
    # Esta nueva expresión regular es más robusta y sí captura la URL del formato de tus archivos.
    url_pattern = re.compile(r'\*\*URL:\*\*\s*<(.+?)>')

    for doc in documentos:
        url_match = url_pattern.search(doc.page_content)
        # Asignar la URL encontrada al metadato 'source' que usa el retriever
        if url_match:
            doc.metadata['source'] = url_match.group(1)
        else:
            doc.metadata['source'] = doc.metadata.get('source', 'Fuente no encontrada')

    logger.info(f"Se cargaron {len(documentos)} documentos.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documentos)
    
    logger.info(f"Documentos divididos en {len(chunks)} fragmentos (chunks).")

    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)
    
    logger.info("Iniciando creación de la base de datos vectorial. Esto puede tardar...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DIRECTORIO_CHROMA_DB
    )
    
    logger.info("✅ Proceso completado. La base de datos ha sido guardada en '%s'.", DIRECTORIO_CHROMA_DB)

if __name__ == "__main__":
    construir_base_de_datos_simple()