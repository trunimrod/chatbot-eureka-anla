# 1_ingestar_documentos_json.py (Versión Optimizada para JSON)

import os
import json
import shutil
import logging
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- Configuración de Logging ---
# Configura un registro detallado para seguir el proceso.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes de Configuración ---
# Define las rutas y modelos a utilizar.
# El script ahora apunta directamente al archivo JSON.
ARCHIVO_JSON = "./documentos/sentencias_extraidas.json" 
DIRECTORIO_CHROMA_DB = "./chroma_db"
MODELO_EMBEDDING = "nomic-embed-text" # Modelo para crear los vectores (embeddings)

def construir_base_de_datos_desde_json():
    """
    Construye una base de datos vectorial a partir de un archivo JSON estructurado.
    Esta versión es más robusta y extrae metadatos ricos.
    """
    # 1. Verificar la existencia del archivo JSON de entrada.
    if not os.path.exists(ARCHIVO_JSON):
        logger.error(f"El archivo de datos '{ARCHIVO_JSON}' no fue encontrado.")
        return
    
    # 2. Si existe una base de datos vectorial anterior, la elimina para reconstruirla.
    if os.path.exists(DIRECTORIO_CHROMA_DB):
        logger.warning(f"Eliminando base de datos vectorial existente en '{DIRECTORIO_CHROMA_DB}'.")
        shutil.rmtree(DIRECTORIO_CHROMA_DB)

    # 3. Cargar los datos desde el archivo JSON.
    logger.info(f"Cargando documentos desde '{ARCHIVO_JSON}'...")
    try:
        with open(ARCHIVO_JSON, 'r', encoding='utf-8') as f:
            datos_json = json.load(f)
    except Exception as e:
        logger.error(f"Error al leer o decodificar el archivo JSON: {e}")
        return
    
    documentos_langchain = []
    # 4. Procesar cada registro del JSON para convertirlo en un objeto "Document" de LangChain.
    for item in datos_json:
        # Combinar los campos de texto más relevantes para crear un contenido completo.
        # Esto le da más contexto al modelo de lenguaje para entender el documento.
        contenido_combinado = (
            f"Título: {item.get('titulo', '')}\n\n"
            f"Descripción: {item.get('descripcion_breve', '')}\n\n"
            f"Resumen: {item.get('resumen', '')}"
        )
        
        # Extraer y estructurar los metadatos. Son cruciales para la búsqueda y citación.
        # El campo 'url' se mapea a 'source' que es el estándar usado por LangChain para las fuentes.
        metadata = {
            'source': item.get('url', 'Fuente no encontrada'),
            'title': item.get('titulo', 'Sin título'),
            'category': item.get('categoria', 'Sin categoría'),
            'source_entity': item.get('fuente', 'Fuente no especificada'),
            'keywords': ', '.join(item.get('palabras_claves', []))
        }
        
        # Crear el objeto Document y añadirlo a la lista.
        documentos_langchain.append(Document(page_content=contenido_combinado, metadata=metadata))

    logger.info(f"Se han procesado {len(documentos_langchain)} documentos desde el archivo JSON.")
    
    # 5. Dividir los documentos en fragmentos (chunks) más pequeños.
    # Esto es necesario para que los modelos de lenguaje puedan procesarlos eficientemente.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documentos_langchain)
    
    logger.info(f"Documentos divididos en {len(chunks)} fragmentos (chunks).")

    # 6. Inicializar el modelo de embeddings que convertirá el texto a vectores.
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)
    
    # 7. Crear la base de datos vectorial ChromaDB a partir de los chunks y sus embeddings.
    logger.info("Iniciando creación de la base de datos vectorial. Este proceso puede tardar...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DIRECTORIO_CHROMA_DB # Directorio donde se guardará la DB
    )
    
    logger.info("✅ Proceso completado. La base de datos vectorial ha sido guardada en '%s'.", DIRECTORIO_CHROMA_DB)

if __name__ == "__main__":
    # Asegurarse de que el directorio de documentos exista para poner el JSON allí.
    if not os.path.isdir("./documentos"):
        os.makedirs("./documentos")
        logger.info("Se ha creado el directorio './documentos/'. Asegúrate de colocar tu archivo 'sentencias_extraidas.json' dentro.")
    
    construir_base_de_datos_desde_json()
