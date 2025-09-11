# 2_app_chatbot.py — MMR + límite de contexto + streaming + logs SQLite + ngrok sin secrets
# Detecta automáticamente las variables del prompt (context/question vs contexto/pregunta)

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
from typing import Dict, Any, Iterable

import streamlit as st
import requests

from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate  # puede recibirse str o PromptTemplate desde prompts.py
from langchain_core.output_parsers import StrOutputParser

# Prompts existentes (debes tener este archivo junto a 2_app_chatbot.py)
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
    return {"k": k, "fetch_k": FETCH_K, "lambda_mult": MMR_LAMBDA}


def filtrar_documentos_por_relevancia(documentos, pregunta: str, es_especifica: bool):
    """Para preguntas generales, evita docs con demasiados nombres propios (>3)."""
    if es_especifica:
        return documentos
    docs_filtrados = []
    patron_np = re.compile(r"\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)*\b")
    for doc in documentos:
        contenido = doc.page_content or ""
