# prompts.py — prompts con enfoque de derechos para casos tipo “embalse / seguridad hídrica”
from langchain.prompts import PromptTemplate

# --- Modo técnico general (mantén para preguntas neutras) ---
EXTRACTOR_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Eres analista técnico ambiental. Responde SOLO con información sustentada en el CONTEXTO.\n"
        "Pregunta del usuario: {question}\n\n"
        "CONTEXTO (fragmentos de documentos recuperados):\n{context}\n\n"
        "Tarea: elabora un RESUMEN TÉCNICO con:\n"
        "1) Hechos/medidas relevantes explícitas en el contexto (sin inventar)\n"
        "2) Incertidumbres o datos faltantes (di 'no encontrado' si no aparece)\n"
        "3) Procedimientos/obligaciones mencionadas (p. ej., EIA, licencia, monitoreo)\n"
        "4) Extractos breves entre comillas si aportan\n"
        "5) Lista corta de términos clave\n"
        "No cites normas ni números si el contexto no los trae. No hagas recomendaciones aún."
    ),
)

EUREKA_PROMPT = PromptTemplate(
    input_variables=["technical_summary", "original_question"],
    template=(
        "Convierte el siguiente RESUMEN TÉCNICO en una explicación clara para público general.\n\n"
        "Pregunta original: {original_question}\n\n"
        "Resumen técnico:\n{technical_summary}\n\n"
        "Instrucciones:\n"
        "- Explica en 2–4 párrafos, con viñetas si ayuda.\n"
        "- Evita jerga; sé concreto.\n"
        "- Señala lo que falta por confirmar, sin inventar."
    ),
)

# --- Modo DERECHOS (nuevo): centra la respuesta en protección de la comunidad afectada ---
EXTRACTOR_PROMPT_RIGHTS = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Actúas como analista de licenciamiento ambiental con ENFOQUE DE DERECHOS. "
        "El usuario es una PERSONA/COMUNIDAD POTENCIALMENTE AFECTADA por un proyecto (p.ej., embalse, captación de agua).\n\n"
        "Pregunta: {question}\n\n"
        "CONTEXTO (fragmentos de documentos recuperados):\n{context}\n\n"
        "Tarea: produce un **RESUMEN TÉCNICO-REGULATORIO** SOLO con lo que esté en el contexto, organizado en secciones:\n"
        "A) Impactos y riesgos relevantes para la **seguridad hídrica**, especialmente en estiaje (verano): disponibilidad, prioridad de usos, caudal ecológico, balances hídricos.\n"
        "B) **Derechos de la comunidad** potencialmente involucrados (ej.: derecho al agua, ambiente sano, participación, acceso a información, consulta previa si aplica). "
        "Menciona normas o jurisprudencia SOLO si aparecen en el contexto; si no, enuncia el derecho en términos generales (sin números).\n"
        "C) **Obligaciones del titular y de la autoridad** (evaluación de impactos, monitoreo, medidas, contingencias, participación ciudadana). No des responsabilidades a la comunidad.\n"
        "D) **Qué exigir en la evaluación/licencia**: (ejemplos típicos) balances hídricos con series históricas, definición/soporte del caudal ecológico, modelación de estiaje, plan de manejo de captaciones, PUEAA/ahorro, plan de contingencia y suspensión temporal de captación si se alcanzan umbrales, programa de monitoreo con umbrales y reporte público.\n"
        "E) Falencias o información faltante en el contexto (di 'no encontrado' si aplica).\n\n"
        "Reglas:\n"
        "- No inventes normas, cifras ni autoridades si no están en el contexto.\n"
        "- Si el contexto menciona normas/sentencias, refiérete a ellas de forma breve y fiel.\n"
        "- No responsabilices a la comunidad de garantizar la seguridad hídrica; enfoca en deberes del proyecto y de la autoridad."
    ),
)

EUREKA_PROMPT_RIGHTS = PromptTemplate(
    input_variables=["technical_summary", "original_question"],
    template=(
        "Transforma el RESUMEN TÉCNICO-REGULATORIO en una guía clara y empática centrada en DERECHOS para una comunidad afectada.\n\n"
        "Pregunta original: {original_question}\n\n"
        "Resumen técnico-regulatorio:\n{technical_summary}\n\n"
        "Entrega la respuesta con esta estructura:\n"
        "1) **En breve (3 viñetas)**: qué pasaría con la seguridad hídrica y en estiaje.\n"
        "2) **Tus derechos** (agua, ambiente sano, participación, acceso a información, consulta previa si aplica). "
        "Si no hay norma específica en el contexto, nómbralos sin números.\n"
        "3) **Qué debe garantizar el proyecto y la autoridad** (obligaciones concretas, p.ej., caudal ecológico, balances hídricos, umbrales de suspensión, monitoreo público).\n"
        "4) **Qué puedes pedir formalmente**: participación/audiencia, acceso a información, observaciones al EIA, PQRD ante la autoridad, y recursos administrativos. "
        "Usa lenguaje simple y pasos accionables.\n"
        "5) **Señales de alerta** que justifican pedir medidas preventivas.\n"
        "Nota: esto es orientación informativa, no asesoría legal."
    ),
)
