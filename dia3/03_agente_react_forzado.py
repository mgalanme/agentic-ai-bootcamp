#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agente ReAct que SIEMPRE usa herramientas cuando la pregunta lo requiere.
Incluye verificación post-llamada y reintento forzado.
"""

import os
import time
import json
import logging
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# ============================================================
# HERRAMIENTAS (mismas que antes)
# ============================================================

@tool
def buscar_documentacion(query: str) -> str:
    """Busca en la documentación técnica de arquitectura de datos."""
    docs = {
        "particionado": "El particionado de tablas divide los datos en segmentos físicos por una columna (fecha, región). Mejora el rendimiento de queries que filtran por esa columna al hacer partition pruning. En Snowflake se llama clustering, en BigQuery partitioning.",
        "medallion": "La arquitectura Medallion organiza el Data Lake en capas Bronze (raw), Silver (curado) y Gold (agregado). Bronze es inmutable, Silver aplica reglas de negocio, Gold sirve consumo directo de BI y ML.",
        "estrella": "El modelo en estrella tiene una tabla de hechos central conectada a dimensiones desnormalizadas. Optimizado para queries OLAP. El modelo en copo de nieve normaliza las dimensiones reduciendo redundancia pero aumentando los JOINs.",
        "embeddings": "Los embeddings son vectores de alta dimensión que representan el significado semántico del texto. Textos similares tienen vectores cercanos en el espacio vectorial medido por cosine similarity.",
        "rag": "RAG combina recuperación de documentos relevantes con generación de texto. El retriever busca los chunks más similares a la pregunta, el LLM genera la respuesta basándose en ese contexto.",
        "langgraph": "LangGraph modela flujos agénticos como grafos de estado. State=esquema de datos, Nodes=transformaciones, Edges=conexiones condicionales. Permite ciclos, memoria persistente y human-in-the-loop."
    }
    for keyword, content in docs.items():
        if keyword.lower() in query.lower():
            return content
    return f"No encontré documentación específica para: {query}. Prueba con: particionado, medallion, estrella, embeddings, rag, langgraph"

@tool
def calcular_coste_almacenamiento(gb_datos: float, proveedor: str, tipo: str) -> str:
    """Calcula el coste mensual estimado de almacenamiento de datos."""
    costes = {
        "snowflake": {"storage": 23.0, "compute": 2.0},
        "bigquery":  {"storage": 0.02, "compute": 5.0},
        "redshift":  {"storage": 0.024, "compute": 0.25},
        "s3":        {"storage": 0.023, "compute": 0.0}
    }
    p = proveedor.lower()
    if p not in costes:
        return f"Proveedor no reconocido: {proveedor}. Opciones: snowflake, bigquery, redshift, s3"

    storage_cost = gb_datos * costes[p]["storage"]
    compute_cost = costes[p]["compute"] * 100
    total = storage_cost + compute_cost

    if tipo == "storage":
        return f"{proveedor}: ${storage_cost:.2f}/mes por {gb_datos}GB almacenamiento"
    elif tipo == "compute":
        return f"{proveedor}: ${compute_cost:.2f}/mes estimado de cómputo"
    else:
        return f"{proveedor}: ${total:.2f}/mes total (storage: ${storage_cost:.2f} + compute: ${compute_cost:.2f}) para {gb_datos}GB"

@tool
def analizar_patron_arquitectonico(descripcion: str) -> str:
    """Analiza un problema de arquitectura de datos y recomienda el patrón más adecuado."""
    desc = descripcion.lower()
    if any(w in desc for w in ["tiempo real", "streaming", "eventos", "kafka"]):
        return "Patrón recomendado: Lambda Architecture o Kappa Architecture. Lambda combina batch y streaming. Kappa unifica todo en streaming. Tecnologías: Apache Kafka + Flink o Spark Streaming."
    elif any(w in desc for w in ["histórico", "batch", "diario", "etl"]):
        return "Patrón recomendado: Medallion Architecture con Delta Lake o Apache Iceberg. Procesos batch orquestados con Airflow. Transformaciones con dbt. Particionado por fecha para optimizar queries históricas."
    elif any(w in desc for w in ["ml", "modelo", "features", "entrenamiento"]):
        return "Patrón recomendado: Feature Store + Model Registry. MLflow para tracking de experimentos. Feast o Tecton para feature store. Separar features offline (batch) de online (tiempo real)."
    elif any(w in desc for w in ["rag", "documentos", "búsqueda", "lenguaje natural"]):
        return "Patrón recomendado: RAG con vector DB. Chroma para desarrollo, Qdrant para producción. Chunking con RecursiveCharacterTextSplitter. Embeddings con nomic-embed-text. LangGraph para orquestación."
    return "Patrón recomendado: Analizar con más contexto. Describe volumen de datos, latencia requerida y casos de uso principales para una recomendación más precisa."

# ============================================================
# PROMPT DEL SISTEMA QUE OBLIGA A USAR HERRAMIENTAS
# ============================================================
SYSTEM_PROMPT = """Eres un asistente de arquitectura de datos. Para responder correctamente, DEBES usar las herramientas disponibles:

- Si te preguntan sobre conceptos técnicos (particionado, medallion, estrella, embeddings, RAG, LangGraph), usa la herramienta 'buscar_documentacion'.
- Si te preguntan sobre costes de almacenamiento o cómputo en la nube (Snowflake, BigQuery, Redshift, S3), usa la herramienta 'calcular_coste_almacenamiento'.
- Si te preguntan sobre arquitecturas para casos de uso específicos (batch, streaming, ML, RAG), usa la herramienta 'analizar_patron_arquitectonico'.

NUNCA respondas directamente sin antes invocar la herramienta apropiada. 
Tu flujo debe ser: detectar la necesidad -> llamar a la herramienta -> esperar el resultado -> responder basándote en el resultado de la herramienta.
Si no usas ninguna herramienta, tu respuesta será considerada inválida.
"""

# ============================================================
# LLM CONFIGURACIÓN
# ============================================================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    timeout=30,
    max_retries=2
)

tools = [buscar_documentacion, calcular_coste_almacenamiento, analizar_patron_arquitectonico]
llm_con_tools = llm.bind_tools(tools)

# ============================================================
# ESTADO Y NODOS DEL GRAFO (con verificación de herramienta forzada)
# ============================================================

class EstadoReAct(TypedDict):
    messages: Annotated[List, add_messages]

def agente(state: EstadoReAct) -> dict:
    """Nodo agente que fuerza el uso de herramientas."""
    messages = state["messages"]
    # Si es la primera vez y no hay mensaje del sistema, lo añadimos
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm_con_tools.invoke(messages)
    
    # Verificación: si la pregunta es de tipo que requiere herramienta y el LLM no llamó a ninguna, reintentamos
    last_user_msg = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    
    # Palabras clave que indican que se necesita herramienta
    necesita_herramienta = any(kw in last_user_msg.lower() for kw in 
                               ["particionado", "coste", "costo", "snowflake", "bigquery", "redshift", 
                                "arquitectura", "batch", "streaming", "ml", "features", "patrón"])
    
    if necesita_herramienta and not response.tool_calls:
        logger.warning("El LLM no usó herramientas a pesar de necesitarlas. Reintentando con prompt más fuerte.")
        # Añadir un mensaje de usuario adicional forzando la herramienta
        forced_prompt = HumanMessage(content="IMPORTANTE: No respondas directamente. Usa la herramienta apropiada para esta pregunta.")
        new_messages = messages + [forced_prompt]
        response = llm_con_tools.invoke(new_messages)
    
    return {"messages": [response]}

tool_node = ToolNode(tools)

# ============================================================
# CONSTRUCCIÓN DEL GRAFO
# ============================================================
builder = StateGraph(EstadoReAct)
builder.add_node("agente", agente)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agente")
builder.add_conditional_edges("agente", tools_condition)
builder.add_edge("tools", "agente")

graph = builder.compile()

# ============================================================
# EJECUCIÓN CON FORZADO DE HERRAMIENTAS
# ============================================================
def ejecutar_preguntas(preguntas: List[str]):
    for pregunta in preguntas:
        print(f"\n{'='*55}")
        print(f"Pregunta: {pregunta}")
        print("Trazado de ejecución:")
        try:
            resultado = graph.invoke(
                {"messages": [HumanMessage(content=pregunta)]},
                {"recursion_limit": 10}
            )
            # Mostrar herramientas usadas
            tool_used = False
            for msg in resultado["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_used = True
                    for tc in msg.tool_calls:
                        print(f"  -> tool: {tc['name']} | args: {json.dumps(tc['args'], ensure_ascii=False)[:80]}")
                elif isinstance(msg, ToolMessage):
                    print(f"  <- resultado: {msg.content[:100]}...")
            
            if not tool_used:
                print("  [ADVERTENCIA] No se usó ninguna herramienta (a pesar del forzado).")
            
            # Respuesta final
            for msg in reversed(resultado["messages"]):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    print(f"\nRespuesta final:\n{msg.content[:500]}...")
                    break
        except Exception as e:
            logger.error(f"Error: {e}")
        print("-"*55)

if __name__ == "__main__":
    print("="*55)
    print("AGENTE ReAct CON USO FORZADO DE HERRAMIENTAS")
    print("="*55)
    
    preguntas = [
        "¿Qué es el particionado de tablas y cuándo usarlo?",
        "¿Cuánto me costaría en Snowflake almacenar 500GB de datos y cuál sería el coste total?",
        "Tengo datos de ventas que proceso en batch cada noche y también necesito features para un modelo de ML. ¿Qué arquitectura me recomiendas?"
    ]
    ejecutar_preguntas(preguntas)
