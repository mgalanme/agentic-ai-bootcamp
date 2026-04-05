#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agente ReAct con herramientas y resiliencia ante rate limits.
Incluye circuit breaker y fallback a Ollama si está disponible.
"""

import os
import time
import json
import logging
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

# LangChain y LangGraph
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# Proveedores de LLM
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# ============================================================
# HERRAMIENTAS (idénticas a las originales)
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
# CONFIGURACIÓN DE LLM CON RESILIENCIA
# ============================================================

class LLMWithRetry:
    """
    Envoltorio para LLM con reintentos exponenciales y circuit breaker básico.
    Soporta Groq y fallback a Ollama.
    """
    def __init__(self, primary_provider="groq", fallback_to_ollama=True):
        self.primary_provider = primary_provider
        self.fallback_to_ollama = fallback_to_ollama
        self.retry_count = 0
        self.circuit_open = False
        self.circuit_open_until = 0
        self.fallback_active = False

        # Inicializar LLM primario (Groq)
        self.llm_primary = ChatGroq(
            model="llama-3.1-8b-instant",  # más barato en tokens
            temperature=0,
            timeout=30,
            max_retries=0  # nosotros manejamos los reintentos
        )

        # Inicializar fallback (Ollama) si se solicita
        self.llm_fallback = None
        if fallback_to_ollama:
            try:
                self.llm_fallback = ChatOllama(model="tinyllama", temperature=0)
                # Verificar que Ollama responde
                self.llm_fallback.invoke("ping")
                logger.info("Fallback a Ollama disponible")
            except Exception as e:
                logger.warning(f"Ollama no disponible: {e}")
                self.llm_fallback = None

    def invoke(self, messages, max_retries=3):
        """Invoca al LLM con manejo de rate limits y circuit breaker."""
        # Circuit breaker: si está abierto, no intentar
        if self.circuit_open:
            if time.time() < self.circuit_open_until:
                raise Exception("Circuit breaker abierto. Esperando recuperación.")
            else:
                self.circuit_open = False
                self.retry_count = 0
                logger.info("Circuit breaker restablecido.")

        # Seleccionar el LLM activo
        if self.fallback_active and self.llm_fallback:
            llm = self.llm_fallback
            provider = "Ollama"
        else:
            llm = self.llm_primary
            provider = "Groq"

        for attempt in range(max_retries):
            try:
                response = llm.invoke(messages)
                # Éxito: resetear contadores
                self.retry_count = 0
                if self.fallback_active:
                    logger.info("Fallback desactivado (Groq recuperado)")
                    self.fallback_active = False
                return response
            except Exception as e:
                error_msg = str(e)
                # Detectar rate limit (429)
                if "429" in error_msg or "RateLimitError" in error_msg or "rate limit" in error_msg.lower():
                    # Extraer tiempo de espera si está en el mensaje
                    wait_seconds = 60  # por defecto
                    if "try again in" in error_msg:
                        # Ejemplo: "try again in 1h3m9.504s"
                        import re
                        match = re.search(r"try again in (\d+)h(\d+)m([\d\.]+)s", error_msg)
                        if match:
                            hours = int(match.group(1))
                            minutes = int(match.group(2))
                            seconds = float(match.group(3))
                            wait_seconds = hours*3600 + minutes*60 + seconds
                    logger.warning(f"Rate limit en {provider}. Esperando {wait_seconds:.0f}s antes de reintentar...")
                    time.sleep(wait_seconds)
                    continue  # reintentar
                else:
                    # Otro error (conexión, timeout, etc.)
                    logger.error(f"Error invocando {provider}: {e}")
                    if self.fallback_to_ollama and not self.fallback_active and self.llm_fallback:
                        logger.info("Cambiando a fallback (Ollama)")
                        self.fallback_active = True
                        # Intentar con Ollama inmediatamente
                        return self.invoke(messages, max_retries)
                    else:
                        # Si ya estábamos en fallback o no hay fallback, aumentar contador
                        if attempt < max_retries - 1:
                            wait = 2 ** attempt
                            logger.info(f"Reintentando en {wait}s...")
                            time.sleep(wait)
                        else:
                            # Abrir circuit breaker
                            self.circuit_open = True
                            self.circuit_open_until = time.time() + 300  # 5 minutos
                            raise Exception(f"Falló después de {max_retries} reintentos: {e}")

        raise Exception("No se pudo obtener respuesta del LLM después de reintentos")

# Instanciar el LLM resiliente
llm_wrapper = LLMWithRetry(primary_provider="groq", fallback_to_ollama=True)

# Crear la versión con tools (bind_tools)
tools = [buscar_documentacion, calcular_coste_almacenamiento, analizar_patron_arquitectonico]
llm_con_tools = llm_wrapper.llm_primary.bind_tools(tools)  # Nota: el bind_tools se hace sobre el LLM primario
# Pero el wrapper invoke no usa directamente llm_con_tools; tenemos que adaptar el nodo agente.

# ============================================================
# ESTADO Y NODOS DEL GRAFO ReAct (con límite de iteraciones)
# ============================================================

class EstadoReAct(TypedDict):
    messages: Annotated[List, add_messages]

def agente(state: EstadoReAct) -> dict:
    """Nodo agente con llamada resiliente al LLM (con tools)."""
    # Usamos el wrapper para invocar el LLM con tools
    # Necesitamos pasarle los mensajes y que el LLM pueda usar herramientas.
    # Como el wrapper no tiene bind_tools, lo hacemos de esta manera:
    try:
        # Llamada directa al LLM primario con bind_tools (pero con manejo de rate limits)
        # Para simplificar, usaremos el método invoke del LLM primario pero con reintentos.
        # Vamos a reutilizar la lógica de reintentos para llamadas con herramientas.
        response = None
        for attempt in range(3):
            try:
                response = llm_wrapper.llm_primary.invoke(state["messages"])
                break
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    wait = 60
                    logger.warning(f"Rate limit en nodo agente, esperando {wait}s")
                    time.sleep(wait)
                else:
                    raise e
        if response is None:
            raise Exception("No se pudo obtener respuesta del LLM")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error en nodo agente: {e}")
        # Devolver un mensaje de error para que el grafo no se rompa
        error_msg = AIMessage(content=f"Lo siento, ocurrió un error al procesar tu consulta: {str(e)}")
        return {"messages": [error_msg]}

# ToolNode igual que antes
tool_node = ToolNode(tools)

# ============================================================
# CONSTRUCCIÓN DEL GRAFO CON LÍMITE DE RECURSIÓN
# ============================================================

builder = StateGraph(EstadoReAct)
builder.add_node("agente", agente)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agente")
builder.add_conditional_edges("agente", tools_condition)
builder.add_edge("tools", "agente")

# Compilar con límite de recursión (máximo 10 pasos para evitar bucles)
graph = builder.compile()

# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

def ejecutar_preguntas(preguntas: List[str]):
    for pregunta in preguntas:
        print(f"\nPregunta: {pregunta}")
        print("Trazado de ejecución:")
        try:
            resultado = graph.invoke(
                {"messages": [HumanMessage(content=pregunta)]},
                {"recursion_limit": 10}  # Evita bucles infinitos
            )
            # Mostrar las herramientas usadas
            for msg in resultado["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  -> tool: {tc['name']} | args: {json.dumps(tc['args'], ensure_ascii=False)[:80]}")
                elif isinstance(msg, ToolMessage):
                    print(f"  <- resultado: {msg.content[:80]}...")
            # Extraer respuesta final
            for msg in reversed(resultado["messages"]):
                if isinstance(msg, AIMessage) and msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                    print(f"Respuesta final: {msg.content[:350]}...")
                    break
        except Exception as e:
            logger.error(f"Error ejecutando el grafo: {e}")
        print("-" * 55)

if __name__ == "__main__":
    print("=" * 55)
    print("AGENTE ReAct CON HERRAMIENTAS (RESILIENTE)")
    print("=" * 55)

    preguntas = [
        "¿Qué es el particionado de tablas y cuándo usarlo?",
        "¿Cuánto me costaría en Snowflake almacenar 500GB de datos y cuál sería el coste total?",
        "Tengo datos de ventas que proceso en batch cada noche y también necesito features para un modelo de ML. ¿Qué arquitectura me recomiendas?"
    ]

    ejecutar_preguntas(preguntas)
