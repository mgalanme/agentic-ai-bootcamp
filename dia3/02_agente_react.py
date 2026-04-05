import time
import json
import logging
import re
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# ============================================================
# HERRAMIENTAS
# ============================================================

@tool
def buscar_documentacion(query: str) -> str:
    """Busca información técnica sobre arquitectura de datos.
    Usar para: particionado, medallion, estrella, embeddings, rag, langgraph."""
    docs = {
        "particionado": "El particionado divide tablas en segmentos físicos por columna (fecha, región). Mejora rendimiento con partition pruning. En Snowflake: clustering. En BigQuery: partitioning. Útil cuando queries filtran habitualmente por la misma columna.",
        "medallion": "Arquitectura Medallion: Bronze (raw inmutable), Silver (curado con reglas de negocio), Gold (agregado para BI y ML). Estándar en Data Lakehouses modernos.",
        "estrella": "Modelo estrella: tabla de hechos central + dimensiones desnormalizadas. Optimizado para OLAP. Copo de nieve normaliza dimensiones: menos redundancia pero más JOINs.",
        "embeddings": "Embeddings: vectores de alta dimensión que representan significado semántico. Textos similares tienen vectores cercanos medidos por cosine similarity.",
        "rag": "RAG: retriever busca chunks similares a la pregunta, LLM genera respuesta con ese contexto. Reduce alucinaciones al fundamentar respuestas en documentos reales.",
        "langgraph": "LangGraph: grafos de estado para agentes. State=esquema de datos, Nodes=transformaciones, Edges=conexiones condicionales. Soporta ciclos, memoria persistente y human-in-the-loop."
    }
    for keyword, content in docs.items():
        if keyword.lower() in query.lower():
            return content
    return f"No encontré docs para: {query}. Términos válidos: particionado, medallion, estrella, embeddings, rag, langgraph"

@tool
def calcular_coste_almacenamiento(gb_datos: float, proveedor: str, tipo: str) -> str:
    """Calcula coste mensual de almacenamiento.
    proveedor: snowflake, bigquery, redshift, s3
    tipo: storage, compute, total"""
    costes = {
        "snowflake": {"storage": 23.0,  "compute": 2.0},
        "bigquery":  {"storage": 0.02,  "compute": 5.0},
        "redshift":  {"storage": 0.024, "compute": 0.25},
        "s3":        {"storage": 0.023, "compute": 0.0}
    }
    p = proveedor.lower()
    if p not in costes:
        return f"Proveedor no reconocido: {proveedor}. Usa: snowflake, bigquery, redshift, s3"
    storage_cost = gb_datos * costes[p]["storage"]
    compute_cost = costes[p]["compute"] * 100
    total = storage_cost + compute_cost
    if tipo == "storage":
        return f"{proveedor}: ${storage_cost:.2f}/mes por {gb_datos}GB"
    elif tipo == "compute":
        return f"{proveedor}: ${compute_cost:.2f}/mes de cómputo"
    else:
        return f"{proveedor}: ${total:.2f}/mes total (storage: ${storage_cost:.2f} + compute: ${compute_cost:.2f}) para {gb_datos}GB"

@tool
def analizar_patron_arquitectonico(descripcion: str) -> str:
    """Recomienda patrón de arquitectura de datos según el problema descrito.
    Usar cuando el usuario describa un caso de uso técnico de datos."""
    desc = descripcion.lower()
    if any(w in desc for w in ["tiempo real", "streaming", "eventos", "kafka"]):
        return "Patrón: Lambda Architecture (batch + streaming) o Kappa Architecture (solo streaming). Tecnologías: Kafka + Flink o Spark Streaming."
    elif any(w in desc for w in ["ml", "modelo", "features", "entrenamiento", "machine learning"]):
        return "Patrón: Feature Store + Model Registry. MLflow para experimentos. Feast para feature store. Separa features offline (batch) de online (tiempo real)."
    elif any(w in desc for w in ["histórico", "batch", "diario", "etl", "noche"]):
        return "Patrón: Medallion Architecture con Delta Lake. Batch orquestado con Airflow. Transformaciones con dbt. Particionado por fecha."
    elif any(w in desc for w in ["rag", "documentos", "búsqueda semántica"]):
        return "Patrón: RAG con Qdrant en producción. Chunking con RecursiveCharacterTextSplitter. Embeddings nomic-embed-text. LangGraph para orquestación."
    return "Necesito más contexto: describe latencia requerida, volumen de datos y casos de uso principales."

# ============================================================
# LLM CON TOOLS
# ============================================================

tools = [buscar_documentacion, calcular_coste_almacenamiento, analizar_patron_arquitectonico]

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_retries=0)
llm_con_tools = llm.bind_tools(tools)

# System prompt que guía al modelo en la selección de tools.
# El modelo 8b necesita más orientación que el 70b.
SYSTEM_PROMPT = SystemMessage(content="""Eres un arquitecto de datos senior.
Tienes acceso a estas herramientas:
- buscar_documentacion: para conceptos técnicos (particionado, medallion, estrella, rag, langgraph)
- calcular_coste_almacenamiento: para preguntas de coste con GB y proveedor específicos
- analizar_patron_arquitectonico: para recomendar patrones según un caso de uso descrito

Usa SIEMPRE la herramienta más específica para la pregunta.
Cuando tengas la información necesaria de las tools, responde directamente sin llamar más tools.""")

# ============================================================
# ESTADO CON CONTADOR DE ITERACIONES
# ============================================================
# Añadimos iteraciones al estado para cortar el ciclo
# desde dentro del grafo, más fiable que recursion_limit externo.

class EstadoReAct(TypedDict):
    messages: Annotated[List, add_messages]
    iteraciones: int

# ============================================================
# NODOS
# ============================================================

def agente(state: EstadoReAct) -> dict:
    """Nodo agente con system prompt, tools y circuit breaker."""
    # Cortar si llevamos demasiadas iteraciones
    if state["iteraciones"] >= 5:
        logger.warning("Máximo de iteraciones alcanzado, forzando respuesta final")
        return {
            "messages": [AIMessage(content="He recopilado suficiente información. Basándome en las herramientas consultadas, te proporciono mi recomendación.")],
            "iteraciones": state["iteraciones"]
        }

    # Construir lista de mensajes con system prompt al inicio
    mensajes = [SYSTEM_PROMPT] + state["messages"]

    for intento in range(3):
        try:
            response = llm_con_tools.invoke(mensajes)
            return {
                "messages": [response],
                "iteraciones": state["iteraciones"] + 1
            }
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower():
                wait = 60
                match = re.search(r"try again in (\d+)h(\d+)m([\d\.]+)s", error_str)
                if match:
                    wait = int(match.group(1))*3600 + int(match.group(2))*60 + float(match.group(3))
                wait = min(wait, 90)
                logger.warning(f"Rate limit. Esperando {wait:.0f}s (intento {intento+1}/3)")
                time.sleep(wait)
            elif "400" in error_str and "tool_use_failed" in error_str:
                # El modelo generó una tool call malformada.
                # Devolvemos un AIMessage sin tool_calls para que el grafo termine.
                logger.warning("Tool call malformada (400). Devolviendo respuesta directa.")
                respuesta_directa = llm.invoke(mensajes)
                return {
                    "messages": [respuesta_directa],
                    "iteraciones": state["iteraciones"] + 1
                }
            else:
                logger.error(f"Error no recuperable: {e}")
                raise e

    return {
        "messages": [AIMessage(content="No pude procesar la consulta tras varios reintentos.")],
        "iteraciones": state["iteraciones"]
    }

tool_node = ToolNode(tools)

# ============================================================
# GRAFO
# ============================================================

builder = StateGraph(EstadoReAct)
builder.add_node("agente", agente)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agente")
builder.add_conditional_edges("agente", tools_condition)
builder.add_edge("tools", "agente")

graph = builder.compile()

# ============================================================
# EJECUCIÓN
# ============================================================

def ejecutar_pregunta(pregunta: str):
    print(f"\nPregunta: {pregunta}")
    print("Trazado de ejecución:")
    try:
        resultado = graph.invoke(
            {
                "messages": [HumanMessage(content=pregunta)],
                "iteraciones": 0
            },
            {"recursion_limit": 20}
        )
        for msg in resultado["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  -> tool: {tc['name']} | args: {json.dumps(tc['args'], ensure_ascii=False)[:80]}")
            elif isinstance(msg, ToolMessage):
                print(f"  <- resultado: {msg.content[:80]}...")

        for msg in reversed(resultado["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                print(f"Respuesta final: {msg.content[:400]}...")
                break
    except Exception as e:
        logger.error(f"Error ejecutando el grafo: {e}")
    print("-" * 55)

if __name__ == "__main__":
    print("=" * 55)
    print("AGENTE ReAct CON HERRAMIENTAS (ROBUSTO)")
    print("=" * 55)

    preguntas = [
        "¿Qué es el particionado de tablas y cuándo usarlo?",
        "¿Cuánto me costaría en Snowflake almacenar 500GB, dame el coste total?",
        "Tengo datos de ventas que proceso en batch cada noche y necesito features para ML. ¿Qué arquitectura me recomiendas?"
    ]

    for pregunta in preguntas:
        ejecutar_pregunta(pregunta)
        time.sleep(8)
