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

# Configura logging: pon INFO si quieres ver reintentos, WARNING para ciertos mensajes y ERROR para salida limpia
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# ============================================================
# HERRAMIENTA MEJORADA: buscar_documentacion con extracción de keywords
# ============================================================
def extraer_keywords(query: str) -> List[str]:
    """Extrae palabras clave relevantes de la consulta."""
    query_lower = query.lower()
    keywords = []
    for term in ["particionado", "medallion", "estrella", "embeddings", "rag", "langgraph", "feature store", "mlflow"]:
        if term in query_lower:
            keywords.append(term)
    return keywords if keywords else ["arquitectura"]

@tool
def buscar_documentacion(query: str) -> str:
    """Busca información técnica sobre arquitectura de datos. Úsala para conceptos como particionado, medallion, estrella, embeddings, RAG, LangGraph."""
    # Mapeo de términos clave a su documentación
    docs = {
        "particionado": "El particionado de tablas divide una tabla en segmentos físicos por una columna (ej. fecha, región). Mejora el rendimiento de queries que filtran por esa columna (partition pruning). En Snowflake se llama clustering, en BigQuery partitioning. Úsalo cuando las consultas filtren habitualmente por la misma columna.",
        "medallion": "Arquitectura Medallion: Bronze (raw inmutable), Silver (curado con reglas de negocio), Gold (agregado para BI y ML). Estándar en Data Lakehouses modernos.",
        "estrella": "Modelo estrella: tabla de hechos central + dimensiones desnormalizadas. Optimizado para OLAP. Copo de nieve normaliza dimensiones: menos redundancia pero más JOINs.",
        "embeddings": "Embeddings: vectores de alta dimensión que representan significado semántico. Textos similares tienen vectores cercanos medidos por cosine similarity.",
        "rag": "RAG (Retrieval-Augmented Generation): combina un retriever (vector DB) con un LLM. El retriever busca chunks similares a la pregunta, el LLM genera respuesta con ese contexto. Reduce alucinaciones.",
        "langgraph": "LangGraph: biblioteca para construir agentes como grafos de estado. State=esquema, Nodes=transformaciones, Edges=conexiones condicionales. Soporta ciclos, memoria persistente y human-in-the-loop.",
        "feature store": "Feature Store: repositorio centralizado de features para ML. Permite reutilización y consistencia entre entrenamiento (offline) e inferencia (online). Ejemplos: Feast, Tecton, Vertex AI Feature Store.",
        "mlflow": "MLflow: plataforma open source para gestionar el ciclo de vida del ML: tracking de experimentos, empaquetado de modelos, despliegue."
    }
    keywords = extraer_keywords(query)
    for kw in keywords:
        if kw in docs:
            return docs[kw]
    # Si no coincide ninguna keyword, devolver los términos válidos
    return f"No encontré información para '{query}'. Términos válidos: particionado, medallion, estrella, embeddings, rag, langgraph, feature store, mlflow."

@tool
def calcular_coste_almacenamiento(gb_datos: float, proveedor: str, tipo: str) -> str:
    """Calcula coste mensual de almacenamiento. proveedor: snowflake, bigquery, redshift, s3. tipo: storage, compute, total."""
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
    """Recomienda patrón de arquitectura de datos según el problema descrito."""
    desc = descripcion.lower()
    if any(w in desc for w in ["tiempo real", "streaming", "eventos", "kafka"]):
        return "Patrón recomendado: Lambda Architecture (batch + streaming) o Kappa Architecture (solo streaming). Tecnologías: Kafka + Flink o Spark Streaming."
    elif any(w in desc for w in ["ml", "modelo", "features", "entrenamiento", "machine learning"]):
        return "Patrón recomendado: Feature Store + Model Registry. MLflow para tracking de experimentos. Feast para feature store (offline y online). Separa features offline (batch) de online (tiempo real)."
    elif any(w in desc for w in ["histórico", "batch", "diario", "etl", "noche"]):
        return "Patrón recomendado: Medallion Architecture con Delta Lake. Batch orquestado con Airflow. Transformaciones con dbt. Particionado por fecha para consultas eficientes."
    elif any(w in desc for w in ["rag", "documentos", "búsqueda semántica"]):
        return "Patrón recomendado: RAG con Qdrant (producción) o Chroma (desarrollo). Chunking con RecursiveCharacterTextSplitter. Embeddings con nomic-embed-text. Orquestación con LangGraph."
    return "Necesito más contexto: describe latencia requerida, volumen de datos y casos de uso principales para una recomendación precisa."

# ============================================================
# LLM CON TOOLS
# ============================================================
tools = [buscar_documentacion, calcular_coste_almacenamiento, analizar_patron_arquitectonico]

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_retries=0)
llm_con_tools = llm.bind_tools(tools)

# System prompt mejorado: exige integrar la información de las tools en la respuesta final
SYSTEM_PROMPT = SystemMessage(content="""Eres un arquitecto de datos senior. Responde SIEMPRE en el mismo idioma que la pregunta del usuario (ej. si preguntan en español, responde en español).

Tienes acceso a estas herramientas:
- buscar_documentacion: para conceptos técnicos (particionado, medallion, estrella, rag, langgraph, feature store, mlflow)
- calcular_coste_almacenamiento: para preguntas de coste con GB y proveedor específicos
- analizar_patron_arquitectonico: para recomendar patrones según un caso de uso

REGLAS OBLIGATORIAS:
1. Antes de responder, DEBES llamar a la herramienta más específica para la pregunta.
2. Cuando recibas el resultado de la herramienta, DEBES usarlo TEXTUALMENTE como base de tu respuesta final. No inventes información alternativa.
3. Si usaste analizar_patron_arquitectonico, tu respuesta final debe comenzar con "La arquitectura recomendada es: " seguida del resultado de la herramienta.
4. Si usaste calcular_coste_almacenamiento, tu respuesta final debe ser el valor numérico devuelto (ej. "El coste total es X").
5. Si usaste buscar_documentacion, incorpora la definición exacta en tu respuesta.
6. No respondas directamente sin usar herramientas, a menos que la pregunta sea un saludo o agradecimiento.
7. Si después de usar una herramienta aún necesitas más información, puedes llamar a otra herramienta, pero limita a un máximo de 2 herramientas por pregunta.
8. Cuando consideres que ya tienes suficiente información, responde de forma concisa y útil, citando la fuente (la herramienta usada).
""")

# ============================================================
# ESTADO CON CONTADOR DE ITERACIONES
# ============================================================
class EstadoReAct(TypedDict):
    messages: Annotated[List, add_messages]
    iteraciones: int

# ============================================================
# NODO AGENTE CON MANEJO ROBUSTO DE ERRORES Y FALLBACK
# ============================================================
def agente(state: EstadoReAct) -> dict:
    if state["iteraciones"] >= 5:
        logger.warning("Máximo de iteraciones alcanzado, forzando respuesta final")
        return {
            "messages": [AIMessage(content="He recopilado suficiente información. Basándome en las herramientas consultadas, te proporciono mi respuesta.")],
            "iteraciones": state["iteraciones"]
        }

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
                # Extraer parámetros mediante regex y llamar directamente a la herramienta
                logger.warning("Tool call malformada, extrayendo parámetros manualmente")
                # Extraer la herramienta y argumentos del mensaje de error (última generación fallida)
                fallback_match = re.search(r"<function=(\w+)>(.*?)</function>", error_str, re.DOTALL)
                if fallback_match:
                    tool_name = fallback_match.group(1)
                    args_str = fallback_match.group(2)
                    try:
                        args = json.loads(args_str)
                        # Validar y corregir tipos
                        if tool_name == "calcular_coste_almacenamiento":
                            args["gb_datos"] = float(args.get("gb_datos", 100))
                            args["proveedor"] = args.get("proveedor", "snowflake").lower()
                            args["tipo"] = args.get("tipo", "total")
                            tool_result = calcular_coste_almacenamiento(**args)
                        elif tool_name == "buscar_documentacion":
                            tool_result = buscar_documentacion(args.get("query", "arquitectura"))
                        elif tool_name == "analizar_patron_arquitectonico":
                            tool_result = analizar_patron_arquitectonico(args.get("descripcion", ""))
                        else:
                            raise ValueError("Herramienta no soportada")
                        # Crear mensajes de tool call y tool result
                        fake_aim = AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": "fallback"}])
                        tool_msg = ToolMessage(content=tool_result, tool_call_id="fallback")
                        return {
                            "messages": [fake_aim, tool_msg],
                            "iteraciones": state["iteraciones"] + 1
                        }
                    except Exception as parse_err:
                        logger.error(f"Fallo en fallback manual: {parse_err}")
                        # En caso de error, responder directamente con el LLM sin tools
                        respuesta_directa = llm.invoke(mensajes)
                        return {
                            "messages": [respuesta_directa],
                            "iteraciones": state["iteraciones"] + 1
                        }
                else:
                    # No se pudo extraer, responder directamente
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
                print(f"  <- resultado: {msg.content[:100]}...")
        # Extraer respuesta final
        for msg in reversed(resultado["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                print(f"Respuesta final:\n{msg.content}")
                break
    except Exception as e:
        logger.error(f"Error ejecutando el grafo: {e}")
    print("-" * 55)

if __name__ == "__main__":
    print("=" * 55)
    print("AGENTE ReAct 10/10 - CON MEJORAS INTEGRALES")
    print("=" * 55)

    preguntas = [
        "¿Qué es el particionado de tablas y cuándo usarlo?",
        "¿Cuánto me costaría en Snowflake almacenar 500GB, dame el coste total?",
        "Tengo datos de ventas que proceso en batch cada noche y necesito features para ML. ¿Qué arquitectura me recomiendas?"
    ]

    for pregunta in preguntas:
        ejecutar_pregunta(pregunta)
        time.sleep(8)  # Pequeña pausa entre preguntas para evitar rate limits
