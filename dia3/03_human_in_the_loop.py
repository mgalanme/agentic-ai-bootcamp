import time
import logging
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# ============================================================
# Human-in-the-Loop: el patrón más crítico para producción
# ============================================================
# En sistemas de compliance, finanzas o cualquier dominio
# donde el agente puede ejecutar acciones irreversibles,
# necesitas un punto de control humano antes de continuar.
#
# LangGraph implementa esto con dos mecanismos:
# 1. interrupt_before: pausa el grafo ANTES de ejecutar un nodo
# 2. checkpointer: persiste el estado para poder reanudarlo
#
# Como EA: es exactamente un gateway de aprobación en BPMN.
# El proceso se detiene, un humano revisa y aprueba o rechaza.
# El estado persiste entre la pausa y la reanudación.

# ============================================================
# HERRAMIENTAS: acciones con distintos niveles de riesgo
# ============================================================

@tool
def consultar_esquema(tabla: str, entorno: str) -> str:
    """Consulta el esquema de una tabla. Operación de SOLO LECTURA, segura."""
    esquemas = {
        "fact_ventas": "id_venta BIGINT, id_cliente INT, id_producto INT, fecha DATE, importe DECIMAL(10,2), divisa VARCHAR(3)",
        "dim_cliente": "id_cliente INT, nombre VARCHAR(100), segmento VARCHAR(50), pais VARCHAR(3), fecha_alta DATE",
        "dim_producto": "id_producto INT, nombre VARCHAR(200), categoria VARCHAR(50), precio DECIMAL(10,2)"
    }
    esquema = esquemas.get(tabla.lower(), f"Tabla '{tabla}' no encontrada en el catálogo")
    return f"Esquema de {tabla} [{entorno}]: {esquema}"

@tool
def validar_datos(tabla: str, entorno: str) -> str:
    """Valida la calidad de datos de una tabla. Operación de SOLO LECTURA, segura."""
    return f"Validación de {tabla} [{entorno}]: 2.847.291 filas OK | nulos: 0.3% | duplicados: 0 | rango fechas: 2020-01-01 a 2024-12-31"

@tool
def ejecutar_migracion(tabla: str, entorno_origen: str, entorno_destino: str) -> str:
    """ACCIÓN DESTRUCTIVA: migra datos entre entornos.
    Requiere aprobación humana explícita antes de ejecutar."""
    return f"MIGRACIÓN EJECUTADA: {tabla} copiada de {entorno_origen} a {entorno_destino} | filas migradas: 2.847.291 | checksum: OK"

@tool
def ejecutar_truncate(tabla: str, entorno: str) -> str:
    """ACCIÓN DESTRUCTIVA IRREVERSIBLE: borra todos los datos de una tabla.
    Requiere aprobación humana explícita antes de ejecutar."""
    return f"TRUNCATE EJECUTADO: {tabla} en {entorno} | filas eliminadas: 2.847.291"

# ============================================================
# CONFIGURACIÓN
# ============================================================

tools_seguras = [consultar_esquema, validar_datos]
tools_destructivas = [ejecutar_migracion, ejecutar_truncate]
todas_las_tools = tools_seguras + tools_destructivas

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_retries=2)
llm_con_tools = llm.bind_tools(todas_las_tools)

SYSTEM_PROMPT = SystemMessage(content="""Eres un ingeniero de datos que gestiona migraciones.
Herramientas disponibles:
- consultar_esquema: lectura segura del esquema de una tabla
- validar_datos: lectura segura de estadísticas de calidad
- ejecutar_migracion: DESTRUCTIVA - migra datos entre entornos
- ejecutar_truncate: DESTRUCTIVA IRREVERSIBLE - borra todos los datos

Antes de ejecutar cualquier acción destructiva, consulta y valida primero.""")

# ============================================================
# ESTADO
# ============================================================

class EstadoMigracion(TypedDict):
    messages: Annotated[List, add_messages]
    iteraciones: int
    accion_pendiente: str  # descripción de la acción que espera aprobación

# ============================================================
# NODOS
# ============================================================

def agente(state: EstadoMigracion) -> dict:
    """Nodo agente: razona y decide qué tools usar."""
    if state["iteraciones"] >= 6:
        return {
            "messages": [AIMessage(content="He completado el análisis disponible.")],
            "iteraciones": state["iteraciones"],
            "accion_pendiente": ""
        }

    mensajes = [SYSTEM_PROMPT] + state["messages"]
    response = llm_con_tools.invoke(mensajes)

    # Detectar si el agente quiere ejecutar una acción destructiva
    accion = ""
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] in ["ejecutar_migracion", "ejecutar_truncate"]:
                accion = f"Tool: {tc['name']} | Args: {tc['args']}"
                break

    return {
        "messages": [response],
        "iteraciones": state["iteraciones"] + 1,
        "accion_pendiente": accion
    }

tool_node = ToolNode(todas_las_tools)

# ============================================================
# GRAFO CON CHECKPOINT Y HUMAN-IN-THE-LOOP
# ============================================================

builder = StateGraph(EstadoMigracion)
builder.add_node("agente", agente)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agente")
builder.add_conditional_edges("agente", tools_condition)
builder.add_edge("tools", "agente")

# MemorySaver persiste el estado completo del grafo en memoria.
# En producción usarías SqliteSaver o PostgresSaver para
# persistencia entre reinicios del proceso.
checkpointer = MemorySaver()

# interrupt_before=["tools"]: el grafo se pausa ANTES de ejecutar
# el nodo "tools". El humano puede revisar qué tool va a ejecutar
# y decidir si aprueba o rechaza antes de que ocurra.
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"]
)

# ============================================================
# EJECUCIÓN CON FLUJO DE APROBACIÓN
# ============================================================

def ejecutar_con_aprobacion(instruccion: str, thread_id: str):
    """
    Ejecuta el agente con pausa para aprobación humana
    antes de cualquier acción (incluyendo las seguras).
    El humano ve qué va a hacer el agente y puede aprobar o cancelar.
    """
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\nInstrucción: {instruccion}")
    print("=" * 55)

    estado_inicial = {
        "messages": [HumanMessage(content=instruccion)],
        "iteraciones": 0,
        "accion_pendiente": ""
    }

    # Primera ejecución: el agente razona y se detiene antes de ejecutar tools
    print("Agente razonando...")
    events = list(graph.stream(estado_inicial, config=config, stream_mode="values"))

    # Bucle de aprobación: continúa hasta que el grafo termine
    while True:
        # Verificar estado actual del grafo
        state = graph.get_state(config)

        # Si no hay siguiente nodo, el grafo terminó
        if not state.next:
            print("\nGrafo completado.")
            # Mostrar respuesta final
            for msg in reversed(state.values["messages"]):
                if isinstance(msg, AIMessage) and msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                    print(f"\nRespuesta final del agente:\n{msg.content}")
                    break
            break

        # El grafo está pausado antes de ejecutar tools
        # Mostrar qué tool va a ejecutar
        ultimo_msg = state.values["messages"][-1]
        if hasattr(ultimo_msg, "tool_calls") and ultimo_msg.tool_calls:
            for tc in ultimo_msg.tool_calls:
                es_destructiva = tc["name"] in ["ejecutar_migracion", "ejecutar_truncate"]
                nivel = "DESTRUCTIVA" if es_destructiva else "lectura"
                print(f"\nEl agente quiere ejecutar [{nivel}]:")
                print(f"  Tool: {tc['name']}")
                print(f"  Args: {tc['args']}")

                if es_destructiva:
                    print("\nACCIÓN DESTRUCTIVA detectada. Se requiere aprobación explícita.")

        aprobacion = input("\n¿Aprobar ejecución? (s/n): ").strip().lower()

        if aprobacion == "s":
            print("Aprobado. Continuando...")
            # Reanudar el grafo desde donde se pausó
            list(graph.stream(None, config=config, stream_mode="values"))
        else:
            print("Rechazado por el operador. Deteniendo el flujo.")
            break

# ============================================================
# CASOS DE PRUEBA
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("HUMAN-IN-THE-LOOP: GESTIÓN DE MIGRACIONES")
    print("=" * 55)
    print("El grafo se pausa antes de cada acción para aprobación.")
    print("Responde 's' para aprobar o 'n' para rechazar.\n")

    # Caso 1: solo operaciones de lectura (consulta + validación)
    ejecutar_con_aprobacion(
        instruccion="Consulta el esquema de fact_ventas en producción y valida su calidad de datos",
        thread_id="session-lectura-001"
    )

    time.sleep(5)
    print("\n" + "=" * 55)

    # Caso 2: operación destructiva (migración)
    ejecutar_con_aprobacion(
        instruccion="Valida los datos de fact_ventas en desarrollo y luego migra la tabla de desarrollo a producción",
        thread_id="session-migracion-001"
    )
