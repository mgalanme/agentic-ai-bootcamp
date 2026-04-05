from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ============================================================
# EL ESTADO: el esquema de datos que fluye por el grafo
# ============================================================
# TypedDict define el contrato de datos del sistema.
# Como DA: es exactamente el esquema de tu tabla de estado.
# Todos los nodos leen y escriben sobre este objeto.
#
# Annotated[List, add_messages] es especial:
# en lugar de sobreescribir la lista en cada nodo,
# add_messages ACUMULA los mensajes nuevos sobre los existentes.
# Es el mecanismo que da memoria conversacional al grafo.

class EstadoAgente(TypedDict):
    messages: Annotated[List, add_messages]
    iteracion: int
    calidad: str  # "buena", "mejorable", "mala", "pendiente"

# ============================================================
# LOS NODOS: funciones que transforman el estado
# ============================================================
# Cada nodo recibe el estado completo y devuelve
# un diccionario con SOLO las claves que modifica.
# Las claves no devueltas mantienen su valor anterior.
# Como DA: cada nodo es una transformación del pipeline,
# igual que una capa de dbt que modifica solo sus columnas.

def generar_respuesta(state: EstadoAgente) -> dict:
    """Nodo 1: genera una respuesta técnica al usuario."""
    system = SystemMessage(content="""Eres un arquitecto de datos senior.
Responde con precisión técnica y ejemplos concretos.""")

    response = llm.invoke([system] + state["messages"])

    return {
        "messages": [response],
        "iteracion": state["iteracion"] + 1,
        "calidad": "pendiente"
    }

def evaluar_calidad(state: EstadoAgente) -> dict:
    """Nodo 2: evalúa si la respuesta es suficientemente buena."""
    ultima_respuesta = state["messages"][-1].content

    prompt = f"""Evalúa si esta respuesta técnica sobre arquitectura de datos
es completa, precisa y tiene ejemplos concretos.
Responde SOLO con una de estas palabras exactas: buena, mejorable, mala

Respuesta a evaluar:
{ultima_respuesta[:500]}"""

    evaluacion = llm.invoke([HumanMessage(content=prompt)])
    calidad_raw = evaluacion.content.strip().lower()

    if "buena" in calidad_raw:
        calidad = "buena"
    elif "mejorable" in calidad_raw:
        calidad = "mejorable"
    else:
        calidad = "mala"

    print(f"  [Evaluador] iteracion={state['iteracion']} calidad={calidad}")
    return {"calidad": calidad}

def refinar_respuesta(state: EstadoAgente) -> dict:
    """Nodo 3: mejora la respuesta si no es suficientemente buena."""
    system = SystemMessage(content="""Eres un revisor de arquitectura de datos.
La respuesta anterior fue evaluada como mejorable o mala.
Genera una versión mejorada: más completa, con ejemplos concretos
y mejores prácticas de la industria.""")

    response = llm.invoke([system] + state["messages"])

    return {
        "messages": [response],
        "iteracion": state["iteracion"] + 1,
        "calidad": "pendiente"
    }

# ============================================================
# EL ROUTING: lógica de decisión sobre el siguiente nodo
# ============================================================
# Esta función es el cerebro del grafo.
# Recibe el estado y devuelve el nombre del siguiente nodo.
# Como EA: es exactamente un gateway de decisión en BPMN.

def decidir_siguiente(state: EstadoAgente) -> str:
    if state["iteracion"] >= 3:
        print("  [Router] Máximo de iteraciones alcanzado, finalizando")
        return "fin"
    if state["calidad"] == "buena":
        print("  [Router] Calidad buena, finalizando")
        return "fin"
    else:
        print(f"  [Router] Calidad {state['calidad']}, refinando")
        return "refinar"

# ============================================================
# CONSTRUIR EL GRAFO
# ============================================================
# StateGraph recibe el esquema del estado como parámetro.
# Esto es lo que diferencia LangGraph de un grafo genérico:
# el estado es tipado y fluye por todos los nodos.

builder = StateGraph(EstadoAgente)

# Registrar los nodos (nombre -> función)
builder.add_node("generar", generar_respuesta)
builder.add_node("evaluar", evaluar_calidad)
builder.add_node("refinar", refinar_respuesta)

# Edges fijos: siempre van de A a B sin condición
builder.add_edge(START, "generar")
builder.add_edge("generar", "evaluar")
builder.add_edge("refinar", "evaluar")

# Edge condicional: desde "evaluar" decide a dónde ir
# según lo que devuelve la función decidir_siguiente
builder.add_conditional_edges(
    "evaluar",
    decidir_siguiente,
    {
        "fin": END,
        "refinar": "refinar"
    }
)

# Compilar: convierte el builder en un grafo ejecutable
graph = builder.compile()

# ============================================================
# EJECUTAR EL GRAFO
# ============================================================
print("=" * 55)
print("GRAFO CON AUTO-REFINAMIENTO")
print("=" * 55)

preguntas = [
    "¿Cuándo debería usar particionado de tablas en un Data Warehouse?",
    "¿Qué diferencia hay entre un modelo en estrella y en copo de nieve?"
]

for pregunta in preguntas:
    print(f"\nPregunta: {pregunta}")

    estado_inicial = {
        "messages": [HumanMessage(content=pregunta)],
        "iteracion": 0,
        "calidad": "pendiente"
    }

    resultado = graph.invoke(estado_inicial)

    print(f"Iteraciones: {resultado['iteracion']}")
    print(f"Calidad final: {resultado['calidad']}")
    print(f"Respuesta:\n{resultado['messages'][-1].content[:400]}...")
    print("-" * 55)
