"""
ARGOS-FCC · Script 06-03: Simulador completo integrado
=======================================================
El orquestador que cierra el sistema. Recibe cualquier
tipo de alerta FCC y la enruta al módulo correcto.

Este script demuestra el sistema completo funcionando:
AML (LangGraph) + Fraude (structured output) + KYC (CrewAI)
sobre la misma infraestructura compartida.

Como EA: este es el proceso de negocio unificado.
Como DA: todos los módulos leen del mismo grafo Neo4j.
Como AI Architect: cada patrón agéntico está donde aporta
más valor, no donde es más fácil de implementar.
"""
import os
import json
import time
from datetime import datetime
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

print("ARGOS-FCC · Simulador completo integrado")
print("=" * 55)
print("Módulos: AML (LangGraph) · Fraude (structured output) · KYC (CrewAI)")
print("=" * 55)

# ============================================================
# COMPONENTES COMPARTIDOS
# ============================================================

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

print("\nInicializando componentes compartidos...")
graph_db = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    enhanced_schema=False
)

embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True}
)
qdrant_client = QdrantClient(url="http://localhost:6333")
vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="fcc_regulacion",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("Componentes listos\n")

# ============================================================
# STRUCTURED OUTPUT: CLASIFICADOR DE ALERTAS
# ============================================================

class ClasificacionAlerta(BaseModel):
    tipo_modulo: str = Field(
        description="Módulo FCC responsable: AML, FRAUDE o KYC")
    prioridad: str = Field(
        description="CRITICA, ALTA, MEDIA o BAJA")
    motivo: str = Field(
        description="Breve justificación de la clasificación")
    score_inicial: float = Field(
        description="Score de riesgo inicial entre 0.0 y 1.0")

# ============================================================
# ESTADO DEL ORQUESTADOR
# ============================================================

class EstadoOrquestador(TypedDict):
    messages:         Annotated[List, add_messages]
    alerta_id:        str
    tipo_alerta:      str
    datos_alerta:     dict
    clasificacion:    Optional[dict]
    resultado_modulo: str
    timestamp:        str

# ============================================================
# NODO: CLASIFICADOR
# Determina qué módulo debe procesar la alerta.
# ============================================================

def nodo_clasificador(state: EstadoOrquestador) -> dict:
    """
    Clasifica la alerta y determina el módulo FCC responsable.
    Este nodo es el dispatcher del sistema.
    """
    datos = state["datos_alerta"]
    tipo  = state["tipo_alerta"]

    # Clasificación basada en el tipo declarado + análisis LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres el clasificador de alertas del sistema ARGOS-FCC.
Determina qué módulo FCC debe procesar esta alerta:
- AML: patrones de lavado (structuring, smurfing, layering, jurisdicciones FATF)
- FRAUDE: transacción anómala respecto al perfil histórico del cliente
- KYC: revisión de identidad, sanciones, PEP o cambio de perfil de riesgo"""),
        ("human", "Alerta tipo '{tipo}':\n{datos}\n\nClasifica esta alerta.")
    ])

    clasificador = llm.with_structured_output(ClasificacionAlerta)
    chain = prompt | clasificador
    resultado = chain.invoke({
        "tipo": tipo,
        "datos": json.dumps(datos, ensure_ascii=False, indent=2)
    })

    clasificacion = {
        "modulo":       resultado.tipo_modulo,
        "prioridad":    resultado.prioridad,
        "motivo":       resultado.motivo,
        "score_inicial": resultado.score_inicial
    }

    msg = AIMessage(content=f"[CLASIFICADOR] Alerta {state['alerta_id']} → "
                             f"Módulo: {resultado.tipo_modulo} | "
                             f"Prioridad: {resultado.prioridad} | "
                             f"Score: {resultado.score_inicial:.2f}")

    return {
        "messages":    [msg],
        "clasificacion": clasificacion,
    }

# ============================================================
# NODO: MÓDULO FRAUDE (inline)
# Scoring rápido de transacción anómala.
# ============================================================

class ScoringFraude(BaseModel):
    score_fraude: float = Field(description="Score 0.0-1.0")
    nivel_riesgo: str = Field(description="BAJO, MEDIO, ALTO o CRITICO")
    anomalias_detectadas: List[str] = Field(description="Anomalías identificadas")
    accion_recomendada: str = Field(description="APROBAR, REVISAR_MANUAL o BLOQUEAR_TEMPORAL")
    justificacion: str = Field(description="Justificación del scoring")

def nodo_fraude(state: EstadoOrquestador) -> dict:
    """Pipeline de scoring de fraude de alta frecuencia."""
    datos = state["datos_alerta"]
    cuenta = datos.get("id_cuenta_origen", "")

    # Perfil histórico de Neo4j
    try:
        perfil = graph_db.query("""
            MATCH (cta:Cuenta {id: $cid})
            OPTIONAL MATCH (cli:Cliente)-[:TITULAR_DE]->(cta)
            OPTIONAL MATCH (cta)-[:ORIGEN_DE]->(t:Transaccion)
            WITH cli, count(t) AS n_txs,
                 round(avg(t.importe)) AS importe_medio,
                 collect(DISTINCT t.pais_destino)[0..4] AS paises
            RETURN cli.nombre AS titular, cli.riesgo AS riesgo,
                   n_txs, importe_medio, paises
        """, params={"cid": cuenta})
    except:
        perfil = []

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analiza esta transacción contra el perfil histórico. "
                   "Detecta anomalías y asigna score de fraude."),
        ("human", "Transacción: {tx}\n\nPerfil histórico: {perfil}")
    ])

    t0 = time.time()
    scoring_llm = llm.with_structured_output(ScoringFraude)
    resultado = (prompt | scoring_llm).invoke({
        "tx":    json.dumps(datos, ensure_ascii=False),
        "perfil": json.dumps(perfil, ensure_ascii=False)
    })
    latencia = time.time() - t0

    resumen = (
        f"FRAUDE SCORING:\n"
        f"  Score: {resultado.score_fraude:.2f} | Nivel: {resultado.nivel_riesgo}\n"
        f"  Acción: {resultado.accion_recomendada}\n"
        f"  Anomalías: {', '.join(resultado.anomalias_detectadas[:3])}\n"
        f"  Latencia: {latencia:.2f}s"
    )

    msg = AIMessage(content=f"[FRAUDE] Score: {resultado.score_fraude:.2f} | "
                             f"{resultado.nivel_riesgo} | {resultado.accion_recomendada}")
    return {
        "messages":       [msg],
        "resultado_modulo": resumen,
    }

# ============================================================
# NODO: MÓDULO AML (simplificado para el orquestador)
# ============================================================

def nodo_aml(state: EstadoOrquestador) -> dict:
    """
    Investigación AML: traversal Neo4j + RAG regulatorio.
    Versión simplificada para el orquestador integrado.
    Para el flujo completo con HITL usar dia5/04_agente_aml.py
    """
    datos  = state["datos_alerta"]
    cuenta = datos.get("cuenta_sospechosa", "")
    patron = datos.get("patron_detectado", "")

    # Traversal Neo4j
    try:
        red = graph_db.query("""
            MATCH (cta:Cuenta {id: $cid})
            OPTIONAL MATCH (cli:Cliente)-[:TITULAR_DE]->(cta)
            OPTIONAL MATCH (cta)-[:ORIGEN_DE]->(t:Transaccion)
            WITH cli, count(t) AS n_txs,
                 round(sum(t.importe)) AS volumen,
                 collect(DISTINCT t.pais_destino)[0..5] AS paises,
                 collect(DISTINCT t.patron_riesgo)[0..3] AS patrones
            RETURN cli.nombre AS titular, cli.pais AS pais_cliente,
                   cli.riesgo AS riesgo, n_txs, volumen, paises, patrones
        """, params={"cid": cuenta})
    except:
        red = []

    # RAG regulatorio
    docs = retriever.invoke(f"regulación {patron} blanqueo transferencias sospechosas")
    marco_reg = " | ".join([
        f"{d.metadata['fuente']} {d.metadata['articulo']}"
        for d in docs
    ])

    # Análisis LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres analista AML senior. Analiza el patrón de riesgo y "
                   "emite una recomendación regulatoriamente fundamentada."),
        ("human", "Cuenta: {cuenta}\nPatrón: {patron}\n"
                  "Datos Neo4j: {red}\nMarco regulatorio: {reg}\n\n"
                  "Emite recomendación concisa (máx 100 palabras).")
    ])

    recomendacion = (prompt | llm).invoke({
        "cuenta": cuenta,
        "patron": patron,
        "red":    json.dumps(red, ensure_ascii=False),
        "reg":    marco_reg
    }).content

    resumen = (
        f"AML ANÁLISIS:\n"
        f"  Cuenta: {cuenta} | Patrón: {patron}\n"
        f"  Marco regulatorio: {marco_reg}\n"
        f"  Recomendación: {recomendacion[:150]}...\n"
        f"  → Para flujo completo con HITL: python dia5/04_agente_aml.py"
    )

    msg = AIMessage(content=f"[AML] Análisis completado para {cuenta} | {patron}")
    return {
        "messages":       [msg],
        "resultado_modulo": resumen,
    }

# ============================================================
# NODO: MÓDULO KYC (simplificado para el orquestador)
# ============================================================

def nodo_kyc(state: EstadoOrquestador) -> dict:
    """
    Due diligence KYC básico.
    Para el flujo completo con crew usar dia6/02_crew_kyc.py
    """
    datos      = state["datos_alerta"]
    id_cliente = datos.get("id_cliente", "")

    FATF_RIESGO = ["IR", "KP", "MM", "PA", "VG", "KY", "LR", "SY"]

    try:
        perfil = graph_db.query("""
            MATCH (cli:Cliente {id: $id})
            OPTIONAL MATCH (cli)-[:TITULAR_DE]->(cta:Cuenta)
            OPTIONAL MATCH (cta)-[:ORIGEN_DE]->(t:Transaccion)
            WITH cli, count(t) AS n_txs, round(sum(t.importe)) AS volumen
            RETURN cli.nombre AS nombre, cli.pais AS pais,
                   cli.riesgo AS riesgo, cli.pep AS es_pep,
                   n_txs, volumen
        """, params={"id": id_cliente})
    except:
        perfil = []

    if perfil:
        p = perfil[0]
        pais = p.get("pais", "")
        en_lista = pais in FATF_RIESGO

        # RAG regulatorio
        docs = retriever.invoke("diligencia debida PEP jurisdicción riesgo sanciones")
        marco_reg = " | ".join([
            f"{d.metadata['fuente']} {d.metadata['articulo']}" for d in docs
        ])

        nivel = "ALTO" if (en_lista or p.get("es_pep")) else p.get("riesgo", "medio").upper()

        resumen = (
            f"KYC DUE DILIGENCE:\n"
            f"  Cliente: {id_cliente} | {p.get('nombre')}\n"
            f"  País: {pais} {'⚠ LISTA FATF' if en_lista else '✓'} | PEP: {p.get('es_pep')}\n"
            f"  Txs: {p.get('n_txs')} | Volumen: {p.get('volumen'):,}€\n"
            f"  Nivel riesgo: {nivel}\n"
            f"  Marco regulatorio: {marco_reg}\n"
            f"  → Para crew completa: python dia6/02_crew_kyc.py"
        )
    else:
        resumen = f"KYC: Cliente {id_cliente} no encontrado en el sistema."

    msg = AIMessage(content=f"[KYC] Due diligence completado para {id_cliente}")
    return {
        "messages":       [msg],
        "resultado_modulo": resumen,
    }

# ============================================================
# ROUTING: DESPACHA AL MÓDULO CORRECTO
# ============================================================

def routing_modulo(state: EstadoOrquestador) -> str:
    clasificacion = state.get("clasificacion", {})
    modulo = clasificacion.get("modulo", "AML").upper()
    if modulo == "FRAUDE":
        return "fraude"
    elif modulo == "KYC":
        return "kyc"
    else:
        return "aml"

# ============================================================
# CONSTRUIR EL GRAFO ORQUESTADOR
# ============================================================

builder = StateGraph(EstadoOrquestador)

builder.add_node("clasificador", nodo_clasificador)
builder.add_node("aml",          nodo_aml)
builder.add_node("fraude",       nodo_fraude)
builder.add_node("kyc",          nodo_kyc)

builder.add_edge(START, "clasificador")
builder.add_conditional_edges(
    "clasificador",
    routing_modulo,
    {"aml": "aml", "fraude": "fraude", "kyc": "kyc"}
)
builder.add_edge("aml",    END)
builder.add_edge("fraude", END)
builder.add_edge("kyc",    END)

checkpointer = MemorySaver()
orquestador  = builder.compile(checkpointer=checkpointer)

# ============================================================
# FUNCIÓN DE EJECUCIÓN
# ============================================================

def procesar_alerta(alerta_id: str, tipo: str, datos: dict):
    print(f"\n{'='*55}")
    print(f"ALERTA: {alerta_id} | Tipo declarado: {tipo}")
    print(f"{'='*55}")

    config = {"configurable": {"thread_id": alerta_id}}

    estado_inicial = {
        "messages":         [HumanMessage(content=f"Procesar alerta {alerta_id}")],
        "alerta_id":        alerta_id,
        "tipo_alerta":      tipo,
        "datos_alerta":     datos,
        "clasificacion":    None,
        "resultado_modulo": "",
        "timestamp":        datetime.now().isoformat(),
    }

    for event in orquestador.stream(estado_inicial, config, stream_mode="values"):
        msgs = event.get("messages", [])
        if msgs:
            ultimo = msgs[-1]
            if hasattr(ultimo, "content") and ultimo.content.startswith("["):
                print(f"  {ultimo.content}")

    estado_final = orquestador.get_state(config)
    clasificacion = estado_final.values.get("clasificacion", {})
    resultado     = estado_final.values.get("resultado_modulo", "")

    print(f"\nClasificación: {clasificacion.get('modulo')} | "
          f"Prioridad: {clasificacion.get('prioridad')} | "
          f"Score: {clasificacion.get('score_inicial', 0):.2f}")
    print(f"\n{resultado}")

# ============================================================
# ALERTAS DE PRUEBA: LOS TRES MÓDULOS
# ============================================================

# Alerta 1: AML - layering detectado en el Día 5
procesar_alerta(
    alerta_id="ORQ-2024-001",
    tipo="transaccional_aml",
    datos={
        "cuenta_sospechosa": "CTA-00074",
        "patron_detectado":  "layering",
        "descripcion":       "Cadena de transferencias hacia jurisdicciones FATF",
        "score_previo":      0.85
    }
)

# Alerta 2: Fraude - transacción anómala
procesar_alerta(
    alerta_id="ORQ-2024-002",
    tipo="transaccional_fraude",
    datos={
        "id": "TX-99001",
        "id_cuenta_origen":  "CTA-00015",
        "id_cuenta_destino": "CTA-00090",
        "importe":           35_000.0,
        "divisa":            "EUR",
        "canal":             "online",
        "pais_origen":       "ES",
        "pais_destino":      "RU",
        "tipo":              "transferencia",
        "descripcion":       "Inversión urgente",
        "fecha":             datetime.now().isoformat()
    }
)

# Alerta 3: KYC - revisión de cliente de alto riesgo
procesar_alerta(
    alerta_id="ORQ-2024-003",
    tipo="revision_kyc",
    datos={
        "id_cliente":  "CLI-0013",   # cliente con pais=KP (Corea del Norte)
        "motivo":      "alerta_aml_relacionada",
        "descripcion": "Cliente con transacciones hacia jurisdicción lista negra FATF"
    }
)

print(f"\n{'='*55}")
print("ARGOS-FCC · Simulador completo operativo")
print("Los tres módulos FCC funcionando sobre infraestructura compartida")
print(f"{'='*55}")
