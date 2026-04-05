"""
ARGOS-FCC · Script 04: Agente AML con LangGraph
================================================
El corazón del Día 5. Un grafo LangGraph con tres nodos
especializados que investiga alertas AML de forma autónoma
y genera borradores de SAR para aprobación humana.

Flujo: detección → investigación (Neo4j + RAG) → SAR draft
       → PAUSA human-in-the-loop → decisión → log inmutable
"""
import os
import json
from datetime import datetime
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_neo4j import Neo4jGraph
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

load_dotenv()

print("ARGOS-FCC · Agente AML con LangGraph")
print("=" * 55)

# ============================================================
# INICIALIZAR COMPONENTES
# ============================================================

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

print("Conectando a Neo4j...")
graph_db = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    enhanced_schema=False
)

print("Conectando a Qdrant RAG regulatorio...")
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Componentes listos\n")

# ============================================================
# ESTADO DEL GRAFO AML
# ============================================================
# El estado es el expediente vivo de la alerta.
# Fluye por todos los nodos acumulando información.
# Como DA: es el DTO del workflow de investigación.

class EstadoAML(TypedDict):
    messages:         Annotated[List, add_messages]
    alerta_id:        str
    cuenta_sospechosa: str
    patron_detectado: str
    score_riesgo:     float
    contexto_neo4j:   str   # resultado del traversal del grafo
    contexto_regulatorio: str  # resultado del RAG
    analisis_agente:  str   # conclusiones del investigador
    borrador_sar:     str   # texto del SAR generado
    decision_humana:  str   # aprobado | rechazado | pendiente
    motivo_decision:  str
    timestamp:        str

# ============================================================
# STRUCTURED OUTPUT: FICHA DE RIESGO
# ============================================================

class FichaRiesgo(BaseModel):
    patron_confirmado: str = Field(
        description="Patrón AML confirmado: structuring, smurfing, layering u otro")
    score_final: float = Field(
        description="Score de riesgo entre 0.0 y 1.0")
    red_flags: List[str] = Field(
        description="Lista de señales de alerta identificadas")
    articulos_aplicables: List[str] = Field(
        description="Artículos regulatorios aplicables (ej: FATF R.20, Art.18 Ley 10/2010)")
    recomendacion: str = Field(
        description="EMITIR_SAR, INVESTIGACION_ADICIONAL o CERRAR_SIN_ACCION")
    justificacion: str = Field(
        description="Justificación técnica y regulatoria de la recomendación")

# ============================================================
# NODO 1: DETECTOR
# Analiza la cuenta y genera la alerta inicial con score.
# ============================================================

def nodo_detector(state: EstadoAML) -> dict:
    """
    Detecta el patrón de riesgo consultando Neo4j.
    Genera score inicial y abre el expediente de alerta.
    """
    cuenta = state["cuenta_sospechosa"]
    patron = state["patron_detectado"]

    # Query Neo4j para contexto de la cuenta
    try:
        resultado = graph_db.query("""
            MATCH (cta:Cuenta {id: $cuenta_id})
            OPTIONAL MATCH (cli:Cliente)-[:TITULAR_DE]->(cta)
            OPTIONAL MATCH (cta)-[:ORIGEN_DE]->(t:Transaccion)
            WITH cta, cli,
                 count(t)          AS n_txs,
                 sum(t.importe)    AS volumen_total,
                 avg(t.importe)    AS importe_medio,
                 collect(DISTINCT t.pais_destino)[0..5] AS paises,
                 collect(DISTINCT t.patron_riesgo)[0..5] AS patrones
            RETURN cta.id         AS cuenta,
                   cta.tipo       AS tipo_cuenta,
                   cli.nombre     AS titular,
                   cli.riesgo     AS riesgo_cliente,
                   cli.pep        AS es_pep,
                   cli.pais       AS pais_cliente,
                   n_txs          AS num_transacciones,
                   round(volumen_total)  AS volumen_eur,
                   round(importe_medio) AS importe_medio_eur,
                   paises, patrones
        """, params={"cuenta_id": cuenta})
    except Exception as e:
        resultado = []

    if resultado:
        r = resultado[0]
        contexto = (
            f"Cuenta: {r.get('cuenta')} ({r.get('tipo_cuenta')})\n"
            f"Titular: {r.get('titular')} | País: {r.get('pais_cliente')} | "
            f"Riesgo: {r.get('riesgo_cliente')} | PEP: {r.get('es_pep')}\n"
            f"Transacciones: {r.get('num_transacciones')} | "
            f"Volumen: {r.get('volumen_eur'):,}€ | "
            f"Importe medio: {r.get('importe_medio_eur'):,}€\n"
            f"Países destino: {r.get('paises')}\n"
            f"Patrones detectados: {r.get('patrones')}"
        )
    else:
        contexto = f"Cuenta {cuenta} - datos del grafo no disponibles"

    # Score inicial basado en el patrón
    score_map = {"structuring": 0.72, "smurfing": 0.68,
                 "layering": 0.85, "normal": 0.15}
    score = score_map.get(patron, 0.50)

    msg = AIMessage(content=f"[DETECTOR] Alerta {state['alerta_id']} generada. "
                             f"Cuenta: {cuenta} | Patrón: {patron} | Score: {score}")

    return {
        "messages":       [msg],
        "contexto_neo4j": contexto,
        "score_riesgo":   score,
    }

# ============================================================
# NODO 2: INVESTIGADOR
# Combina Neo4j (red de entidades) + RAG (regulación)
# para construir el expediente de investigación completo.
# ============================================================

def nodo_investigador(state: EstadoAML) -> dict:
    """
    Investigación profunda: traversal de grafo + RAG regulatorio.
    Produce la ficha de riesgo estructurada con Pydantic.
    """
    cuenta   = state["cuenta_sospechosa"]
    patron   = state["patron_detectado"]
    neo4j_ctx = state["contexto_neo4j"]

    # ── Traversal adicional según el patrón ──
    try:
        if patron == "structuring":
            red = graph_db.query("""
                MATCH (origen:Cuenta {id: $cuenta_id})-[:ORIGEN_DE]->(t:Transaccion)
                WHERE t.patron_riesgo = 'structuring'
                WITH t ORDER BY t.fecha
                RETURN collect({importe: round(t.importe),
                                fecha: t.fecha,
                                destino: t.id_cuenta_destino})[0..8] AS txs
            """, params={"cuenta_id": cuenta})
            detalle_red = f"Transacciones structuring: {json.dumps(red[0]['txs'] if red else [], ensure_ascii=False)}"

        elif patron == "smurfing":
            red = graph_db.query("""
                MATCH (orig:Cuenta)-[:ORIGEN_DE]->(t:Transaccion {patron_riesgo:'smurfing'})
                      -[:DESTINO_A]->(dest:Cuenta {id: $cuenta_id})
                RETURN count(DISTINCT orig) AS n_origenes,
                       collect(DISTINCT orig.id)[0..6] AS origenes,
                       round(sum(t.importe)) AS total_recibido
            """, params={"cuenta_id": cuenta})
            r = red[0] if red else {}
            detalle_red = (f"Smurfing: {r.get('n_origenes',0)} cuentas origen → "
                          f"total {r.get('total_recibido',0):,}€ | "
                          f"Orígenes: {r.get('origenes',[])}")

        elif patron == "layering":
            red = graph_db.query("""
                MATCH path = (c1:Cuenta {id: $cuenta_id})
                    -[:ORIGEN_DE]->(t1:Transaccion {patron_riesgo:'layering'})
                    -[:DESTINO_A]->(c2:Cuenta)
                    -[:ORIGEN_DE]->(t2:Transaccion {patron_riesgo:'layering'})
                    -[:DESTINO_A]->(c3:Cuenta)
                RETURN c1.id AS inicio, c2.id AS intermedio, c3.id AS fin,
                       round(t1.importe) AS imp1, round(t2.importe) AS imp2,
                       t2.pais_destino AS pais_final
                LIMIT 3
            """, params={"cuenta_id": cuenta})
            detalle_red = f"Cadenas layering: {json.dumps(red, ensure_ascii=False)}"
        else:
            detalle_red = "Sin detalle adicional de red"
    except Exception as e:
        detalle_red = f"Error en traversal: {str(e)[:100]}"

    # ── RAG regulatorio ──
    query_rag = f"obligaciones regulatorias para {patron} transferencias bancarias sospechosas SAR"
    docs_reg = retriever.invoke(query_rag)
    contexto_reg = "\n\n".join([
        f"[{d.metadata['fuente']} · {d.metadata['articulo']}]: {d.page_content[:300]}"
        for d in docs_reg
    ])

    # ── Análisis con LLM + Structured Output ──
    structured_llm = llm.with_structured_output(FichaRiesgo)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un analista AML senior de un banco retail español.
Tu tarea es analizar una alerta de posible blanqueo de capitales y producir
una ficha de riesgo estructurada con recomendación regulatoria.

Datos de la cuenta investigada:
{neo4j_ctx}

Detalle de la red transaccional:
{detalle_red}

Marco regulatorio aplicable:
{contexto_reg}

Analiza el patrón '{patron}' y produce la ficha de riesgo."""),
        ("human", "Produce la ficha de riesgo para la alerta {alerta_id}.")
    ])

    chain = prompt | structured_llm
    ficha = chain.invoke({
        "neo4j_ctx":    neo4j_ctx,
        "detalle_red":  detalle_red,
        "contexto_reg": contexto_reg,
        "patron":       patron,
        "alerta_id":    state["alerta_id"]
    })

    analisis = (
        f"Patrón confirmado: {ficha.patron_confirmado}\n"
        f"Score final: {ficha.score_final}\n"
        f"Red flags: {', '.join(ficha.red_flags)}\n"
        f"Artículos aplicables: {', '.join(ficha.articulos_aplicables)}\n"
        f"Recomendación: {ficha.recomendacion}\n"
        f"Justificación: {ficha.justificacion}"
    )

    msg = AIMessage(content=f"[INVESTIGADOR] Ficha completada. "
                             f"Score: {ficha.score_final} | "
                             f"Recomendación: {ficha.recomendacion}")

    return {
        "messages":            [msg],
        "contexto_regulatorio": contexto_reg,
        "analisis_agente":     analisis,
        "score_riesgo":        ficha.score_final,
    }

# ============================================================
# NODO 3: REDACTOR SAR
# Genera el borrador del SAR en formato SEPBLAC.
# Este nodo se ejecuta solo si la recomendación es EMITIR_SAR.
# ============================================================

def nodo_redactor_sar(state: EstadoAML) -> dict:
    """
    Genera el borrador del SAR en formato estructurado.
    Tras este nodo viene el PAUSE para aprobación humana.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres el oficial de cumplimiento de un banco retail español.
Redacta un Suspicious Activity Report (SAR) para el SEPBLAC basándote
en el expediente de investigación proporcionado.

El SAR debe incluir:
1. DATOS DE LA ENTIDAD DECLARANTE
2. DATOS DEL SUJETO INVESTIGADO (cuenta y titular)
3. DESCRIPCIÓN DE LAS OPERACIONES SOSPECHOSAS
4. TIPOLOGÍA AML IDENTIFICADA Y BASE REGULATORIA
5. MEDIDAS ADOPTADAS
6. RECOMENDACIÓN AL SEPBLAC

Sé preciso, técnico y cita los artículos regulatorios aplicables.
Indica claramente que es un BORRADOR PENDIENTE DE APROBACIÓN."""),
        ("human", """Expediente de la alerta {alerta_id}:

ANÁLISIS DEL INVESTIGADOR:
{analisis}

CONTEXTO REGULATORIO:
{regulacion}

Redacta el borrador del SAR.""")
    ])

    chain = prompt | llm | StrOutputParser()
    borrador = chain.invoke({
        "alerta_id": state["alerta_id"],
        "analisis":  state["analisis_agente"],
        "regulacion": state["contexto_regulatorio"][:800]
    })

    msg = AIMessage(content=f"[REDACTOR] Borrador SAR generado para alerta "
                             f"{state['alerta_id']}. "
                             f"PENDIENTE DE APROBACIÓN HUMANA.")

    return {
        "messages":     [msg],
        "borrador_sar": borrador,
    }

# ============================================================
# ROUTING: decide si generar SAR o cerrar sin acción
# ============================================================

def routing_post_investigacion(state: EstadoAML) -> str:
    analisis = state.get("analisis_agente", "")
    score    = state.get("score_riesgo", 0)

    if "EMITIR_SAR" in analisis or score >= 0.70:
        return "redactar_sar"
    elif "INVESTIGACION_ADICIONAL" in analisis:
        return "redactar_sar"  # también genera SAR preliminar
    else:
        return "cerrar_alerta"

def nodo_cerrar_alerta(state: EstadoAML) -> dict:
    msg = AIMessage(content=f"[CIERRE] Alerta {state['alerta_id']} cerrada sin acción. "
                             f"Score {state['score_riesgo']:.2f} por debajo del umbral.")
    return {
        "messages":       [msg],
        "decision_humana": "cerrado_automatico",
        "borrador_sar":    "N/A - Cerrado sin acción",
    }

# ============================================================
# CONSTRUIR EL GRAFO
# ============================================================

builder = StateGraph(EstadoAML)

builder.add_node("detector",    nodo_detector)
builder.add_node("investigador",nodo_investigador)
builder.add_node("redactor_sar",nodo_redactor_sar)
builder.add_node("cerrar",      nodo_cerrar_alerta)

builder.add_edge(START, "detector")
builder.add_edge("detector", "investigador")
builder.add_conditional_edges(
    "investigador",
    routing_post_investigacion,
    {"redactar_sar": "redactor_sar", "cerrar_alerta": "cerrar"}
)
builder.add_edge("redactor_sar", END)
builder.add_edge("cerrar",       END)

# MemorySaver para el checkpoint del human-in-the-loop
checkpointer = MemorySaver()

# interrupt_before redactor_sar: el grafo se pausa
# ANTES de generar el SAR, permitiendo que el analista
# revise el análisis del investigador primero.
grafo_aml = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["redactor_sar"]
)

# ============================================================
# FUNCIÓN DE INVESTIGACIÓN CON HITL
# ============================================================

def investigar_alerta(alerta_id: str, cuenta: str, patron: str):
    """
    Ejecuta el workflow completo de investigación AML
    con pausa para aprobación humana antes del SAR.
    """
    print(f"\n{'='*55}")
    print(f"ALERTA: {alerta_id}")
    print(f"Cuenta sospechosa: {cuenta} | Patrón: {patron}")
    print(f"{'='*55}")

    config = {"configurable": {"thread_id": alerta_id}}

    estado_inicial = {
        "messages":            [HumanMessage(content=f"Investigar alerta {alerta_id}")],
        "alerta_id":           alerta_id,
        "cuenta_sospechosa":   cuenta,
        "patron_detectado":    patron,
        "score_riesgo":        0.0,
        "contexto_neo4j":      "",
        "contexto_regulatorio":"",
        "analisis_agente":     "",
        "borrador_sar":        "",
        "decision_humana":     "pendiente",
        "motivo_decision":     "",
        "timestamp":           datetime.now().isoformat(),
    }

    # ── Fase 1: Detector + Investigador (hasta el PAUSE) ──
    print("\n[1/3] Ejecutando detector e investigador...")
    for event in grafo_aml.stream(estado_inicial, config, stream_mode="values"):
        msgs = event.get("messages", [])
        if msgs:
            ultimo = msgs[-1]
            if hasattr(ultimo, "content") and ultimo.content.startswith("["):
                print(f"  {ultimo.content[:100]}")

    estado = grafo_aml.get_state(config)

    # Si el grafo terminó (cerrado sin acción), no hay HITL
    if not estado.next:
        print("\n  Alerta cerrada automáticamente (score bajo)")
        return None

    # ── Mostrar análisis para revisión humana ──
    analisis = estado.values.get("analisis_agente", "")
    score    = estado.values.get("score_riesgo", 0)

    print(f"\n[2/3] PAUSA - Revisión humana requerida")
    print(f"{'─'*55}")
    print(f"Score de riesgo: {score:.2f}")
    print(f"\nAnálisis del investigador:")
    for linea in analisis.split("\n"):
        print(f"  {linea}")
    print(f"{'─'*55}")

    # ── Decisión humana ──
    print("\nEl agente redactor generará el borrador del SAR.")
    decision = input("¿Proceder con la generación del SAR? (s/n): ").strip().lower()

    if decision != "s":
        print("\n  SAR rechazado por el analista. Alerta archivada.")
        grafo_aml.update_state(config, {
            "decision_humana": "rechazado",
            "motivo_decision": "Rechazado por el analista antes de generar SAR"
        })
        return None

    # ── Fase 2: Redacción del SAR ──
    print("\n[3/3] Generando borrador SAR...")
    for event in grafo_aml.stream(None, config, stream_mode="values"):
        msgs = event.get("messages", [])
        if msgs:
            ultimo = msgs[-1]
            if hasattr(ultimo, "content") and "[REDACTOR]" in ultimo.content:
                print(f"  {ultimo.content}")

    estado_final = grafo_aml.get_state(config)
    borrador = estado_final.values.get("borrador_sar", "")

    print(f"\n{'─'*55}")
    print("BORRADOR SAR (pendiente de firma del oficial de riesgos):")
    print(f"{'─'*55}")
    print(borrador[:1200])
    if len(borrador) > 1200:
        print(f"\n[... +{len(borrador)-1200} caracteres ...]")

    # Segunda aprobación: oficial de riesgos
    print(f"\n{'─'*55}")
    decision2 = input("\n¿Aprobar y archivar el SAR? (s/n): ").strip().lower()

    if decision2 == "s":
        grafo_aml.update_state(config, {
            "decision_humana": "aprobado",
            "motivo_decision":  "Aprobado por el analista de compliance"
        })
        # Guardar expediente
        expediente = {
            **estado_final.values,
            "decision_humana": "aprobado",
            "timestamp_decision": datetime.now().isoformat()
        }
        expediente.pop("messages", None)
        os.makedirs("./data/fcc/expedientes", exist_ok=True)
        ruta = f"./data/fcc/expedientes/{alerta_id}.json"
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(expediente, f, ensure_ascii=False, indent=2)
        print(f"\n  SAR aprobado. Expediente guardado en {ruta}")
        print("  PRÓXIMO PASO: envío al SEPBLAC (requiere firma digital del director)")
    else:
        grafo_aml.update_state(config, {
            "decision_humana": "rechazado",
            "motivo_decision":  "SAR rechazado en segunda revisión"
        })
        print("\n  SAR rechazado en segunda revisión. Devuelto al investigador.")

    return estado_final

# ============================================================
# EJECUTAR CASOS DE PRUEBA
# ============================================================

print("\nSistema AML operativo. Iniciando investigaciones...")
print("(Responde 's' para aprobar o 'n' para rechazar en cada paso)\n")

# Caso 1: Structuring confirmado
investigar_alerta(
    alerta_id="AML-2024-001",
    cuenta="CTA-00053",    # la que detectamos con más txs structuring
    patron="structuring"
)

print("\n" + "="*55)
respuesta = input("\n¿Investigar el caso de Layering? (s/n): ").strip().lower()
if respuesta == "s":
    # Caso 2: Layering con jurisdicción de riesgo
    investigar_alerta(
        alerta_id="AML-2024-002",
        cuenta="CTA-00074",    # inicio de cadena layering hacia LR/VG
        patron="layering"
    )

print("\nDía 5 completado. Sistema AML operativo.")
