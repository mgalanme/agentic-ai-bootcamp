"""
ARGOS-FCC · Script 06-01: Módulo de Fraude Transaccional
=========================================================
Agente de scoring de alta frecuencia usando LangChain
structured output con Pydantic.

Patrón arquitectónico: pipeline lineal sin ciclos.
Transacción entra → score tipado sale → <2s latencia.

Como AI Architect: cuando el flujo es lineal y la velocidad
es crítica, structured output directo es mejor que LangGraph.
LangGraph añade overhead de estado que aquí no aporta valor.
"""
import os
import json
import time
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph

load_dotenv()

print("ARGOS-FCC · Módulo Fraude Transaccional")
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
print("Componentes listos\n")

# ============================================================
# STRUCTURED OUTPUT: SCORING DE FRAUDE
# ============================================================
# Este es el contrato de datos del módulo de fraude.
# Como DA: es el esquema de la tabla de alertas de fraude.
# Como AI Architect: el LLM produce este objeto tipado,
# no texto libre que habría que parsear después.

class ScoringFraude(BaseModel):
    score_fraude: float = Field(
        description="Score de riesgo de fraude entre 0.0 (normal) y 1.0 (fraude seguro)")
    nivel_riesgo: str = Field(
        description="BAJO, MEDIO, ALTO o CRITICO")
    anomalias_detectadas: List[str] = Field(
        description="Lista de anomalías identificadas respecto al perfil histórico")
    patron_fraude: str = Field(
        description="Tipo de fraude sospechado: cuenta_comprometida, fraude_identidad, "
                    "uso_inusual, transaccion_normal u otro")
    accion_recomendada: str = Field(
        description="APROBAR, REVISAR_MANUAL, BLOQUEAR_TEMPORAL o BLOQUEAR_DEFINITIVO")
    requiere_hitl: bool = Field(
        description="True si requiere revisión humana antes de ejecutar la acción")
    justificacion: str = Field(
        description="Explicación concisa de por qué se asignó ese score y acción")
    confianza: float = Field(
        description="Confianza del modelo en su scoring entre 0.0 y 1.0")

# ============================================================
# RECUPERAR PERFIL HISTÓRICO DEL CLIENTE DESDE NEO4J
# ============================================================

def obtener_perfil_cliente(id_cuenta: str) -> dict:
    """
    Consulta Neo4j para obtener el perfil histórico de la cuenta:
    importe medio, canales habituales, países frecuentes, horarios.
    Este perfil es la baseline contra la que se compara la transacción.
    """
    try:
        resultado = graph_db.query("""
            MATCH (cta:Cuenta {id: $cuenta_id})
            OPTIONAL MATCH (cli:Cliente)-[:TITULAR_DE]->(cta)
            OPTIONAL MATCH (cta)-[:ORIGEN_DE]->(t:Transaccion)
            WITH cta, cli,
                 count(t)                          AS n_txs,
                 avg(t.importe)                    AS importe_medio,
                 max(t.importe)                    AS importe_max,
                 min(t.importe)                    AS importe_min,
                 collect(DISTINCT t.canal)[0..4]   AS canales_habituales,
                 collect(DISTINCT t.pais_destino)[0..5] AS paises_frecuentes,
                 collect(DISTINCT t.tipo)[0..4]    AS tipos_habituales
            RETURN cta.id          AS cuenta,
                   cta.tipo        AS tipo_cuenta,
                   cli.nombre      AS titular,
                   cli.riesgo      AS riesgo_cliente,
                   cli.pep         AS es_pep,
                   cli.pais        AS pais_cliente,
                   n_txs           AS num_transacciones_historicas,
                   round(coalesce(importe_medio, 0)) AS importe_medio_eur,
                   round(coalesce(importe_max, 0))   AS importe_max_historico,
                   round(coalesce(importe_min, 0))   AS importe_min_historico,
                   canales_habituales,
                   paises_frecuentes,
                   tipos_habituales
        """, params={"cuenta_id": id_cuenta})
        return resultado[0] if resultado else {}
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# PIPELINE DE SCORING: EL CORAZÓN DEL MÓDULO
# ============================================================

# El LLM con structured output tipado
llm_scoring = llm.with_structured_output(ScoringFraude)

prompt_scoring = ChatPromptTemplate.from_messages([
    ("system", """Eres un sistema experto en detección de fraude bancario.
Analiza la transacción entrante comparándola con el perfil histórico del cliente.
Detecta anomalías y asigna un score de fraude preciso.

Criterios de scoring:
- CRITICO (0.85-1.0): importe >5x la media histórica, país nunca visto, canal inusual simultáneo
- ALTO (0.65-0.84): importe >3x la media, país de riesgo, cambio brusco de comportamiento
- MEDIO (0.35-0.64): importe >2x la media o alguna anomalía aislada
- BAJO (0.0-0.34): transacción consistente con el perfil histórico

Reglas de acción:
- BLOQUEAR_TEMPORAL: score >= 0.85 o país lista negra FATF
- REVISAR_MANUAL: score >= 0.55 o cliente PEP
- APROBAR: score < 0.35 y sin anomalías críticas"""),
    ("human", """TRANSACCIÓN ENTRANTE:
{transaccion}

PERFIL HISTÓRICO DE LA CUENTA:
{perfil}

Produce el scoring de fraude.""")
])

# Pipeline lineal: prompt | llm_scoring
# Sin ciclos, sin estado, máxima velocidad
pipeline_scoring = prompt_scoring | llm_scoring


def analizar_transaccion(transaccion: dict) -> tuple[ScoringFraude, float]:
    """
    Ejecuta el pipeline de scoring sobre una transacción.
    Retorna el scoring y el tiempo de ejecución en segundos.
    """
    t0 = time.time()

    perfil = obtener_perfil_cliente(transaccion["id_cuenta_origen"])

    resultado = pipeline_scoring.invoke({
        "transaccion": json.dumps(transaccion, ensure_ascii=False, indent=2),
        "perfil": json.dumps(perfil, ensure_ascii=False, indent=2)
    })

    latencia = time.time() - t0
    return resultado, latencia


def mostrar_resultado(tx: dict, scoring: ScoringFraude, latencia: float):
    """Muestra el resultado del scoring de forma legible."""
    colores = {"BAJO": "✓", "MEDIO": "⚠", "ALTO": "⚠⚠", "CRITICO": "✗✗"}
    icono = colores.get(scoring.nivel_riesgo, "?")

    print(f"\n{'─'*55}")
    print(f"Transacción: {tx['id']} | Cuenta: {tx['id_cuenta_origen']}")
    print(f"Importe: {tx['importe']:,.2f}€ | Canal: {tx['canal']} | País: {tx['pais_destino']}")
    print(f"{'─'*55}")
    print(f"Score fraude:   {scoring.score_fraude:.2f} {icono} {scoring.nivel_riesgo}")
    print(f"Patrón:         {scoring.patron_fraude}")
    print(f"Acción:         {scoring.accion_recomendada}")
    print(f"Requiere HITL:  {'Sí' if scoring.requiere_hitl else 'No'}")
    print(f"Confianza:      {scoring.confianza:.2f}")
    print(f"Latencia:       {latencia:.2f}s")
    if scoring.anomalias_detectadas:
        print(f"Anomalías:")
        for a in scoring.anomalias_detectadas:
            print(f"  · {a}")
    print(f"Justificación:  {scoring.justificacion[:120]}...")


# ============================================================
# CASOS DE PRUEBA
# ============================================================
# Usamos cuentas reales del grafo Neo4j del Día 5.
# Cada caso representa un tipo de fraude distinto.

casos = [
    {
        # Caso 1: transacción normal dentro del perfil
        "id": "TX-FRAUDE-001",
        "id_cuenta_origen": "CTA-00001",
        "id_cuenta_destino": "CTA-00010",
        "importe": 450.00,
        "divisa": "EUR",
        "fecha": datetime.now().isoformat(),
        "canal": "online",
        "pais_origen": "ES",
        "pais_destino": "ES",
        "tipo": "transferencia",
        "descripcion": "Pago alquiler mensual"
    },
    {
        # Caso 2: importe muy superior a la media histórica
        "id": "TX-FRAUDE-002",
        "id_cuenta_origen": "CTA-00015",
        "id_cuenta_destino": "CTA-00080",
        "importe": 48_500.00,
        "divisa": "EUR",
        "fecha": datetime.now().isoformat(),
        "canal": "online",
        "pais_origen": "ES",
        "pais_destino": "RU",
        "tipo": "transferencia",
        "descripcion": "Inversión"
    },
    {
        # Caso 3: país de lista negra FATF + importe elevado
        "id": "TX-FRAUDE-003",
        "id_cuenta_origen": "CTA-00033",
        "id_cuenta_destino": "CTA-00099",
        "importe": 9_750.00,
        "divisa": "EUR",
        "fecha": datetime.now().isoformat(),
        "canal": "banca_movil",
        "pais_origen": "ES",
        "pais_destino": "IR",  # Irán: lista negra FATF
        "tipo": "transferencia",
        "descripcion": "Servicios profesionales"
    },
    {
        # Caso 4: cuenta con patrón smurfing previo + nuevo movimiento
        "id": "TX-FRAUDE-004",
        "id_cuenta_origen": "CTA-00053",  # cuenta con historial structuring
        "id_cuenta_destino": "CTA-00070",
        "importe": 9_200.00,
        "divisa": "EUR",
        "fecha": datetime.now().isoformat(),
        "canal": "oficina",
        "pais_origen": "ES",
        "pais_destino": "PA",  # Panamá: jurisdicción de riesgo
        "tipo": "transferencia",
        "descripcion": "Pago servicios"
    },
]

print("Analizando transacciones...\n")
resultados = []

for tx in casos:
    scoring, latencia = analizar_transaccion(tx)
    mostrar_resultado(tx, scoring, latencia)
    resultados.append({
        "tx_id": tx["id"],
        "importe": tx["importe"],
        "pais_destino": tx["pais_destino"],
        "score": scoring.score_fraude,
        "nivel": scoring.nivel_riesgo,
        "accion": scoring.accion_recomendada,
        "hitl": scoring.requiere_hitl,
        "latencia_s": round(latencia, 2)
    })

# ============================================================
# RESUMEN EJECUTIVO
# ============================================================

print(f"\n{'='*55}")
print("RESUMEN EJECUTIVO - Módulo Fraude Transaccional")
print(f"{'='*55}")
print(f"{'TX':<18} {'Importe':>10} {'País':>6} {'Score':>6} {'Nivel':<8} {'Acción':<20} {'ms':>6}")
print(f"{'─'*18} {'─'*10} {'─'*6} {'─'*6} {'─'*8} {'─'*20} {'─'*6}")

for r in resultados:
    print(f"{r['tx_id']:<18} {r['importe']:>10,.0f} {r['pais_destino']:>6} "
          f"{r['score']:>6.2f} {r['nivel']:<8} {r['accion']:<20} "
          f"{r['latencia_s']*1000:>6.0f}")

bloqueadas = sum(1 for r in resultados if "BLOQUEAR" in r["accion"])
revisar    = sum(1 for r in resultados if r["accion"] == "REVISAR_MANUAL")
aprobadas  = sum(1 for r in resultados if r["accion"] == "APROBAR")
latencia_media = sum(r["latencia_s"] for r in resultados) / len(resultados)

print(f"\nAprobadas: {aprobadas} · Revisar: {revisar} · Bloqueadas: {bloqueadas}")
print(f"Latencia media: {latencia_media:.2f}s")
print(f"\nMódulo Fraude Transaccional operativo.")
