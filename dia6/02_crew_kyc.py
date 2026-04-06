"""
ARGOS-FCC · Script 06-02: Módulo KYC/KYB con CrewAI
=====================================================
Due diligence continuo sobre clientes usando una crew
de dos agentes especializados.

Patrón arquitectónico: multi-agente con roles de negocio.
CrewAI modela el proceso de KYC exactamente como ocurre
en la realidad del banco: un equipo de especialistas
trabajando en secuencia con contexto acumulado.

Como EA: KYC es una obligación legal (Art.14 Ley 10/2010).
Como DA: el output actualiza el perfil de riesgo del cliente.
Como AI Architect: CrewAI es el patrón correcto aquí porque
el proceso tiene roles claros y entregables definidos.
"""
import os
import json
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List
from neo4j import GraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

print("ARGOS-FCC · Módulo KYC/KYB con CrewAI")
print("=" * 55)

# ============================================================
# LLM
# ============================================================

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

# ============================================================
# CONEXIONES
# ============================================================

NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "bootcamp1234")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

print("Cargando embeddings y RAG regulatorio...")
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
print("Componentes listos\n")

# ============================================================
# LISTAS DE SANCIONES (simuladas)
# ============================================================
# En producción conectarías con OFAC SDN List API,
# UN Consolidated List o World-Check.

LISTAS_SANCIONES = {
    "OFAC_SDN": ["IR", "KP", "SY", "CU"],      # países sancionados OFAC
    "ONU":      ["IR", "KP", "LY", "SS"],       # países sancionados ONU
    "UE":       ["IR", "KP", "SY", "RU", "BY"], # países sancionados UE
    "FATF_BLACK": ["IR", "KP", "MM"],            # lista negra FATF
    "FATF_GREY":  ["PA", "VG", "KY", "LR", "SY"], # lista gris FATF
}

# ============================================================
# HERRAMIENTAS KYC
# ============================================================

class InputPerfil(BaseModel):
    id_cliente: str = Field(description="ID del cliente a consultar (ej: CLI-0001)")

class HerramientaPerfilCliente(BaseTool):
    name: str = "consultar_perfil_cliente"
    description: str = """Consulta el perfil completo de un cliente en Neo4j:
    datos personales, cuentas asociadas, historial transaccional,
    segmento de riesgo actual y si es PEP."""
    args_schema: Type[BaseModel] = InputPerfil

    def _run(self, id_cliente: str) -> str:
        with driver.session() as s:
            r = s.run("""
                MATCH (cli:Cliente {id: $id})
                OPTIONAL MATCH (cli)-[:TITULAR_DE]->(cta:Cuenta)
                OPTIONAL MATCH (cta)-[:ORIGEN_DE]->(t:Transaccion)
                WITH cli,
                     collect(DISTINCT cta.id)[0..5]   AS cuentas,
                     collect(DISTINCT cta.tipo)[0..4]  AS tipos_cuenta,
                     count(t)                          AS n_txs,
                     round(avg(t.importe))             AS importe_medio,
                     round(sum(t.importe))             AS volumen_total,
                     collect(DISTINCT t.pais_destino)[0..6] AS paises
                RETURN cli.id       AS id,
                       cli.nombre   AS nombre,
                       cli.tipo     AS tipo,
                       cli.pais     AS pais,
                       cli.riesgo   AS riesgo_actual,
                       cli.pep      AS es_pep,
                       cli.fecha_alta AS fecha_alta,
                       cli.nif      AS nif,
                       cuentas, tipos_cuenta,
                       n_txs, importe_medio,
                       volumen_total, paises
            """, id=id_cliente)
            datos = r.data()
            if not datos:
                return f"Cliente {id_cliente} no encontrado"
            return json.dumps(datos[0], ensure_ascii=False, indent=2)


class InputSanciones(BaseModel):
    pais: str = Field(description="Código ISO del país a verificar (ej: IR, KP, ES)")
    nombre: str = Field(description="Nombre del cliente o empresa a verificar")

class HerramientaVerificadorSanciones(BaseTool):
    name: str = "verificar_sanciones"
    description: str = """Verifica si un cliente o país está en listas de sanciones
    internacionales: OFAC SDN, ONU, UE y listas FATF (negra y gris).
    Devuelve las listas en las que aparece y el nivel de riesgo."""
    args_schema: Type[BaseModel] = InputSanciones

    def _run(self, pais: str, nombre: str) -> str:
        listas_encontradas = []
        for lista, paises in LISTAS_SANCIONES.items():
            if pais.upper() in paises:
                listas_encontradas.append(lista)

        if listas_encontradas:
            return (
                f"ALERTA SANCIONES: El país {pais} del cliente '{nombre}' "
                f"aparece en las siguientes listas: {', '.join(listas_encontradas)}. "
                f"Se requiere diligencia debida reforzada y posible bloqueo "
                f"según Art.26 Ley 10/2010 y normativa OFAC."
            )
        return (
            f"País {pais} del cliente '{nombre}' no aparece en listas "
            f"de sanciones activas. Sin restricciones por este criterio."
        )


class InputRAG(BaseModel):
    consulta: str = Field(description="Consulta sobre regulación KYC/AML a buscar")

class HerramientaRAGRegulatorio(BaseTool):
    name: str = "consultar_regulacion_kyc"
    description: str = """Consulta la base de conocimiento regulatoria para
    obtener los requisitos aplicables a un caso KYC concreto.
    Indexa FATF, AMLD6, Ley 10/2010 y guías SEPBLAC."""
    args_schema: Type[BaseModel] = InputRAG

    def _run(self, consulta: str) -> str:
        docs = vectorstore.similarity_search(consulta, k=3)
        if not docs:
            return "No se encontraron documentos regulatorios relevantes."
        resultado = f"Regulación aplicable para '{consulta}':\n\n"
        for doc in docs:
            resultado += (
                f"[{doc.metadata['fuente']} · {doc.metadata['articulo']}]\n"
                f"{doc.page_content[:250]}\n\n"
            )
        return resultado


class InputActualizarRiesgo(BaseModel):
    id_cliente: str = Field(description="ID del cliente a actualizar")
    nuevo_riesgo: str = Field(description="Nuevo nivel de riesgo: bajo, medio o alto")
    justificacion: str = Field(description="Justificación regulatoria del cambio de riesgo")

class HerramientaActualizarRiesgo(BaseTool):
    name: str = "actualizar_perfil_riesgo"
    description: str = """Actualiza el perfil de riesgo del cliente en Neo4j
    con la nueva clasificación y su justificación regulatoria.
    Esta actualización queda registrada como parte del expediente KYC."""
    args_schema: Type[BaseModel] = InputActualizarRiesgo

    def _run(self, id_cliente: str, nuevo_riesgo: str, justificacion: str) -> str:
        with driver.session() as s:
            s.run("""
                MATCH (cli:Cliente {id: $id})
                SET cli.riesgo             = $riesgo,
                    cli.kyc_justificacion  = $justificacion,
                    cli.kyc_fecha          = $fecha
            """,
            id=id_cliente,
            riesgo=nuevo_riesgo,
            justificacion=justificacion,
            fecha=str(__import__('datetime').datetime.now().isoformat()))
        return (
            f"Perfil de riesgo del cliente {id_cliente} actualizado a '{nuevo_riesgo}'. "
            f"Justificación registrada en Neo4j. "
            f"Expediente KYC actualizado con fecha y motivo regulatorio."
        )

# ============================================================
# AGENTES KYC
# ============================================================

agente_verificador = Agent(
    role="Especialista en Verificación KYC",
    goal="""Verificar la identidad y el perfil de riesgo del cliente,
    comprobar su presencia en listas de sanciones internacionales,
    e identificar si es una Persona Políticamente Expuesta (PEP).""",
    backstory="""Llevas 10 años en el departamento de compliance de un banco retail
    español. Conoces a fondo las listas OFAC, ONU y UE. Has tramitado miles de
    procesos KYC y sabes exactamente qué señales de alerta buscar.
    Tu lema: ningún cliente de alto riesgo pasa desapercibido.""",
    tools=[
        HerramientaPerfilCliente(),
        HerramientaVerificadorSanciones(),
        HerramientaRAGRegulatorio()
    ],
    llm=llm,
    verbose=True,
    max_iter=6
)

agente_riesgo = Agent(
    role="Analista de Riesgo KYC/KYB",
    goal="""Evaluar el riesgo regulatorio del cliente basándose en el informe
    del verificador, aplicar la normativa correspondiente y actualizar
    el perfil de riesgo con justificación regulatoria explícita.""",
    backstory="""Eres el analista senior de riesgo del banco. Recibes los informes
    del verificador y tomas la decisión final sobre la clasificación de riesgo.
    Cada decisión que tomas tiene respaldo regulatorio explícito: citas el artículo,
    la fuente y el criterio aplicado. Tu trabajo es la última línea de defensa
    antes de que el director de compliance firme.""",
    tools=[
        HerramientaRAGRegulatorio(),
        HerramientaActualizarRiesgo()
    ],
    llm=llm,
    verbose=True,
    max_iter=6
)

# ============================================================
# PROCESO KYC: FUNCIÓN PRINCIPAL
# ============================================================

def ejecutar_kyc(id_cliente: str, motivo: str = "revision_periodica"):
    """
    Ejecuta el proceso de due diligence KYC completo para un cliente.
    """
    print(f"\n{'='*55}")
    print(f"KYC DUE DILIGENCE: {id_cliente}")
    print(f"Motivo: {motivo}")
    print(f"{'='*55}")

    tarea_verificacion = Task(
        description=f"""Realiza la verificación KYC completa del cliente {id_cliente}.

Sigue este proceso:
1. Consulta el perfil completo del cliente en Neo4j
2. Verifica el país de residencia contra listas de sanciones
3. Consulta la regulación aplicable según el perfil encontrado
4. Identifica todas las señales de alerta (PEP, país de riesgo,
   volumen transaccional, patrones inusuales)
5. Produce un informe de verificación detallado

Motivo de la revisión: {motivo}""",
        expected_output="""Informe de verificación KYC con:
- Datos del cliente verificados
- Resultado de verificación en listas de sanciones
- Identificación de PEP (sí/no con justificación)
- Señales de alerta detectadas
- Clasificación de riesgo preliminar con base regulatoria
Máximo 300 palabras.""",
        agent=agente_verificador
    )

    tarea_riesgo = Task(
        description=f"""Basándote en el informe de verificación del cliente {id_cliente},
evalúa el riesgo regulatorio y actualiza el perfil.

Sigue este proceso:
1. Revisa las señales de alerta identificadas por el verificador
2. Consulta la regulación aplicable para determinar el nivel de riesgo correcto
3. Decide el nivel de riesgo final: bajo, medio o alto
4. Actualiza el perfil del cliente en el sistema con justificación regulatoria
5. Produce el dictamen final de riesgo KYC

El nivel de riesgo debe basarse explícitamente en artículos regulatorios concretos.""",
        expected_output="""Dictamen KYC con:
- Nivel de riesgo final: BAJO / MEDIO / ALTO
- Artículos regulatorios que justifican la clasificación
- Medidas de diligencia debida requeridas
- Confirmación de actualización del perfil en el sistema
- Próxima fecha de revisión recomendada
Máximo 250 palabras.""",
        agent=agente_riesgo,
        context=[tarea_verificacion]
    )

    crew = Crew(
        agents=[agente_verificador, agente_riesgo],
        tasks=[tarea_verificacion, tarea_riesgo],
        process=Process.sequential,
        verbose=True
    )

    resultado = crew.kickoff()
    return resultado

# ============================================================
# CASOS DE PRUEBA KYC
# ============================================================

# Caso 1: Cliente con país de riesgo (Irán)
resultado1 = ejecutar_kyc(
    id_cliente="CLI-0073",   # cliente con pais=IR detectado en Día 5
    motivo="alerta_transaccional_AML"
)

print(f"\n{'='*55}")
print("DICTAMEN FINAL KYC - CLI-0073")
print(f"{'='*55}")
print(resultado1.raw)

# Caso 2: Cliente PEP
print(f"\n{'─'*55}")
respuesta = input("\n¿Ejecutar KYC para cliente PEP? (s/n): ").strip().lower()
if respuesta == "s":
    resultado2 = ejecutar_kyc(
        id_cliente="CLI-0030",   # cliente con pais=VG (Islas Vírgenes)
        motivo="revision_periodica_pep"
    )
    print(f"\n{'='*55}")
    print("DICTAMEN FINAL KYC - CLI-0030")
    print(f"{'='*55}")
    print(resultado2.raw)

driver.close()
print("\nMódulo KYC/KYB operativo.")
