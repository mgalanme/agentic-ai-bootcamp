from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
import os
import json

load_dotenv()

# ============================================================
# SISTEMA DE INTEGRACIÓN FINAL: CrewAI + RAG + Vector DB
# ============================================================
# Este ejercicio integra todo lo aprendido en los 4 días:
# - Día 1: LangChain + LCEL + Embeddings
# - Día 2: Vector DB (Chroma) para recuperación semántica
# - Día 3: Patrón agéntico con herramientas
# - Día 4: CrewAI con agentes especializados y herramientas
#
# Caso de uso: sistema de consultoría de arquitectura que
# recupera ADRs relevantes semánticamente y los usa como
# contexto para que los agentes tomen decisiones informadas.

# ============================================================
# PASO 1: Preparar el vector store con ADRs
# ============================================================

print("Inicializando vector store con ADRs...")

embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True}
)

adrs = [
    Document(
        page_content="ADR-001: Para producción con más de 1M vectores usar Qdrant. Soporta filtros de payload compuestos, cuantización escalar y sharding horizontal. Chroma solo para desarrollo.",
        metadata={"id": "ADR-001", "categoria": "infraestructura", "estado": "aprobado"}
    ),
    Document(
        page_content="ADR-002: Modelo de embeddings estándar: nomic-embed-text-v1, dimensión 768. Evaluado frente a OpenAI ada-002. Mejor rendimiento en castellano e inglés técnico.",
        metadata={"id": "ADR-002", "categoria": "ml", "estado": "aprobado"}
    ),
    Document(
        page_content="ADR-003: Chunking strategy: 400 caracteres, 40 de overlap, separadores jerárquicos. Evaluado en corpus de documentación técnica de arquitectura.",
        metadata={"id": "ADR-003", "categoria": "rag", "estado": "aprobado"}
    ),
    Document(
        page_content="ADR-004: LangGraph para flujos agénticos complejos con estado persistente y ciclos. CrewAI para pipelines multi-agente orientados a roles de negocio.",
        metadata={"id": "ADR-004", "categoria": "agentes", "estado": "aprobado"}
    ),
    Document(
        page_content="ADR-005: GraphRAG con Neo4j para dominios con relaciones complejas: catálogos de datos, linaje, mapas de sistemas. Combina similitud vectorial con traversal de grafo.",
        metadata={"id": "ADR-005", "categoria": "grafo", "estado": "aprobado"}
    ),
    Document(
        page_content="ADR-006: Human-in-the-loop obligatorio para acciones destructivas. Usar LangGraph interrupt_before con MemorySaver. En producción: SqliteSaver o PostgresSaver.",
        metadata={"id": "ADR-006", "categoria": "governance", "estado": "aprobado"}
    ),
    Document(
        page_content="ADR-007: LLM para producción: Claude Sonnet para razonamiento complejo. Groq llama-3.3-70b para alta frecuencia. Siempre temperature=0 para tareas técnicas.",
        metadata={"id": "ADR-007", "categoria": "infraestructura", "estado": "aprobado"}
    ),
]

os.makedirs("./data/chroma_integracion", exist_ok=True)
chroma_client = chromadb.PersistentClient(path="./data/chroma_integracion")
if "adrs_integracion" in [c.name for c in chroma_client.list_collections()]:
    chroma_client.delete_collection("adrs_integracion")

vectorstore = Chroma.from_documents(
    documents=adrs,
    embedding=embeddings,
    persist_directory="./data/chroma_integracion",
    collection_name="adrs_integracion",
    client=chroma_client
)
print(f"Vector store listo: {vectorstore._collection.count()} ADRs indexados\n")

# ============================================================
# PASO 2: Herramienta RAG sobre el vector store
# ============================================================

class InputConsultarADRs(BaseModel):
    consulta: str = Field(description="Descripción del problema o decisión técnica a consultar")
    k: int = Field(default=3, description="Número de ADRs a recuperar (1-5)")

class HerramientaConsultarADRs(BaseTool):
    name: str = "consultar_adrs"
    description: str = """Consulta la base de conocimiento de Architecture Decision Records (ADRs)
    usando búsqueda semántica. Devuelve las decisiones arquitectónicas más relevantes
    para el problema o tecnología consultada."""
    args_schema: Type[BaseModel] = InputConsultarADRs

    def _run(self, consulta: str, k: int = 3) -> str:
        resultados = vectorstore.similarity_search(consulta, k=k)
        if not resultados:
            return "No se encontraron ADRs relevantes para esta consulta."
        respuesta = f"ADRs relevantes para '{consulta}':\n\n"
        for i, doc in enumerate(resultados, 1):
            respuesta += f"{i}. [{doc.metadata['id']}] {doc.page_content}\n"
            respuesta += f"   Categoría: {doc.metadata['categoria']} | Estado: {doc.metadata['estado']}\n\n"
        return respuesta

class InputEvaluarDecision(BaseModel):
    decision: str = Field(description="Decisión técnica propuesta a evaluar")
    contexto: str = Field(description="Contexto del proyecto donde se aplicará")

class HerramientaEvaluarDecision(BaseTool):
    name: str = "evaluar_decision_arquitectonica"
    description: str = """Evalúa una decisión técnica propuesta contra los ADRs existentes.
    Identifica conflictos, alineaciones y riesgos respecto a las decisiones previas."""
    args_schema: Type[BaseModel] = InputEvaluarDecision

    def _run(self, decision: str, contexto: str) -> str:
        adrs_relacionados = vectorstore.similarity_search(decision, k=4)
        evaluacion = f"Evaluación de decisión: '{decision}'\n"
        evaluacion += f"Contexto: {contexto}\n\n"
        evaluacion += "ADRs relacionados encontrados:\n"
        for doc in adrs_relacionados:
            evaluacion += f"- {doc.metadata['id']}: {doc.page_content[:100]}...\n"
        evaluacion += "\nVeredicto preliminar: decisión evaluada contra base de conocimiento."
        return evaluacion

class InputRegistrarDecision(BaseModel):
    id_adr: str = Field(description="ID del nuevo ADR (ej: ADR-008)")
    titulo: str = Field(description="Título corto de la decisión")
    decision: str = Field(description="Descripción completa de la decisión tomada")
    justificacion: str = Field(description="Justificación técnica de la decisión")
    categoria: str = Field(description="Categoría: infraestructura, ml, rag, agentes, grafo, governance")

class HerramientaRegistrarDecision(BaseTool):
    name: str = "registrar_nueva_decision"
    description: str = """Registra una nueva decisión arquitectónica (ADR) en la base de conocimiento.
    El ADR queda disponible inmediatamente para futuras consultas semánticas."""
    args_schema: Type[BaseModel] = InputRegistrarDecision

    def _run(self, id_adr: str, titulo: str, decision: str,
             justificacion: str, categoria: str) -> str:
        nuevo_adr = Document(
            page_content=f"{id_adr}: {titulo}. Decisión: {decision}. Justificación: {justificacion}",
            metadata={"id": id_adr, "categoria": categoria, "estado": "nuevo"}
        )
        vectorstore.add_documents([nuevo_adr])
        return f"ADR {id_adr} registrado correctamente. Total ADRs: {vectorstore._collection.count()}"

# ============================================================
# PASO 3: LLM y agentes
# ============================================================

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

consultor_arquitectura = Agent(
    role="Consultor de Arquitectura de IA",
    goal="""Analizar requisitos técnicos, consultar la base de conocimiento
    de ADRs y proponer la arquitectura óptima para sistemas de IA agéntica.""",
    backstory="""Arquitecto senior especializado en sistemas de IA agéntica.
    Conoces LangChain, LangGraph, CrewAI y todas las bases de datos vectoriales.
    Tu método es siempre consultar primero las decisiones previas antes de
    proponer algo nuevo, para garantizar consistencia arquitectónica.""",
    tools=[HerramientaConsultarADRs()],
    llm=llm,
    verbose=True,
    max_iter=6
)

revisor_governance = Agent(
    role="Revisor de Governance Arquitectónico",
    goal="""Revisar las propuestas arquitectónicas contra los ADRs existentes,
    identificar conflictos o alineaciones y registrar las nuevas decisiones.""",
    backstory="""Arquitecto empresarial responsable del governance técnico.
    Eres el guardián de la coherencia arquitectónica. Antes de aprobar cualquier
    decisión nueva, verificas que no contradiga las decisiones previas y
    registras las decisiones aprobadas en el catálogo.""",
    tools=[HerramientaEvaluarDecision(), HerramientaRegistrarDecision()],
    llm=llm,
    verbose=True,
    max_iter=6
)

# ============================================================
# PASO 4: Tareas
# ============================================================

tarea_consulta = Task(
    description="""Un cliente quiere construir un sistema de análisis de contratos legales
    que permita a los abogados hacer preguntas en lenguaje natural sobre los contratos,
    detectar cláusulas de riesgo automáticamente y mantener un grafo de relaciones
    entre contratos, partes y obligaciones.

    Usando la base de conocimiento de ADRs, propón la arquitectura técnica óptima:
    1. Consulta los ADRs relevantes para este caso de uso
    2. Recomienda el stack tecnológico basándote en las decisiones previas
    3. Identifica qué componentes ya están decididos y cuáles necesitan nueva decisión
    4. Propón la arquitectura con justificación basada en los ADRs""",
    expected_output="""Propuesta arquitectónica que incluya:
    - Stack tecnológico recomendado con referencia a ADRs que lo justifican
    - Componentes que requieren nueva decisión arquitectónica
    - Diagrama conceptual del sistema en texto
    Máximo 400 palabras.""",
    agent=consultor_arquitectura
)

tarea_governance = Task(
    description="""Revisa la propuesta arquitectónica del consultor y:
    1. Evalúa si la propuesta es consistente con los ADRs existentes
    2. Identifica cualquier decisión nueva que no esté cubierta por ADRs existentes
    3. Registra al menos una nueva decisión arquitectónica que emerja de este proyecto
    4. Produce el veredicto final de governance""",
    expected_output="""Informe de governance con:
    - Veredicto: APROBADO, APROBADO CON CONDICIONES o RECHAZADO
    - Justificación referenciando ADRs específicos
    - Nuevas decisiones registradas en el catálogo
    - Próximos pasos recomendados""",
    agent=revisor_governance,
    context=[tarea_consulta]
)

# ============================================================
# PASO 5: Ejecutar
# ============================================================

crew = Crew(
    agents=[consultor_arquitectura, revisor_governance],
    tasks=[tarea_consulta, tarea_governance],
    process=Process.sequential,
    verbose=True
)

print("=" * 60)
print("SISTEMA DE CONSULTORÍA ARQUITECTÓNICA CON RAG + CrewAI")
print("=" * 60)
print("Agentes: consultor + revisor de governance")
print("Herramientas: búsqueda semántica en ADRs + registro\n")

resultado = crew.kickoff()

print("\n" + "=" * 60)
print("RESULTADO FINAL")
print("=" * 60)
print(resultado.raw)

print(f"\nVector store final: {vectorstore._collection.count()} ADRs")
