from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
import os

load_dotenv()

# ============================================================
# CrewAI: equipos de agentes especializados
# ============================================================
# Como EA: piensa en esto como modelar una organización virtual.
# Cada Agent = un rol de negocio con su expertise específico.
# Cada Task = un entregable concreto con criterios de aceptación.
# La Crew = el proceso de negocio que coordina roles y entregables.
#
# A diferencia de LangGraph donde cablearas el flujo nodo a nodo,
# aquí defines QUÉ se necesita (roles + entregables) y CrewAI
# gestiona el CÓMO (coordinación, delegación, contexto entre tareas).

# ============================================================
# CONFIGURACIÓN DEL LLM
# ============================================================
# CrewAI tiene su propio wrapper de LLM independiente de LangChain.
# El formato del modelo es "proveedor/nombre-modelo".

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

# ============================================================
# AGENTES: los roles del equipo
# ============================================================
# El backstory es fundamental. El LLM lo usa para calibrar:
# - profundidad técnica de las respuestas
# - perspectiva desde la que analiza el problema
# - formato y tono de los entregables
# No es texto decorativo: es el prompt de sistema del agente.

arquitecto_datos = Agent(
    role="Arquitecto de Datos Senior",
    goal="""Analizar requisitos de datos y diseñar la arquitectura
    de datos óptima: fuentes, almacenamiento, pipelines y governance.""",
    backstory="""Arquitecto de datos con 15 años de experiencia en
    Data Warehouses, Data Lakes y pipelines modernos. Has liderado
    migraciones a Snowflake, implementado arquitecturas Medallion
    y diseñado estrategias de governance para empresas Fortune 500.
    Eres meticuloso con la calidad del dato, el linaje y la trazabilidad.""",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

arquitecto_ia = Agent(
    role="Arquitecto de Soluciones IA",
    goal="""Diseñar la capa de inteligencia artificial sobre la
    arquitectura de datos: RAG, agentes, vector DBs y patrones
    de integración con los sistemas existentes.""",
    backstory="""Especialista en sistemas de IA en producción con
    experiencia desplegando RAG, agentes LangGraph y sistemas
    multi-agente en entornos enterprise. Conoces los tradeoffs
    de latencia, coste y fiabilidad de los LLMs. Eres pragmático:
    la solución más simple que funcione siempre gana.""",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

arquitecto_empresarial = Agent(
    role="Arquitecto Empresarial",
    goal="""Evaluar el encaje estratégico de la solución técnica,
    identificar riesgos de governance y compliance, y elaborar
    el roadmap de implementación orientado a decisión ejecutiva.""",
    backstory="""Arquitecto empresarial certificado TOGAF con visión
    holística de los sistemas de información. Eres el puente entre
    el negocio y la tecnología. Has dirigido transformaciones digitales
    en banca, retail y manufactura. Tu prioridad es que las soluciones
    escalen, se mantengan y aporten valor de negocio medible.""",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# ============================================================
# TAREAS: los entregables del equipo
# ============================================================
# description: qué debe hacer el agente y qué información tiene.
# expected_output: el contrato de entrega. Cuanto más específico,
# mejor el resultado. Es el equivalente a los criterios de aceptación
# de una historia de usuario.
# context: lista de tareas cuyo output se inyecta como contexto.
# Así el arquitecto IA ve el trabajo del arquitecto de datos antes
# de diseñar la capa de IA.

tarea_datos = Task(
    description="""Analiza este requisito de negocio y diseña la arquitectura de datos:

    REQUISITO: Una empresa de retail online quiere construir un sistema que:
    1. Analice el comportamiento de compra de sus 2 millones de clientes
    2. Genere recomendaciones personalizadas en tiempo cuasi-real (< 2 segundos)
    3. Permita a los equipos de negocio hacer preguntas en lenguaje natural
       sobre ventas históricas sin necesidad de conocer SQL
    4. Detecte patrones de fraude en transacciones en tiempo real

    Volumen: 500GB de datos históricos, 50.000 transacciones/día.

    Tu entregable debe cubrir:
    1. Fuentes de datos identificadas y estrategia de ingesta
    2. Arquitectura de almacenamiento recomendada con justificación
    3. Pipeline de transformación (capas y tecnologías)
    4. Estrategia de vectorización para las recomendaciones y el Q&A
    5. Governance: linaje, calidad y control de acceso""",
    expected_output="""Documento técnico estructurado con las 5 secciones.
    Máximo 500 palabras. Incluye decisiones técnicas con justificación explícita.
    Formato: secciones numeradas con bullets para decisiones clave.""",
    agent=arquitecto_datos
)

tarea_ia = Task(
    description="""Basándote en la arquitectura de datos diseñada,
    diseña la capa de inteligencia artificial del sistema retail.

    Tu entregable debe cubrir:
    1. Diseño del sistema RAG para consultas en lenguaje natural sobre ventas
    2. Arquitectura del motor de recomendaciones (agente o pipeline batch)
    3. Sistema de detección de fraude en tiempo real
    4. Stack tecnológico completo (LangChain/LangGraph/CrewAI + vector DB)
    5. Estimación de latencia por caso de uso y coste mensual aproximado""",
    expected_output="""Especificación técnica de la capa IA con las 5 secciones.
    Incluye justificación de la elección del vector DB con criterios de DA.
    Máximo 500 palabras. Secciones numeradas.""",
    agent=arquitecto_ia,
    context=[tarea_datos]
)

tarea_enterprise = Task(
    description="""Revisa las propuestas de arquitectura de datos e IA
    y elabora el informe ejecutivo de arquitectura empresarial.

    Tu entregable debe cubrir:
    1. Resumen ejecutivo (para CTO, sin jerga técnica excesiva, máximo 100 palabras)
    2. Encaje con arquitectura empresarial típica de retail
    3. Tabla de riesgos: técnicos, de negocio y de compliance/GDPR
    4. Roadmap en tres fases (0-3 meses, 3-6 meses, 6-12 meses)
    5. KPIs para medir el éxito del sistema""",
    expected_output="""Informe ejecutivo estructurado orientado a decisión.
    Lenguaje claro y directo. Tabla de riesgos con columnas: riesgo, probabilidad,
    impacto, mitigación. Roadmap con hitos concretos. Máximo 500 palabras.""",
    agent=arquitecto_empresarial,
    context=[tarea_datos, tarea_ia]
)

# ============================================================
# CREW: el equipo y su proceso de trabajo
# ============================================================
# Process.sequential: las tareas se ejecutan en orden.
# El output de cada tarea se pasa como contexto a la siguiente
# gracias al parámetro context=[...] en cada Task.
#
# Process.hierarchical añadiría un manager agent que delega
# y supervisa. Lo veremos en la práctica 4.2.

crew = Crew(
    agents=[arquitecto_datos, arquitecto_ia, arquitecto_empresarial],
    tasks=[tarea_datos, tarea_ia, tarea_enterprise],
    process=Process.sequential,
    verbose=True
)

# ============================================================
# EJECUCIÓN
# ============================================================
print("=" * 60)
print("ANÁLISIS MULTI-AGENTE: ARQUITECTURA RETAIL IA")
print("=" * 60)
print("3 agentes · 3 tareas secuenciales · contexto acumulado\n")

resultado = crew.kickoff()

print("\n" + "=" * 60)
print("RESULTADO FINAL")
print("=" * 60)
print(resultado.raw)

# Guardar el resultado
os.makedirs("./dia4", exist_ok=True)
with open("./dia4/analisis_retail.md", "w", encoding="utf-8") as f:
    f.write(resultado.raw)
print("\nResultado guardado en dia4/analisis_retail.md")
