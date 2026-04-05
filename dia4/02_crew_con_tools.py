from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import os
import json

load_dotenv()

# ============================================================
# CrewAI con herramientas personalizadas
# ============================================================
# En CrewAI las herramientas se definen como clases que heredan
# de BaseTool. A diferencia del @tool de LangChain, aquí usas
# Pydantic para tipar los inputs de forma explícita.
# Como DA: es exactamente definir el contrato de entrada de
# cada fuente de datos antes de que el agente la consulte.

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

# ============================================================
# HERRAMIENTAS PERSONALIZADAS
# ============================================================

class InputCatalogoTablas(BaseModel):
    dominio: str = Field(description="Dominio de negocio: ventas, clientes, productos, finanzas")
    entorno: str = Field(description="Entorno: produccion, desarrollo, staging")

class HerramientaCatalogoTablas(BaseTool):
    name: str = "consultar_catalogo_tablas"
    description: str = """Consulta el catálogo de datos corporativo y devuelve las tablas
    disponibles en un dominio y entorno específicos con sus metadatos."""
    args_schema: Type[BaseModel] = InputCatalogoTablas

    def _run(self, dominio: str, entorno: str) -> str:
        catalogo = {
            "ventas": {
                "produccion": [
                    {"tabla": "fact_ventas", "filas": 45_230_891, "actualizacion": "diaria", "owner": "equipo-ventas"},
                    {"tabla": "fact_devoluciones", "filas": 2_341_200, "actualizacion": "diaria", "owner": "equipo-ventas"},
                    {"tabla": "agg_ventas_mensual", "filas": 124_800, "actualizacion": "mensual", "owner": "equipo-bi"}
                ],
                "desarrollo": [
                    {"tabla": "fact_ventas", "filas": 100_000, "actualizacion": "manual", "owner": "equipo-datos"}
                ]
            },
            "clientes": {
                "produccion": [
                    {"tabla": "dim_cliente", "filas": 2_000_000, "actualizacion": "diaria", "owner": "equipo-crm"},
                    {"tabla": "fact_comportamiento", "filas": 89_450_000, "actualizacion": "horaria", "owner": "equipo-datos"}
                ]
            },
            "productos": {
                "produccion": [
                    {"tabla": "dim_producto", "filas": 485_000, "actualizacion": "semanal", "owner": "equipo-catalogo"},
                    {"tabla": "fact_inventario", "filas": 12_450_000, "actualizacion": "diaria", "owner": "equipo-logistica"}
                ]
            }
        }
        resultado = catalogo.get(dominio.lower(), {}).get(entorno.lower(), [])
        if not resultado:
            return f"No se encontraron tablas para dominio={dominio} entorno={entorno}"
        return json.dumps(resultado, ensure_ascii=False, indent=2)

class InputCalidadDatos(BaseModel):
    tabla: str = Field(description="Nombre exacto de la tabla a auditar")
    entorno: str = Field(description="Entorno donde está la tabla")

class HerramientaCalidadDatos(BaseTool):
    name: str = "auditar_calidad_datos"
    description: str = """Ejecuta una auditoría de calidad de datos sobre una tabla específica.
    Retorna métricas de completitud, unicidad, validez y frescura de los datos."""
    args_schema: Type[BaseModel] = InputCalidadDatos

    def _run(self, tabla: str, entorno: str) -> str:
        metricas = {
            "fact_ventas": {
                "completitud": "98.7%",
                "unicidad": "99.9%",
                "validez": "97.2%",
                "frescura": "OK - última carga hace 3 horas",
                "problemas": ["2.8% de registros con importe_neto nulo", "1.3% con codigo_postal inválido"],
                "recomendaciones": ["Añadir constraint NOT NULL en importe_neto", "Validar formato postal en ingesta"]
            },
            "dim_cliente": {
                "completitud": "94.3%",
                "unicidad": "100%",
                "validez": "96.8%",
                "frescura": "OK - última carga hace 1 hora",
                "problemas": ["5.7% de clientes sin email", "3.2% con fecha_alta futura"],
                "recomendaciones": ["Campaña de enriquecimiento de email", "Corregir validación de fecha_alta en CRM"]
            },
            "dim_producto": {
                "completitud": "99.8%",
                "unicidad": "100%",
                "validez": "99.1%",
                "frescura": "WARN - última carga hace 8 días",
                "problemas": ["0.9% de productos sin categoria asignada"],
                "recomendaciones": ["Revisar pipeline de actualización semanal", "Asignar categoría por defecto"]
            }
        }
        resultado = metricas.get(tabla.lower(), {
            "completitud": "N/A",
            "unicidad": "N/A",
            "validez": "N/A",
            "frescura": "N/A",
            "problemas": [f"Tabla {tabla} no encontrada en el catálogo de auditoría"],
            "recomendaciones": ["Registrar la tabla en el sistema de calidad de datos"]
        })
        return json.dumps({**resultado, "tabla": tabla, "entorno": entorno}, ensure_ascii=False, indent=2)

class InputGenerarInforme(BaseModel):
    titulo: str = Field(description="Título del informe")
    hallazgos: str = Field(description="Hallazgos principales del análisis")
    recomendaciones: str = Field(description="Recomendaciones priorizadas")

class HerramientaGenerarInforme(BaseTool):
    name: str = "generar_informe_calidad"
    description: str = """Genera y persiste un informe formal de calidad de datos
    en formato Markdown listo para compartir con stakeholders."""
    args_schema: Type[BaseModel] = InputGenerarInforme

    def _run(self, titulo: str, hallazgos: str, recomendaciones: str) -> str:
        contenido = f"""# {titulo}

## Hallazgos
{hallazgos}

## Recomendaciones
{recomendaciones}

---
*Informe generado automáticamente por el sistema de auditoría de datos*
"""
        ruta = f"./dia4/informe_calidad.md"
        os.makedirs("./dia4", exist_ok=True)
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(contenido)
        return f"Informe guardado en {ruta} ({len(contenido)} caracteres)"

# ============================================================
# AGENTES CON HERRAMIENTAS
# ============================================================

auditor_datos = Agent(
    role="Auditor de Calidad de Datos",
    goal="""Analizar la calidad de los datos en el catálogo corporativo,
    identificar problemas críticos y generar informes accionables.""",
    backstory="""Especialista en Data Quality y Data Governance con
    certificación DAMA-DMBOK. Has auditado más de 500 tablas en
    entornos enterprise. Conoces Great Expectations, dbt tests y
    Soda Core. Eres metódico: primero inventarías, luego auditas,
    finalmente reportas con prioridades claras.""",
    tools=[
        HerramientaCatalogoTablas(),
        HerramientaCalidadDatos(),
        HerramientaGenerarInforme()
    ],
    llm=llm,
    verbose=True,
    max_iter=8
)

# ============================================================
# TAREA CON DEPENDENCIA DE HERRAMIENTAS
# ============================================================

tarea_auditoria = Task(
    description="""Realiza una auditoría completa de calidad de datos
    del dominio de ventas en producción. Sigue este proceso:

    1. Consulta el catálogo para ver qué tablas existen en ventas/produccion
    2. Audita la calidad de cada tabla encontrada
    3. Identifica los 3 problemas más críticos por impacto de negocio
    4. Genera el informe formal con hallazgos y recomendaciones priorizadas

    Prioriza los problemas según: impacto en ingresos > compliance > eficiencia.""",
    expected_output="""Informe de auditoría con:
    - Lista de tablas auditadas con sus métricas clave
    - Top 3 problemas críticos con justificación de prioridad
    - Plan de acción con responsables y plazos sugeridos
    - Confirmación de que el informe ha sido guardado""",
    agent=auditor_datos
)

crew = Crew(
    agents=[auditor_datos],
    tasks=[tarea_auditoria],
    process=Process.sequential,
    verbose=True
)

print("=" * 60)
print("AUDITORÍA DE DATOS CON AGENTE Y HERRAMIENTAS")
print("=" * 60)

resultado = crew.kickoff()

print("\n" + "=" * 60)
print("RESULTADO FINAL")
print("=" * 60)
print(resultado.raw)
