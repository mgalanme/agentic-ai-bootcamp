from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ============================================================
# PATRÓN 1: Conversación con historial
# ============================================================
# MessagesPlaceholder es un hueco en el prompt donde inyectamos
# la lista de mensajes anteriores. El LLM los lee antes de
# responder, por eso "recuerda" lo que se dijo antes.
# Como DA: es exactamente una tabla de auditoría de conversación
# que el LLM consulta en cada turno.

prompt_con_historia = ChatPromptTemplate.from_messages([
    ("system", "Eres un arquitecto de soluciones senior. Sé conciso."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain_con_historia = prompt_con_historia | llm | StrOutputParser()

history = []

def chat(mensaje: str) -> str:
    respuesta = chain_con_historia.invoke({
        "input": mensaje,
        "history": history
    })
    # Cada turno se acumula en la lista
    # HumanMessage = lo que dijo el usuario
    # AIMessage = lo que respondió el modelo
    history.append(HumanMessage(content=mensaje))
    history.append(AIMessage(content=respuesta))
    return respuesta

print("=== CONVERSACIÓN CON MEMORIA ===")
r1 = chat("¿Cuál es la diferencia entre arquitectura de datos y arquitectura empresarial?")
print(f"Turno 1:\n{r1}\n")

r2 = chat("¿Y cómo encaja la IA agéntica en ambas?")
print(f"Turno 2 (usa contexto del turno 1):\n{r2}\n")

# ============================================================
# PATRÓN 2: Structured Output
# ============================================================
# with_structured_output obliga al LLM a devolver un objeto
# Python con el esquema exacto que defines en la clase Pydantic.
# Como DA: defines el esquema de los datos que quieres recibir,
# igual que defines el esquema de una tabla. El LLM lo rellena.
# Esto elimina el parsing manual de texto y los errores de formato.

class AnalisisArquitectura(BaseModel):
    nombre_patron: str = Field(description="Nombre del patrón arquitectónico")
    casos_de_uso: List[str] = Field(description="Lista de casos de uso principales")
    ventajas: List[str] = Field(description="Ventajas principales")
    desventajas: List[str] = Field(description="Desventajas o limitaciones")
    complejidad: str = Field(description="Exactamente una de estas palabras: Baja, Media o Alta")

# Aquí el parser NO es StrOutputParser sino el propio esquema Pydantic
# El LLM recibe instrucciones internas para formatear su respuesta
structured_llm = llm.with_structured_output(AnalisisArquitectura)

prompt_analisis = ChatPromptTemplate.from_messages([
    ("system", "Eres un arquitecto experto. Analiza patrones arquitectónicos con rigor técnico."),
    ("human", "Analiza el patrón arquitectónico: {patron}")
])

chain_estructurada = prompt_analisis | structured_llm

print("=== STRUCTURED OUTPUT ===")
resultado = chain_estructurada.invoke({
    "patron": "RAG (Retrieval Augmented Generation)"
})

# resultado es un objeto Python tipado, no texto libre
# Puedes acceder a sus campos directamente, pasarlo a una BD,
# serializarlo a JSON, etc. Sin parsear nada manualmente.
print(f"Patrón: {resultado.nombre_patron}")
print(f"Complejidad: {resultado.complejidad}")
print(f"\nCasos de uso:")
for caso in resultado.casos_de_uso:
    print(f"  - {caso}")
print(f"\nVentajas:")
for v in resultado.ventajas:
    print(f"  - {v}")
print(f"\nDesventajas:")
for d in resultado.desventajas:
    print(f"  - {d}")
