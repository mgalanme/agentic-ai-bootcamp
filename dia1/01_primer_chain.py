# ~/agentic-ai-bootcamp/dia1/01_primer_chain.py
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# Inicializar el LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Modelo gratuito y muy potente
    temperature=0.1
)

# LCEL: la forma moderna de construir chains
# Nota para el Arquitecto: esto es un pipeline declarativo,
# igual que un DAG de datos pero para prompts

prompt = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente experto en arquitectura de datos.
    Responde de forma concisa y técnica."""),
    ("human", "{pregunta}")
])

# El operador | es el pipe de LCEL (igual que Unix pipes)
chain = prompt | llm | StrOutputParser()

# Invocación síncrona
respuesta = chain.invoke({"pregunta": "¿Cuándo usar un índice vectorial vs un índice B-tree?"})
print(respuesta)

# Streaming (importante para UX en producción)
print("\n--- STREAMING ---")
for chunk in chain.stream({"pregunta": "Explica brevemente qué es embeddings en 3 líneas"}):
    print(chunk, end="", flush=True)
print()
