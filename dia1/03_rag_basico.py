from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import chromadb
import os

load_dotenv()

contenido = """
# Arquitectura Medallion en Data Lakes

La arquitectura Medallion organiza los datos en tres capas:

## Capa Bronze
Almacena los datos raw tal como llegan de las fuentes.
No hay transformaciones. Es el registro histórico inmutable.
Se usa para auditoría y reprocesamiento completo.
Los datos pueden tener errores, duplicados y formatos inconsistentes.

## Capa Silver
Datos limpiados, validados y conformados.
Se aplican reglas de negocio básicas y estandarización de formatos.
Formato columnar (Parquet o Delta Lake).
Optimizado para consultas analíticas con esquema definido.

## Capa Gold
Datos agregados y modelados para consumo directo.
Modelos dimensionales en estrella o copo de nieve.
Optimizado para BI, reportes y consumo por aplicaciones.
Actualizaciones incrementales para minimizar coste de cómputo.

# Data Mesh vs Data Lakehouse

Data Mesh es un paradigma organizacional centrado en cuatro principios:
datos como producto, propiedad descentralizada por dominio de negocio,
infraestructura de datos como plataforma self-service,
y gobernanza federada con estándares globales.

Data Lakehouse combina lo mejor de Data Warehouse y Data Lake:
transacciones ACID sobre almacenamiento en objeto,
esquema en escritura con soporte para esquema en lectura,
soporte simultáneo para cargas OLAP y entrenamiento de modelos ML.
Tecnologías principales: Delta Lake, Apache Iceberg, Apache Hudi.

# Indices Vectoriales y Busqueda Semantica

Los embeddings son representaciones vectoriales densas de texto.
Cada fragmento de texto se convierte en un array de numeros flotantes
que captura su significado semantico en un espacio de alta dimension.
Textos con significado similar tienen vectores cercanos en ese espacio.

Los indices HNSW permiten busqueda aproximada de vecinos mas cercanos
con latencia de milisegundos incluso en colecciones de millones de vectores.
La similitud se mide con cosine similarity o distancia euclidiana.

La diferencia clave con un indice B-tree es que B-tree busca
coincidencias exactas o rangos, mientras que un indice vectorial
busca similitud semantica aunque las palabras sean completamente distintas.
"""

os.makedirs("./data/docs", exist_ok=True)
with open("./data/docs/arquitectura_datos.txt", "w", encoding="utf-8") as f:
    f.write(contenido)

print("FASE 1: INDEXACION")
print("=" * 50)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=40,
    separators=["\n## ", "\n# ", "\n\n", "\n", " "]
)

loader = TextLoader("./data/docs/arquitectura_datos.txt", encoding="utf-8")
docs = loader.load()
chunks = splitter.split_documents(docs)

print(f"Documento original: {len(docs)} doc, {len(contenido)} caracteres")
print(f"Chunks generados: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i+1}: {len(chunk.page_content)} chars | {chunk.page_content[:60]}...")

print("\nCargando modelo de embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True}
)
print("Modelo listo")

os.makedirs("./data/chroma", exist_ok=True)
chroma_client = chromadb.PersistentClient(path="./data/chroma")

colecciones_existentes = [c.name for c in chroma_client.list_collections()]
if "arquitectura_datos" in colecciones_existentes:
    chroma_client.delete_collection("arquitectura_datos")
    print("Coleccion anterior eliminada")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./data/chroma",
    collection_name="arquitectura_datos",
    client=chroma_client
)
print(f"Vectorstore creado: {vectorstore._collection.count()} vectores indexados")

print("\nFASE 2: CONSULTA")
print("=" * 50)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt_rag = ChatPromptTemplate.from_messages([
    ("system", """Eres un arquitecto de datos experto.
Responde SOLO basandote en el contexto proporcionado.
Si la informacion no esta en el contexto, dilo explicitamente.

Contexto recuperado:
{context}"""),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt_rag
    | llm
    | StrOutputParser()
)

preguntas = [
    "Que almacena la capa Bronze y para que se usa?",
    "Cual es la diferencia principal entre Data Mesh y Data Lakehouse?",
    "Por que un indice vectorial es diferente a un indice B-tree?"
]

for pregunta in preguntas:
    print(f"\nPregunta: {pregunta}")
    chunks_recuperados = retriever.invoke(pregunta)
    print(f"Chunks recuperados: {len(chunks_recuperados)}")
    for i, chunk in enumerate(chunks_recuperados):
        print(f"  [{i+1}] {chunk.page_content[:80]}...")
    respuesta = rag_chain.invoke(pregunta)
    print(f"Respuesta: {respuesta}")
    print("-" * 60)
