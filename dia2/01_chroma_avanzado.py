from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb

load_dotenv()

# ============================================================
# Chroma avanzado: metadata, filtros y tipos de búsqueda
# ============================================================
# Como DA: los metadatos en Chroma son exactamente como
# columnas adicionales en una tabla. Los vectores son la
# columna de búsqueda semántica, los metadatos son el resto
# de columnas con las que puedes filtrar.
# La combinación de ambos es lo que hace potente a un vector DB.

print("Cargando modelo de embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True}
)

# Dataset: Architecture Decision Records con metadatos ricos
# En producción estos vendrían de tu catálogo de datos,
# Confluence, o cualquier fuente documental corporativa
adrs = [
    Document(
        page_content="ADR-001: Selección de base de datos vectorial. Decisión: Qdrant para producción por su soporte a filtros de payload, cuantización escalar y capacidad de sharding horizontal.",
        metadata={"tipo": "ADR", "categoria": "infraestructura", "anio": 2024, "criticidad": "alta", "estado": "aprobado"}
    ),
    Document(
        page_content="ADR-002: Estrategia de embeddings. Decisión: nomic-embed-text-v1 para texto en castellano e inglés. Dimensión 768. Evaluado frente a OpenAI ada-002 y sentence-transformers.",
        metadata={"tipo": "ADR", "categoria": "ml", "anio": 2024, "criticidad": "alta", "estado": "aprobado"}
    ),
    Document(
        page_content="ADR-003: Chunking strategy para RAG. Decisión: chunks de 400 caracteres con 40 de overlap usando RecursiveCharacterTextSplitter con separadores jerárquicos.",
        metadata={"tipo": "ADR", "categoria": "rag", "anio": 2024, "criticidad": "media", "estado": "aprobado"}
    ),
    Document(
        page_content="ADR-004: Arquitectura de agentes. Decisión: LangGraph para flujos complejos con estado persistente. CrewAI para pipelines multi-agente orientados a negocio.",
        metadata={"tipo": "ADR", "categoria": "agentes", "anio": 2024, "criticidad": "alta", "estado": "en_revision"}
    ),
    Document(
        page_content="ADR-005: Observabilidad de agentes. Decisión: LangSmith para trazabilidad de chains y agentes. Integración con OpenTelemetry para métricas de latencia y coste.",
        metadata={"tipo": "ADR", "categoria": "observabilidad", "anio": 2024, "criticidad": "media", "estado": "aprobado"}
    ),
    Document(
        page_content="ADR-006: Estrategia de reindexado. Decisión: borrado y reconstrucción completa para corpus pequeños. Upsert con IDs únicos para corpus mayores de 10000 documentos.",
        metadata={"tipo": "ADR", "categoria": "rag", "anio": 2024, "criticidad": "baja", "estado": "borrador"}
    ),
    Document(
        page_content="ADR-007: Modelo de LLM para producción. Decisión: Claude Sonnet para tareas complejas de razonamiento. Groq llama-3.3-70b para tareas de alta frecuencia por coste.",
        metadata={"tipo": "ADR", "categoria": "infraestructura", "anio": 2024, "criticidad": "alta", "estado": "en_revision"}
    ),
]

# Crear vectorstore limpio
chroma_client = chromadb.PersistentClient(path="./data/chroma_adrs")
if "adrs" in [c.name for c in chroma_client.list_collections()]:
    chroma_client.delete_collection("adrs")

vectorstore = Chroma.from_documents(
    documents=adrs,
    embedding=embeddings,
    persist_directory="./data/chroma_adrs",
    collection_name="adrs",
    client=chroma_client
)
print(f"Vectorstore creado: {vectorstore._collection.count()} documentos\n")

# ============================================================
# TIPO 1: Búsqueda por similitud básica
# ============================================================
# La más simple: dame los k documentos más parecidos
print("=" * 55)
print("TIPO 1: similitud semántica básica")
print("=" * 55)
resultados = vectorstore.similarity_search(
    "decisión sobre qué framework usar para agentes de IA",
    k=2
)
for r in resultados:
    print(f"  [{r.metadata['categoria']}] {r.page_content[:80]}...")

# ============================================================
# TIPO 2: Búsqueda con score de similitud
# ============================================================
# Igual que el anterior pero devuelve el score de similitud.
# En Chroma el score es distancia L2: menor = más similar.
# Útil para filtrar resultados por umbral de calidad.
print("\n" + "=" * 55)
print("TIPO 2: similitud con score (distancia L2)")
print("=" * 55)
resultados_score = vectorstore.similarity_search_with_score(
    "estrategia de embeddings y vectorización",
    k=3
)
for doc, score in resultados_score:
    print(f"  score {score:.4f} | [{doc.metadata['categoria']}] {doc.page_content[:70]}...")
print("  (menor score = más similar en Chroma)")

# ============================================================
# TIPO 3: Filtrado por metadata
# ============================================================
# Combina búsqueda semántica con filtros exactos sobre metadatos.
# Como DA: es exactamente un WHERE sobre las columnas de metadatos
# aplicado ANTES de la búsqueda semántica.
# Solo busca semánticamente dentro de los documentos que
# pasan el filtro de metadata.
print("\n" + "=" * 55)
print("TIPO 3: filtrado por metadata (WHERE sobre metadatos)")
print("=" * 55)

# Filtro simple: solo ADRs de criticidad alta
resultados_filtro = vectorstore.similarity_search(
    "decisión arquitectónica importante",
    k=3,
    filter={"criticidad": "alta"}
)
print("Solo criticidad='alta':")
for r in resultados_filtro:
    print(f"  [{r.metadata['estado']}] {r.metadata['categoria']} | {r.page_content[:65]}...")

# Filtro por estado: solo aprobados
resultados_aprobados = vectorstore.similarity_search(
    "decisión arquitectónica",
    k=5,
    filter={"estado": "aprobado"}
)
print("\nSolo estado='aprobado':")
for r in resultados_aprobados:
    print(f"  [{r.metadata['criticidad']}] {r.metadata['categoria']} | {r.page_content[:65]}...")

# ============================================================
# TIPO 4: MMR (Maximal Marginal Relevance)
# ============================================================
# Resuelve el problema de redundancia en la recuperación.
# similarity_search puede devolver chunks muy parecidos entre sí.
# MMR equilibra relevancia con diversidad:
# busca documentos relevantes que además sean distintos entre sí.
# Como DA: es como un DISTINCT semántico sobre los resultados.
print("\n" + "=" * 55)
print("TIPO 4: MMR (diversidad + relevancia)")
print("=" * 55)
resultados_mmr = vectorstore.max_marginal_relevance_search(
    "decisiones de arquitectura de datos e IA",
    k=3,
    fetch_k=7,    # candidatos iniciales a evaluar
    lambda_mult=0.5  # 0=máxima diversidad, 1=máxima relevancia
)
print("Resultados MMR (relevantes Y diversos):")
for r in resultados_mmr:
    print(f"  [{r.metadata['categoria']}] {r.page_content[:75]}...")

# ============================================================
# COMPARATIVA: similarity vs MMR
# ============================================================
print("\n" + "=" * 55)
print("COMPARATIVA: similarity vs MMR con misma query")
print("=" * 55)
query = "infraestructura y plataforma de datos"

sim = vectorstore.similarity_search(query, k=3)
mmr = vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=7, lambda_mult=0.4)

print("Similarity (puede repetir categorías):")
for r in sim:
    print(f"  [{r.metadata['categoria']}]")

print("MMR (fuerza diversidad entre resultados):")
for r in mmr:
    print(f"  [{r.metadata['categoria']}]")
