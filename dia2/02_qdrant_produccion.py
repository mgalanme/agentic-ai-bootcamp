from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, Filter,
    FieldCondition, MatchValue, Range, MatchAny
)

load_dotenv()

# ============================================================
# Qdrant: el vector DB de producción
# ============================================================
# Diferencias clave con Chroma desde perspectiva de DA:
#
# Chroma                     Qdrant
# --------                   --------
# embebido (SQLite)          servidor dedicado (Docker/cloud)
# filtros simples            filtros de payload complejos
# sin cuantización           cuantización escalar/binaria
# sin sharding               sharding horizontal nativo
# dev/prototipado            producción real
#
# En Qdrant los "documentos" se llaman "points".
# Los "metadatos" se llaman "payload".
# Las "colecciones" son equivalentes a las "collections" de Chroma.
# La terminología cambia, el concepto es el mismo.

print("Conectando a Qdrant...")
client = QdrantClient(url="http://localhost:6333")

print("Cargando modelo de embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True}
)

# ============================================================
# PASO 1: Crear la colección con esquema explícito
# ============================================================
# A diferencia de Chroma donde el esquema es implícito,
# en Qdrant defines explícitamente el tamaño del vector
# y la métrica de distancia antes de insertar datos.
# Como DA: es como definir el DDL de una tabla antes de cargar.

collection_name = "adrs_produccion"

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)
    print("Colección anterior eliminada")

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=768,                # dimensión de nomic-embed-text-v1
        distance=Distance.COSINE # cosine similarity para texto
        # alternativas: Distance.EUCLID, Distance.DOT
    )
)
print(f"Colección '{collection_name}' creada")

# ============================================================
# PASO 2: Insertar documentos con payload rico
# ============================================================
adrs = [
    Document(
        page_content="ADR-001: Selección de base de datos vectorial. Decisión: Qdrant para producción por su soporte a filtros de payload, cuantización escalar y sharding horizontal.",
        metadata={"tipo": "ADR", "categoria": "infraestructura", "anio": 2024, "criticidad": "alta", "estado": "aprobado", "impacto_equipos": 5}
    ),
    Document(
        page_content="ADR-002: Estrategia de embeddings. Decisión: nomic-embed-text-v1 para texto en castellano e inglés. Dimensión 768. Evaluado frente a OpenAI ada-002.",
        metadata={"tipo": "ADR", "categoria": "ml", "anio": 2024, "criticidad": "alta", "estado": "aprobado", "impacto_equipos": 3}
    ),
    Document(
        page_content="ADR-003: Chunking strategy para RAG. Decisión: chunks de 400 caracteres con 40 de overlap usando RecursiveCharacterTextSplitter.",
        metadata={"tipo": "ADR", "categoria": "rag", "anio": 2024, "criticidad": "media", "estado": "aprobado", "impacto_equipos": 2}
    ),
    Document(
        page_content="ADR-004: Arquitectura de agentes. Decisión: LangGraph para flujos complejos con estado persistente. CrewAI para pipelines multi-agente de negocio.",
        metadata={"tipo": "ADR", "categoria": "agentes", "anio": 2024, "criticidad": "alta", "estado": "en_revision", "impacto_equipos": 8}
    ),
    Document(
        page_content="ADR-005: Observabilidad de agentes. Decisión: LangSmith para trazabilidad. Integración con OpenTelemetry para métricas de latencia y coste por token.",
        metadata={"tipo": "ADR", "categoria": "observabilidad", "anio": 2024, "criticidad": "media", "estado": "aprobado", "impacto_equipos": 4}
    ),
    Document(
        page_content="ADR-006: Estrategia de reindexado. Decisión: borrado completo para corpus pequeños. Upsert con IDs únicos para corpus mayores de 10000 documentos.",
        metadata={"tipo": "ADR", "categoria": "rag", "anio": 2024, "criticidad": "baja", "estado": "borrador", "impacto_equipos": 1}
    ),
    Document(
        page_content="ADR-007: Modelo de LLM para producción. Decisión: Claude Sonnet para razonamiento complejo. Groq llama-3.3-70b para tareas de alta frecuencia por coste.",
        metadata={"tipo": "ADR", "categoria": "infraestructura", "anio": 2024, "criticidad": "alta", "estado": "en_revision", "impacto_equipos": 6}
    ),
]

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)
vectorstore.add_documents(adrs)

info = client.get_collection(collection_name)
print(f"Points insertados: {info.points_count}\n")

# ============================================================
# BÚSQUEDA 1: Similitud básica
# ============================================================
print("=" * 55)
print("BÚSQUEDA 1: similitud básica")
print("=" * 55)
resultados = vectorstore.similarity_search(
    "framework para construir agentes de IA",
    k=2
)
for r in resultados:
    print(f"  [{r.metadata['categoria']}] {r.page_content[:75]}...")

# ============================================================
# BÚSQUEDA 2: Con score de similitud coseno
# ============================================================
# En Qdrant el score es cosine similarity: mayor = más similar.
# Rango de 0 a 1. Opuesto a Chroma donde era distancia L2.
print("\n" + "=" * 55)
print("BÚSQUEDA 2: con score coseno (mayor = más similar)")
print("=" * 55)
resultados_score = vectorstore.similarity_search_with_score(
    "estrategia de embeddings y vectorización de texto",
    k=3
)
for doc, score in resultados_score:
    print(f"  score {score:.4f} | [{doc.metadata['categoria']}] {doc.page_content[:65]}...")
print("  (mayor score = más similar en Qdrant, al contrario que Chroma)")

# ============================================================
# BÚSQUEDA 3: Filtros de payload simples
# ============================================================
# Los filtros de Qdrant son más expresivos que los de Chroma.
# Usa su propio DSL de filtrado con objetos Python tipados.
# must = AND, should = OR, must_not = NOT
print("\n" + "=" * 55)
print("BÚSQUEDA 3: filtro simple por campo exacto")
print("=" * 55)
resultados_filtro = vectorstore.similarity_search(
    "decisión arquitectónica crítica",
    k=5,
    filter=Filter(
        must=[
            FieldCondition(
                key="metadata.criticidad",
                match=MatchValue(value="alta")
            )
        ]
    )
)
print("Solo criticidad='alta':")
for r in resultados_filtro:
    print(f"  [{r.metadata['estado']}] [{r.metadata['categoria']}] {r.page_content[:60]}...")

# ============================================================
# BÚSQUEDA 4: Filtros de payload compuestos (AND + OR)
# ============================================================
# Aquí es donde Qdrant supera claramente a Chroma.
# Puedes combinar condiciones complejas sobre el payload.
print("\n" + "=" * 55)
print("BÚSQUEDA 4: filtro compuesto AND + OR")
print("=" * 55)
resultados_compuesto = vectorstore.similarity_search(
    "arquitectura e infraestructura de datos",
    k=5,
    filter=Filter(
        must=[
            FieldCondition(
                key="metadata.criticidad",
                match=MatchValue(value="alta")
            )
        ],
        should=[
            FieldCondition(
                key="metadata.estado",
                match=MatchValue(value="aprobado")
            ),
            FieldCondition(
                key="metadata.estado",
                match=MatchValue(value="en_revision")
            )
        ]
    )
)
print("criticidad='alta' AND (estado='aprobado' OR estado='en_revision'):")
for r in resultados_compuesto:
    print(f"  [{r.metadata['estado']}] [{r.metadata['categoria']}] impacto={r.metadata['impacto_equipos']} equipos")

# ============================================================
# BÚSQUEDA 5: Filtro por rango numérico
# ============================================================
# Chroma no soporta rangos numéricos en metadatos.
# Qdrant sí. Fundamental para filtrar por fechas, scores, etc.
print("\n" + "=" * 55)
print("BÚSQUEDA 5: filtro por rango numérico")
print("=" * 55)
resultados_rango = vectorstore.similarity_search(
    "decisión con alto impacto organizacional",
    k=5,
    filter=Filter(
        must=[
            FieldCondition(
                key="metadata.impacto_equipos",
                range=Range(gte=4)  # impacto >= 4 equipos
            )
        ]
    )
)
print("impacto_equipos >= 4:")
for r in resultados_rango:
    print(f"  impacto={r.metadata['impacto_equipos']} | [{r.metadata['categoria']}] {r.page_content[:60]}...")

# ============================================================
# INSPECCIÓN DE LA COLECCIÓN
# ============================================================
print("\n" + "=" * 55)
print("INSPECCIÓN: estado de la colección en Qdrant")
print("=" * 55)
info = client.get_collection(collection_name)
print(f"Points totales: {info.points_count}")
print(f"Dimensión vectores: {info.config.params.vectors.size}")
print(f"Métrica distancia: {info.config.params.vectors.distance}")
print(f"Estado: {info.status}")
print("\nColecciones en Qdrant:")
for col in client.get_collections().collections:
    i = client.get_collection(col.name)
    print(f"  {col.name}: {i.points_count} points")
