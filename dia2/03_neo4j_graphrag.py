from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain, Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

print("Conectando a Neo4j...")
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    enhanced_schema=False
)
print("Conexión establecida")

# ============================================================
# PASO 1: Crear el grafo de arquitectura empresarial
# ============================================================
print("\nCreando grafo de arquitectura empresarial...")
graph.query("MATCH (n) DETACH DELETE n")

graph.query("""
MERGE (s1:Sistema {
    nombre: 'Data Lake',
    tipo: 'almacenamiento',
    tecnologia: 'Delta Lake',
    capa: 'Bronze/Silver',
    criticidad: 'alta'
})
MERGE (s2:Sistema {
    nombre: 'Data Warehouse',
    tipo: 'almacenamiento',
    tecnologia: 'Snowflake',
    capa: 'Gold',
    criticidad: 'alta'
})
MERGE (s3:Sistema {
    nombre: 'Pipeline ETL',
    tipo: 'procesamiento',
    tecnologia: 'dbt + Airflow',
    capa: 'transformacion',
    criticidad: 'alta'
})
MERGE (s4:Sistema {
    nombre: 'BI Dashboard',
    tipo: 'consumo',
    tecnologia: 'Power BI',
    capa: 'Gold',
    criticidad: 'media'
})
MERGE (s5:Sistema {
    nombre: 'ML Platform',
    tipo: 'ia',
    tecnologia: 'MLflow + Ray',
    capa: 'Gold',
    criticidad: 'alta'
})
MERGE (s6:Sistema {
    nombre: 'RAG System',
    tipo: 'ia',
    tecnologia: 'LangChain + Qdrant',
    capa: 'aplicacion',
    criticidad: 'media'
})
MERGE (s7:Sistema {
    nombre: 'Feature Store',
    tipo: 'ml',
    tecnologia: 'Feast',
    capa: 'Silver/Gold',
    criticidad: 'media'
})
MERGE (s1)-[:ALIMENTA {frecuencia: 'diaria', formato: 'Parquet'}]->(s3)
MERGE (s3)-[:ALIMENTA {frecuencia: 'diaria', formato: 'Delta'}]->(s2)
MERGE (s2)-[:ALIMENTA {frecuencia: 'diaria', formato: 'DirectQuery'}]->(s4)
MERGE (s2)-[:ALIMENTA {frecuencia: 'semanal', formato: 'Export'}]->(s5)
MERGE (s2)-[:ALIMENTA {frecuencia: 'batch', formato: 'Parquet'}]->(s7)
MERGE (s7)-[:ALIMENTA {frecuencia: 'online', formato: 'REST API'}]->(s5)
MERGE (s2)-[:ALIMENTA {frecuencia: 'batch', formato: 'Parquet'}]->(s6)
MERGE (s5)-[:DEPENDE_DE {tipo: 'modelo_base'}]->(s7)
""")

result = graph.query("MATCH (n:Sistema) RETURN count(n) as sistemas")
print(f"Nodos creados: {result[0]['sistemas']} sistemas")
result = graph.query("MATCH ()-[r]->() RETURN count(r) as relaciones")
print(f"Relaciones creadas: {result[0]['relaciones']} relaciones")

# ============================================================
# PASO 2: GraphCypherQAChain (Text-to-Cypher)
# ============================================================
# El LLM traduce lenguaje natural a Cypher automáticamente.
# enhanced_schema=False evita APOC pero el LLM sigue pudiendo
# generar Cypher correcto porque le pasamos el esquema manualmente.
print("\n" + "=" * 55)
print("PATRÓN 1: lenguaje natural a Cypher (Text-to-Cypher)")
print("=" * 55)

graph.refresh_schema()

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)

preguntas_grafo = [
    "¿Qué sistemas alimentan al Data Warehouse?",
    "¿Cuáles son los sistemas de tipo ia?",
    "¿Qué sistemas tienen criticidad alta?"
]

for pregunta in preguntas_grafo:
    print(f"\nPregunta: {pregunta}")
    resultado = chain.invoke(pregunta)
    print(f"Respuesta: {resultado['result']}")
    print("-" * 50)

# ============================================================
# PASO 3: Neo4j Vector Index
# ============================================================
print("\n" + "=" * 55)
print("PATRÓN 2: índice vectorial en Neo4j")
print("=" * 55)

print("Cargando modelo de embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True}
)

neo4j_vectorstore = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="sistema_embeddings",
    node_label="Sistema",
    text_node_properties=["nombre", "tecnologia", "tipo", "capa"],
    embedding_node_property="embedding"
)
print("Índice vectorial creado sobre nodos del grafo")

busquedas = [
    "plataforma de inteligencia artificial y machine learning",
    "almacenamiento y persistencia de datos",
    "visualización y reporting de negocio"
]

for query in busquedas:
    print(f"\nQuery: {query}")
    resultados = neo4j_vectorstore.similarity_search(query, k=2)
    for r in resultados:
        print(f"  {r.page_content}")

# ============================================================
# PASO 4: GraphRAG completo (vector + traversal)
# ============================================================
# Después de encontrar un nodo por similitud semántica,
# hacemos traversal del grafo para recuperar sus dependencias.
# Esto es imposible con Chroma o Qdrant solos.
print("\n" + "=" * 55)
print("PATRÓN 3: GraphRAG completo (vector + traversal)")
print("=" * 55)

def graphrag_query(pregunta: str) -> str:
    # Paso 1: búsqueda semántica para encontrar el nodo relevante
    nodos = neo4j_vectorstore.similarity_search(pregunta, k=1)
    if not nodos:
        return "No encontré nodos relevantes"

    nombre_sistema = None
    for linea in nodos[0].page_content.split("\n"):
        if "nombre:" in linea.lower():
            nombre_sistema = linea.split(":")[-1].strip()
            break

    if not nombre_sistema:
        return "No pude identificar el sistema"

    print(f"  Nodo encontrado por similitud: {nombre_sistema}")

    # Paso 2: traversal del grafo desde el nodo encontrado
    contexto_grafo = graph.query("""
        MATCH (s:Sistema {nombre: $nombre})-[r]->(d:Sistema)
        RETURN s.nombre as origen, type(r) as relacion,
               d.nombre as destino, r.frecuencia as frecuencia
        UNION
        MATCH (o:Sistema)-[r]->(s:Sistema {nombre: $nombre})
        RETURN o.nombre as origen, type(r) as relacion,
               s.nombre as destino, r.frecuencia as frecuencia
    """, params={"nombre": nombre_sistema})

    if not contexto_grafo:
        return f"{nombre_sistema} no tiene relaciones en el grafo"

    contexto_texto = f"Sistema consultado: {nombre_sistema}\n\nRelaciones:\n"
    for row in contexto_grafo:
        contexto_texto += f"  {row['origen']} --[{row['relacion']}]--> {row['destino']} (frecuencia: {row['frecuencia']})\n"

    # Paso 3: el LLM responde usando el contexto del grafo
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un arquitecto empresarial experto.
Responde usando SOLO la información del grafo proporcionada.
{contexto}"""),
        ("human", "{pregunta}")
    ])

    chain_graphrag = prompt | llm | StrOutputParser()
    return chain_graphrag.invoke({
        "contexto": contexto_texto,
        "pregunta": pregunta
    })

preguntas_graphrag = [
    "¿De qué sistemas depende el Data Warehouse y con qué frecuencia recibe datos?",
    "¿Qué sistemas consumen datos del Data Warehouse?"
]

for pregunta in preguntas_graphrag:
    print(f"\nPregunta: {pregunta}")
    respuesta = graphrag_query(pregunta)
    print(f"Respuesta: {respuesta}")
    print("-" * 50)
