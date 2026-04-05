"""
ARGOS-FCC · Script 03: RAG regulatorio
=======================================
Indexa documentación regulatoria AML en Qdrant.
Los agentes consultarán este índice para fundamentar
sus recomendaciones en normativa concreta.

Como DA: esta es tu colección vectorial de regulación.
Como AI Architect: es el knowledge base que convierte
al agente investigador en un experto regulatorio.
"""
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

print("ARGOS-FCC · Indexando documentación regulatoria")
print("=" * 55)

# ============================================================
# MODELO DE EMBEDDINGS
# ============================================================

print("Cargando modelo de embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True}
)
print("Modelo listo")

# ============================================================
# CORPUS REGULATORIO AML
# ============================================================
# Fragmentos clave de FATF, AMLD6 y Ley 10/2010.
# En producción indexarías los documentos completos.
# Aquí usamos extractos representativos para el simulador.

docs_regulatorios = [

    # ── FATF ──────────────────────────────────────────────
    Document(
        page_content="""FATF Recommendation 1: Assessing Risks and Applying a Risk-Based Approach.
Countries should identify, assess, and understand the money laundering and terrorist financing
risks for the country. Based on that assessment, countries should apply a risk-based approach
to ensure that measures to prevent or mitigate money laundering and terrorist financing
are commensurate with the risks identified.""",
        metadata={"fuente": "FATF", "articulo": "R.1", "categoria": "enfoque_riesgo", "idioma": "en"}
    ),
    Document(
        page_content="""FATF Recommendation 10: Customer Due Diligence.
Financial institutions should be prohibited from keeping anonymous accounts or accounts
in obviously fictitious names. Financial institutions should be required to undertake
customer due diligence (CDD) measures when establishing business relations,
carrying out occasional transactions above the applicable threshold (USD/EUR 15,000),
or when there is a suspicion of money laundering or terrorist financing.""",
        metadata={"fuente": "FATF", "articulo": "R.10", "categoria": "KYC", "idioma": "en"}
    ),
    Document(
        page_content="""FATF Recommendation 20: Reporting of Suspicious Transactions.
If a financial institution suspects or has reasonable grounds to suspect that funds are
the proceeds of a criminal activity or are related to terrorist financing, it should be
required, directly by law or regulation, to report promptly its suspicions to the
Financial Intelligence Unit (FIU). The reporting obligation must apply regardless of
the amount of the transaction.""",
        metadata={"fuente": "FATF", "articulo": "R.20", "categoria": "SAR", "idioma": "en"}
    ),
    Document(
        page_content="""FATF Typology: Structuring (Smurfing).
Structuring involves breaking up large amounts of currency into smaller, less suspicious
amounts, which are then deposited into one or more bank accounts or used to purchase
monetary instruments. Threshold: transactions just below EUR 10,000 to avoid mandatory
reporting requirements. Red flags: multiple transactions of similar amounts below threshold,
same day or consecutive days, same originator, multiple beneficiaries.""",
        metadata={"fuente": "FATF", "articulo": "Typologies", "categoria": "structuring", "idioma": "en"}
    ),
    Document(
        page_content="""FATF Typology: Layering through Wire Transfers.
Layering involves the movement of money through a series of financial transactions to
distance it from its criminal source. Common techniques include: wire transfers through
multiple accounts in different countries, use of shell companies in secrecy jurisdictions,
back-to-back loans, and securities transactions. Red flags: funds passing through 3+ accounts
within short timeframes, transactions to/from high-risk jurisdictions (Iran, North Korea,
Syria, Panama, British Virgin Islands), decreasing amounts suggesting fee deduction.""",
        metadata={"fuente": "FATF", "articulo": "Typologies", "categoria": "layering", "idioma": "en"}
    ),
    Document(
        page_content="""FATF High-Risk Jurisdictions subject to a Call for Action (Black List).
As of 2024: Democratic People's Republic of Korea (DPRK/North Korea), Iran, Myanmar.
Jurisdictions under Increased Monitoring (Grey List) include: Syria, Panama, British
Virgin Islands, Liberia. Financial institutions must apply enhanced due diligence for
transactions involving these jurisdictions. Any transaction with DPRK or Iran requires
immediate reporting to the FIU regardless of amount.""",
        metadata={"fuente": "FATF", "articulo": "ICTF", "categoria": "jurisdicciones_riesgo", "idioma": "en"}
    ),

    # ── AMLD6 ─────────────────────────────────────────────
    Document(
        page_content="""Directiva AMLD6 (2018/1673/UE) - Artículo 2: Infracciones Determinantes.
AMLD6 amplía la lista de infracciones determinantes del blanqueo de capitales a 22 categorías,
incluyendo delitos cibernéticos, delitos fiscales, tráfico de influencias y participación en
grupos delictivos organizados. Esto amplía significativamente el alcance de las obligaciones
de reporte de las entidades financieras respecto a las directivas anteriores.""",
        metadata={"fuente": "AMLD6", "articulo": "Art.2", "categoria": "infracciones", "idioma": "es"}
    ),
    Document(
        page_content="""Directiva AMLD6 (2018/1673/UE) - Responsabilidad de Personas Jurídicas.
Por primera vez, AMLD6 establece responsabilidad penal directa para las personas jurídicas
(empresas) por delitos de blanqueo de capitales cometidos en su beneficio por cualquier
persona que ejerza un poder directivo. Las sanciones incluyen: exclusión de ayudas públicas,
prohibición temporal de actividades comerciales, y liquidación judicial.""",
        metadata={"fuente": "AMLD6", "articulo": "Art.7-8", "categoria": "sanciones", "idioma": "es"}
    ),

    # ── LEY 10/2010 (ESPAÑA) ──────────────────────────────
    Document(
        page_content="""Ley 10/2010 de Prevención del Blanqueo de Capitales (España) - Art. 18: SAR.
Los sujetos obligados examinarán con especial atención cualquier operación que, con independencia
de su cuantía, pueda estar relacionada con el blanqueo de capitales o la financiación del terrorismo.
Cuando se aprecie indicio o certeza de que una operación está relacionada, lo comunicarán
por iniciativa propia al SEPBLAC. El incumplimiento de la obligación de comunicación constituye
infracción muy grave con multas de hasta 10 millones de euros o el 10% del volumen de negocio.""",
        metadata={"fuente": "Ley10/2010", "articulo": "Art.18", "categoria": "SAR_obligatorio", "idioma": "es"}
    ),
    Document(
        page_content="""Ley 10/2010 de Prevención del Blanqueo de Capitales (España) - Art. 26: Umbral.
Con carácter general, los sujetos obligados identificarán a cuantas personas físicas o jurídicas
pretendan establecer relaciones de negocio o intervenir en cualesquiera operaciones.
Umbral de identificación reforzada: operaciones de pago en efectivo por importe igual o superior
a 10.000 euros. Para transferencias internacionales: umbral de 1.000 euros para jurisdicciones
de riesgo o cuando existan indicios de blanqueo independientemente del importe.""",
        metadata={"fuente": "Ley10/2010", "articulo": "Art.26", "categoria": "umbrales", "idioma": "es"}
    ),
    Document(
        page_content="""Ley 10/2010 - Personas con Responsabilidad Pública (PEPs).
Los sujetos obligados aplicarán, además de las medidas normales de diligencia debida,
medidas reforzadas de diligencia debida en las relaciones de negocio u operaciones de
personas con responsabilidad pública (PEP), sus familiares y allegados.
Se consideran PEPs: jefes de Estado, parlamentarios, miembros del Gobierno, magistrados
del Tribunal Supremo, directivos de bancos centrales, embajadores, altos cargos militares,
miembros de órganos de dirección de empresas de titularidad estatal.""",
        metadata={"fuente": "Ley10/2010", "articulo": "Art.14", "categoria": "PEPs", "idioma": "es"}
    ),
    Document(
        page_content="""Ley 10/2010 - Conservación de documentos y trazabilidad.
Los sujetos obligados conservarán durante un período mínimo de diez años la documentación
en que se formalice el cumplimiento de las obligaciones establecidas en la presente Ley.
En particular: documentos de identificación del cliente, documentos acreditativos de las
operaciones, registros de comunicaciones al SEPBLAC, y documentación de las medidas de
diligencia debida aplicadas. La falta de conservación constituye infracción grave.""",
        metadata={"fuente": "Ley10/2010", "articulo": "Art.25", "categoria": "conservacion", "idioma": "es"}
    ),

    # ── SEÑALES DE ALERTA ─────────────────────────────────
    Document(
        page_content="""Red Flags AML - Señales de alerta en transferencias bancarias.
Alto riesgo: transferencias frecuentes a jurisdicciones de lista negra FATF (Corea del Norte,
Irán) independientemente del importe. Riesgo medio-alto: múltiples transferencias de importes
similares ligeramente inferiores a 10.000€ en un período de 30 días (structuring).
Riesgo medio: cliente con domicilio en jurisdicción gris FATF que realiza transferencias
internacionales superiores a 1.000€. Riesgo bajo-medio: cambios bruscos en el patrón
habitual de transacciones sin justificación económica aparente.""",
        metadata={"fuente": "SEPBLAC", "articulo": "Guia_Red_Flags", "categoria": "red_flags", "idioma": "es"}
    ),
    Document(
        page_content="""Red Flags KYC - Señales de alerta en due diligence de clientes.
Señales de alerta que requieren diligencia debida reforzada: cliente que se niega a
proporcionar información sobre el origen de los fondos, cliente que actúa como intermediario
sin revelar la identidad del beneficiario final, cliente PEP sin justificación del origen
del patrimonio, empresa constituida en jurisdicción opaca sin actividad económica aparente,
cliente que realiza operaciones fuera de su perfil de negocio habitual.""",
        metadata={"fuente": "SEPBLAC", "articulo": "Guia_KYC", "categoria": "KYC_red_flags", "idioma": "es"}
    ),
]

# ============================================================
# INDEXAR EN QDRANT
# ============================================================

client = QdrantClient(url="http://localhost:6333")

collection_name = "fcc_regulacion"
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)
    print("Colección anterior eliminada")

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)
vectorstore.add_documents(docs_regulatorios)

info = client.get_collection(collection_name)
print(f"\nColección '{collection_name}': {info.points_count} documentos indexados")

# ============================================================
# VERIFICAR CON BÚSQUEDAS DE PRUEBA
# ============================================================

print("\n" + "=" * 55)
print("VERIFICACIÓN: búsquedas semánticas sobre regulación")
print("=" * 55)

pruebas = [
    ("transferencias a Corea del Norte e Irán", 2),
    ("obligación de reportar operaciones sospechosas al SEPBLAC", 2),
    ("múltiples transferencias por debajo de 10.000 euros", 2),
    ("personas políticamente expuestas PEP diligencia reforzada", 2),
]

for query, k in pruebas:
    print(f"\nQuery: '{query}'")
    resultados = vectorstore.similarity_search(query, k=k)
    for r in resultados:
        print(f"  [{r.metadata['fuente']} · {r.metadata['articulo']}] "
              f"{r.metadata['categoria']} · "
              f"{r.page_content[:80]}...")

print("\n" + "=" * 55)
print("RAG regulatorio operativo. Los agentes AML pueden")
print("fundamentar sus recomendaciones en normativa concreta.")
