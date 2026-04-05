# Agentic AI Bootcamp: De Arquitecto de Datos a Arquitecto de IA Agéntica

Formación intensiva de 6 días para profesionales de arquitectura de datos y arquitectura empresarial que quieren adquirir capacidad de delivery en sistemas de IA agéntica.

## Objetivo

Al terminar este bootcamp puedes diseñar, construir y desplegar sistemas multiagente reales con LangChain, LangGraph, CrewAI y bases de datos vectoriales. El caso de uso final es un simulador completo de Financial Crime Control (FCC) para un banco retail.

## Stack tecnológico

| Componente | Tecnología | Uso |
|---|---|---|
| LLM inference | Groq API (gratuito) | llama-3.3-70b-versatile |
| Embeddings | HuggingFace nomic-embed-text-v1 | 768 dims, local |
| Orquestación agentes | LangGraph 1.x | Flujos stateful, HITL |
| Multi-agente negocio | CrewAI 1.x | Crews y roles |
| Vector DB desarrollo | Chroma | Local, embebido |
| Vector DB producción | Qdrant (Docker) | Filtros complejos |
| Grafo de conocimiento | Neo4j (Docker + APOC) | GraphRAG, AML |
| Trazabilidad | LangSmith (gratuito) | Observabilidad |
| Gestión entornos | uv | Resolución rápida |

## Requisitos previos

- Linux (probado en Linux Mint) o macOS
- Python 3.11+
- Docker instalado y corriendo
- Cuenta gratuita en [Groq](https://console.groq.com) para la API key
- Cuenta gratuita en [LangSmith](https://smith.langchain.com) para trazabilidad

## Instalación del entorno

### 1. Instalar uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # o añadir ~/.local/bin al PATH
```

### 2. Clonar el repositorio
```bash
git clone https://github.com/mgalanme/agentic-ai-bootcamp.git
cd agentic-ai-bootcamp
```

### 3. Crear los dos entornos virtuales

**Entorno LangChain (Días 1-3 y Días 5-6):**
```bash
uv venv .venv-langchain --python 3.11
source .venv-langchain/bin/activate

uv pip install \
  langchain langchain-community langchain-groq \
  langgraph langchain-chroma langchain-qdrant \
  langchain-neo4j langchain-ollama langchain-huggingface \
  chromadb qdrant-client sentence-transformers \
  neo4j python-dotenv jupyter ipykernel \
  tavily-python pypdf tiktoken rich einops faker

deactivate
```

**Entorno CrewAI (Día 4):**
```bash
uv venv .venv-crewai --python 3.11
source .venv-crewai/bin/activate

uv pip install \
  crewai crewai-tools litellm \
  langchain-groq langchain-huggingface \
  langchain-chroma langchain-core chromadb \
  einops python-dotenv jupyter ipykernel rich

deactivate
```

### 4. Levantar los servicios Docker

**Qdrant (vector DB de producción):**
```bash
docker run -d \
  --name qdrant-dev \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/data/qdrant:/qdrant/storage \
  qdrant/qdrant
```

**Neo4j con APOC (grafo de conocimiento):**
```bash
docker run -d \
  --name neo4j-dev \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/bootcamp1234 \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:latest
```

Verificar que están corriendo:
```bash
curl -s http://localhost:6333/ | python3 -m json.tool | grep version
docker logs neo4j-dev | grep Started
```

### 5. Instalar Ollama (embeddings locales)
```bash
# Con sudo (recomendado)
curl -fsSL https://ollama.com/install.sh | sh

# Sin sudo (instalar en home)
curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64 \
  -o ~/.local/bin/ollama && chmod +x ~/.local/bin/ollama

# Descargar modelo de embeddings
ollama pull nomic-embed-text
```

### 6. Configurar variables de entorno
```bash
cp .env.example .env
# Editar .env con tus API keys
```

Contenido del `.env`:
```env
# LLM
GROQ_API_KEY=tu_api_key_de_groq

# Vector DBs
CHROMA_PERSIST_DIR=./data/chroma
QDRANT_URL=http://localhost:6333

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=bootcamp1234

# LangSmith (opcional pero recomendado)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=tu_api_key_de_langsmith
LANGCHAIN_PROJECT=agentic-bootcamp

# CrewAI
CREWAI_DISABLE_UPDATE_CHECK=1
```

### 7. Crear estructura de directorios
```bash
mkdir -p data/{chroma,chroma_adrs,chroma_integracion,docs,qdrant,fcc/expedientes}
mkdir -p {dia1,dia2,dia3,dia4,dia5,dia6}
```

## Uso diario

### Activar el entorno correcto
```bash
# Días 1-3 y 5-6: LangChain + LangGraph
source .venv-langchain/bin/activate

# Día 4: CrewAI
source .venv-crewai/bin/activate
```

### Arrancar/parar servicios
```bash
# Arrancar
docker start neo4j-dev qdrant-dev

# Parar (sin borrar datos)
docker stop neo4j-dev qdrant-dev
```

---

## Contenido del bootcamp

### Día 1: Fundamentos LangChain y RAG básico

**Entorno:** `.venv-langchain`

Construye los bloques fundamentales de cualquier sistema de IA agéntica: LCEL como lenguaje de pipelines declarativos, memoria conversacional, structured output con Pydantic, y un sistema RAG completo con Chroma.

**Scripts:**

| Script | Descripción |
|---|---|
| `dia1/01_primer_chain.py` | LCEL: prompt \| llm \| parser, invocación y streaming |
| `dia1/02_chain_con_memoria.py` | Historial conversacional y structured output Pydantic |
| `dia1/03_rag_basico.py` | RAG completo: chunking, embeddings, Chroma, retriever |

**Conceptos clave:**
- El operador `|` de LCEL encadena componentes como Unix pipes
- El LLM es stateless: la memoria se construye acumulando mensajes
- `with_structured_output` convierte el LLM en productor de datos tipados
- RAG = pipeline de datos donde la "consulta" es semántica

**Ejecutar:**
```bash
source .venv-langchain/bin/activate
python dia1/01_primer_chain.py
python dia1/02_chain_con_memoria.py
python dia1/03_rag_basico.py
```

---

### Día 2: Bases de datos vectoriales en profundidad

**Entorno:** `.venv-langchain`

Domina el ecosistema de vector DBs desde perspectiva de Arquitecto de Datos: Chroma para desarrollo, Qdrant para producción, Neo4j para GraphRAG.

**Scripts:**

| Script | Descripción |
|---|---|
| `dia2/01_chroma_avanzado.py` | Filtros metadata, scores L2, MMR (diversidad) |
| `dia2/02_qdrant_produccion.py` | Filtros AND/OR/rangos numéricos, colecciones tipadas |
| `dia2/03_neo4j_graphrag.py` | Text-to-Cypher, índice vectorial sobre nodos, GraphRAG |

**Conceptos clave:**
- Chroma: SQLite del mundo vectorial. Métrica L2 (menor = más similar)
- Qdrant: PostgreSQL del mundo vectorial. Cosine similarity (mayor = más similar)
- GraphRAG: similitud vectorial + traversal de grafo para dominios relacionales
- MMR: equilibra relevancia con diversidad en la recuperación

**Ejecutar:**
```bash
docker start qdrant-dev neo4j-dev
python dia2/01_chroma_avanzado.py
python dia2/02_qdrant_produccion.py
python dia2/03_neo4j_graphrag.py
```

---

### Día 3: LangGraph — agentes con estado

**Entorno:** `.venv-langchain`

LangGraph es el framework más importante del ecosistema para flujos agénticos complejos. A diferencia de los chains lineales de LangChain, LangGraph permite ciclos, estado persistente y human-in-the-loop.

**Scripts:**

| Script | Descripción |
|---|---|
| `dia3/01_primer_grafo.py` | Estado TypedDict, nodos, edges condicionales, ciclos |
| `dia3/02_agente_react.py` | Patrón ReAct con tools, ToolNode, circuit breaker |
| `dia3/03_human_in_the_loop.py` | MemorySaver, interrupt_before, aprobación humana |

**Conceptos clave:**
- State: esquema TypedDict compartido por todos los nodos
- Nodes: funciones que transforman el estado parcialmente
- Edges: conexiones fijas o condicionales entre nodos
- `interrupt_before`: pausa el grafo para aprobación humana
- `MemorySaver`: persiste el estado entre pausas y reanudaciones

**Ejecutar:**
```bash
python dia3/01_primer_grafo.py
python dia3/02_agente_react.py
python dia3/03_human_in_the_loop.py  # interactivo: responde s/n
```

---

### Día 4: CrewAI — equipos de agentes especializados

**Entorno:** `.venv-crewai`

CrewAI modela organizaciones virtuales de agentes especializados. Como Arquitecto Empresarial, piénsalo como BPMN pero para agentes de IA: cada Agent es un rol, cada Task es un entregable, la Crew es el proceso de negocio.

**Scripts:**

| Script | Descripción |
|---|---|
| `dia4/01_primera_crew.py` | 3 agentes secuenciales con contexto acumulado |
| `dia4/02_crew_con_tools.py` | BaseTool + Pydantic, herramientas tipadas, max_iter |
| `dia4/03_integracion_final.py` | CrewAI + RAG Chroma: consultor + revisor governance |

**Conceptos clave:**
- Agent: role + goal + backstory + tools + llm
- Task: description + expected_output + context (dependencias)
- Process.sequential: output de cada tarea como contexto de la siguiente
- BaseTool con Pydantic: contratos de datos tipados para herramientas
- LangGraph vs CrewAI: control granular vs abstracción de alto nivel

**Ejecutar:**
```bash
source .venv-crewai/bin/activate
python dia4/01_primera_crew.py
python dia4/02_crew_con_tools.py
python dia4/03_integracion_final.py
```

---

### Día 5: Simulador FCC — Módulo AML

**Entorno:** `.venv-langchain`

El corazón del proyecto ARGOS-FCC. Construye de abajo arriba el sistema de detección de lavado de dinero: datos sintéticos de un banco retail, grafo de entidades en Neo4j, RAG regulatorio sobre FATF y Ley 10/2010, y agente LangGraph con human-in-the-loop.

**Scripts:**

| Script | Descripción |
|---|---|
| `dia5/01_datos_sinteticos.py` | Genera banco retail: 80 clientes, 109 cuentas, 639 txs con patrones AML |
| `dia5/02_grafo_neo4j.py` | Carga el grafo y detecta structuring, smurfing y layering con Cypher |
| `dia5/03_rag_regulatorio.py` | Indexa FATF, AMLD6, Ley 10/2010 y SEPBLAC en Qdrant |
| `dia5/04_agente_aml.py` | Agente LangGraph 3 nodos: detector → investigador → redactor SAR |

**Patrones AML detectados:**
- **Structuring**: fragmentación por debajo del umbral de reporte (10.000€)
- **Smurfing**: múltiples orígenes concentrando fondos en una cuenta destino
- **Layering**: cadenas A→B→C→D hacia jurisdicciones FATF de riesgo (KP, IR, SY)

**Flujo del agente AML:**
1. Detector: consulta Neo4j → score inicial por patrón
2. Investigador: traversal profundo + RAG regulatorio → FichaRiesgo Pydantic
3. **PAUSA HITL**: analista revisa análisis y aprueba/rechaza
4. Redactor: genera borrador SAR en formato SEPBLAC
5. **PAUSA HITL**: oficial de riesgos aprueba/rechaza el SAR
6. Expediente guardado en `./data/fcc/expedientes/`

**Ejecutar:**
```bash
docker start neo4j-dev qdrant-dev
source .venv-langchain/bin/activate
python dia5/01_datos_sinteticos.py
python dia5/02_grafo_neo4j.py
python dia5/03_rag_regulatorio.py
python dia5/04_agente_aml.py  # interactivo: responde s/n en cada HITL
```

---

### Día 6: Simulador FCC — Módulos Fraude y KYC/KYB

**Entorno:** `.venv-langchain` + `.venv-crewai`

Completa el simulador ARGOS-FCC añadiendo detección de fraude transaccional y due diligence KYC/KYB sobre la misma infraestructura del Día 5.

**Scripts:**

| Script | Descripción |
|---|---|
| `dia6/01_scoring_fraude.py` | Agente de scoring transaccional: structured output, latencia <2s |
| `dia6/02_crew_kyc.py` | Crew CrewAI: agente verificador + agente riesgo |
| `dia6/03_simulador_completo.py` | Pipeline integrado: alerta → módulo correcto → resolución |

*(Día 6 en desarrollo)*

---

## Arquitectura del sistema ARGOS-FCC
┌─────────────────────────────────────────────────────┐
│                   ARGOS-FCC                         │
│         Agentive Risk & Governance Operating        │
│         System for Financial Crime Control          │
├──────────────┬──────────────────┬───────────────────┤
│ Módulo AML   │ Módulo Fraude    │ Módulo KYC/KYB    │
│ LangGraph    │ LangChain        │ CrewAI             │
│ stateful     │ structured output│ multi-agente       │
│ 3 nodos      │ alta frecuencia  │ 2 roles            │
├──────────────┴──────────────────┴───────────────────┤
│              Capa de datos compartida               │
│  Neo4j (grafo) · Qdrant (regulación) · Bronze JSON  │
├─────────────────────────────────────────────────────┤
│              Governance y trazabilidad              │
│  Human-in-the-Loop · LangSmith · Expedientes        │
│  FATF · AMLD6 · Ley 10/2010 · GDPR · PSD2          │
└─────────────────────────────────────────────────────┘~
## Notas de implementación

**Sobre los límites de Groq (capa gratuita):**
- 100.000 tokens/día por modelo
- 12.000 tokens/minuto (llama-3.3-70b)
- Si alcanzas el límite, espera 1 hora o cambia a `llama-3.1-8b-instant`

**Sobre los embeddings:**
- El repositorio usa HuggingFace `nomic-embed-text-v1` cargado directamente en Python
- La primera ejecución descarga el modelo (~547MB) desde HuggingFace
- Las siguientes ejecuciones usan la caché local (sin descarga)

**Sobre Neo4j:**
- La imagen estándar no incluye APOC
- Usar siempre `NEO4J_PLUGINS='["apoc"]'` en el docker run
- Neo4j tarda ~30 segundos en arrancar completamente

**Sobre Chroma:**
- Borrar la colección antes de reindexar para evitar duplicados
- El código ya incluye el patrón de borrado previo en todos los scripts

## Licencia

MIT — libre para uso educativo y profesional.

---

*Bootcamp diseñado desde los roles de Arquitecto de Datos Senior, Arquitecto Empresarial Senior y Arquitecto de IA Agéntica Senior.*
