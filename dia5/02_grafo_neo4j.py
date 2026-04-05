"""
ARGOS-FCC · Script 02: Construcción del grafo de entidades en Neo4j
====================================================================
Carga el dataset Bronze en Neo4j como grafo de relaciones.
Después ejecuta queries Cypher que demuestran por qué el grafo
detecta patrones AML invisibles en datos tabulares.

Como DA: este es tu modelo de dominio en grafo. Cada entidad
del banco es un nodo. Cada relación financiera es un edge.
Los patrones de lavado emergen como estructuras del grafo.
"""
import json
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from collections import defaultdict

load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "bootcamp1234")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ============================================================
# CARGAR DATOS
# ============================================================

def cargar_json(nombre):
    with open(f"./data/fcc/{nombre}.json", encoding="utf-8") as f:
        return json.load(f)

clientes     = cargar_json("clientes")
cuentas      = cargar_json("cuentas")
transacciones = cargar_json("transacciones")

print("ARGOS-FCC · Construyendo grafo Neo4j")
print("=" * 55)

# ============================================================
# LIMPIAR Y CREAR ÍNDICES
# ============================================================

with driver.session() as s:
    s.run("MATCH (n) DETACH DELETE n")
    print("Grafo limpiado")

    # Índices para búsqueda eficiente (como índices en SQL)
    s.run("CREATE INDEX cliente_id IF NOT EXISTS FOR (c:Cliente) ON (c.id)")
    s.run("CREATE INDEX cuenta_id IF NOT EXISTS FOR (c:Cuenta) ON (c.id)")
    s.run("CREATE INDEX tx_id IF NOT EXISTS FOR (t:Transaccion) ON (t.id)")
    s.run("CREATE INDEX tx_patron IF NOT EXISTS FOR (t:Transaccion) ON (t.patron_riesgo)")
    print("Índices creados")

# ============================================================
# CARGAR CLIENTES
# ============================================================

print("\nCargando nodos...")
BATCH = 50

with driver.session() as s:
    for i in range(0, len(clientes), BATCH):
        lote = clientes[i:i+BATCH]
        s.run("""
        UNWIND $rows AS row
        MERGE (c:Cliente {id: row.id})
        SET c.nombre        = row.nombre,
            c.tipo          = row.tipo,
            c.pais          = row.pais_residencia,
            c.riesgo        = row.segmento_riesgo,
            c.pep           = row.pep,
            c.fecha_alta    = row.fecha_alta,
            c.nif           = row.nif
        """, rows=lote)
print(f"  Clientes cargados:     {len(clientes)}")

# ============================================================
# CARGAR CUENTAS Y RELACIÓN TITULAR_DE
# ============================================================

with driver.session() as s:
    for i in range(0, len(cuentas), BATCH):
        lote = cuentas[i:i+BATCH]
        s.run("""
        UNWIND $rows AS row
        MERGE (c:Cuenta {id: row.id})
        SET c.tipo          = row.tipo,
            c.divisa        = row.divisa,
            c.estado        = row.estado,
            c.saldo_inicial = row.saldo_inicial,
            c.id_cliente    = row.id_cliente
        WITH c, row
        MATCH (cl:Cliente {id: row.id_cliente})
        MERGE (cl)-[:TITULAR_DE]->(c)
        """, rows=lote)
print(f"  Cuentas cargadas:      {len(cuentas)}")

# ============================================================
# CARGAR TRANSACCIONES Y RELACIONES
# ============================================================
# Cada transacción crea tres relaciones:
# (Cuenta)-[:ORIGEN_DE]->(Transaccion)
# (Transaccion)-[:DESTINO_A]->(Cuenta)
# Esto permite traversal bidireccional en el grafo

with driver.session() as s:
    for i in range(0, len(transacciones), BATCH):
        lote = transacciones[i:i+BATCH]
        s.run("""
        UNWIND $rows AS row
        MERGE (t:Transaccion {id: row.id})
        SET t.importe       = row.importe,
            t.divisa        = row.divisa,
            t.fecha         = row.fecha,
            t.canal         = row.canal,
            t.pais_origen   = row.pais_origen,
            t.pais_destino  = row.pais_destino,
            t.tipo          = row.tipo,
            t.descripcion   = row.descripcion,
            t.patron_riesgo = row.patron_riesgo
        WITH t, row
        MATCH (origen:Cuenta  {id: row.id_cuenta_origen})
        MATCH (destino:Cuenta {id: row.id_cuenta_destino})
        MERGE (origen)-[:ORIGEN_DE]->(t)
        MERGE (t)-[:DESTINO_A]->(destino)
        """, rows=lote)
print(f"  Transacciones cargadas:{len(transacciones)}")

# ============================================================
# VERIFICAR
# ============================================================

with driver.session() as s:
    r = s.run("MATCH (n) RETURN labels(n)[0] AS tipo, count(*) AS n ORDER BY n DESC")
    print("\nNodos en el grafo:")
    for rec in r:
        print(f"  {rec['tipo']:<15} {rec['n']:>5}")

    r = s.run("MATCH ()-[r]->() RETURN type(r) AS tipo, count(*) AS n ORDER BY n DESC")
    print("Relaciones en el grafo:")
    for rec in r:
        print(f"  {rec['tipo']:<15} {rec['n']:>5}")

# ============================================================
# QUERIES AML: EL PODER DEL GRAFO
# ============================================================
# Aquí está el valor diferencial. Estas queries detectan
# patrones que son invisibles en datos tabulares.

print("\n" + "=" * 55)
print("DETECCIÓN DE PATRONES AML VÍA TRAVERSAL DE GRAFO")
print("=" * 55)

with driver.session() as s:

    # ── PATRÓN 1: STRUCTURING ────────────────────────────────
    # Cuentas que realizaron múltiples transferencias
    # de importe similar (8.000-10.000€) en poco tiempo.
    # En SQL necesitarías varias subqueries complejas.
    # En Cypher es un pattern match directo.
    print("\nPatrón 1: Structuring (múltiples tx por debajo del umbral)")
    r = s.run("""
        MATCH (origen:Cuenta)-[:ORIGEN_DE]->(t:Transaccion)
        WHERE t.patron_riesgo = 'structuring'
        WITH origen, count(t) AS n_txs, sum(t.importe) AS total,
             collect(t.importe)[0..3] AS muestra_importes
        WHERE n_txs >= 3
        RETURN origen.id AS cuenta,
               n_txs     AS num_transacciones,
               round(total) AS total_eur,
               muestra_importes
        ORDER BY n_txs DESC
        LIMIT 5
    """)
    for rec in r:
        print(f"  Cuenta {rec['cuenta']}: {rec['num_transacciones']} txs "
              f"· total {rec['total_eur']:,.0f}€ "
              f"· importes: {[round(x) for x in rec['muestra_importes']]}")

    # ── PATRÓN 2: SMURFING ───────────────────────────────────
    # Cuentas que reciben fondos de muchos orígenes distintos
    # en un período corto. El grafo lo detecta con un
    # simple conteo de aristas entrantes únicas.
    print("\nPatrón 2: Smurfing (muchos orígenes → una cuenta destino)")
    r = s.run("""
        MATCH (origen:Cuenta)-[:ORIGEN_DE]->(t:Transaccion)-[:DESTINO_A]->(destino:Cuenta)
        WHERE t.patron_riesgo = 'smurfing'
        WITH destino, count(DISTINCT origen) AS n_origenes,
             sum(t.importe) AS total, count(t) AS n_txs
        WHERE n_origenes >= 5
        RETURN destino.id    AS cuenta_destino,
               n_origenes    AS origenes_distintos,
               n_txs         AS num_transacciones,
               round(total)  AS total_recibido_eur
        ORDER BY n_origenes DESC
        LIMIT 3
    """)
    for rec in r:
        print(f"  Cuenta destino {rec['cuenta_destino']}: "
              f"{rec['origenes_distintos']} orígenes distintos · "
              f"{rec['num_transacciones']} txs · "
              f"{rec['total_recibido_eur']:,.0f}€ recibidos")

    # ── PATRÓN 3: LAYERING ───────────────────────────────────
    # Cadenas de transferencias: A→B→C→D
    # El grafo puede hacer path traversal de N saltos.
    # En SQL esto requeriría CTEs recursivos o joins múltiples.
    print("\nPatrón 3: Layering (cadenas de transferencias opacas)")
    r = s.run("""
        MATCH path = (c1:Cuenta)-[:ORIGEN_DE]->(t1:Transaccion {patron_riesgo: 'layering'})
                     -[:DESTINO_A]->(c2:Cuenta)
                     -[:ORIGEN_DE]->(t2:Transaccion {patron_riesgo: 'layering'})
                     -[:DESTINO_A]->(c3:Cuenta)
        RETURN c1.id AS inicio, c2.id AS intermedio, c3.id AS destino,
               round(t1.importe) AS importe1,
               round(t2.importe) AS importe2,
               t2.pais_destino   AS pais_final
        LIMIT 5
    """)
    for rec in r:
        print(f"  {rec['inicio']} →({rec['importe1']:,.0f}€)→ "
              f"{rec['intermedio']} →({rec['importe2']:,.0f}€)→ "
              f"{rec['destino']} [país: {rec['pais_final']}]")

    # ── PATRÓN 4: CLIENTES DE ALTO RIESGO CON ACTIVIDAD ──────
    # El grafo conecta directamente perfil del cliente con
    # su actividad transaccional. No hay JOIN necesario.
    print("\nPatrón 4: Clientes alto riesgo con mayor volumen")
    r = s.run("""
        MATCH (cli:Cliente {riesgo: 'alto'})
              -[:TITULAR_DE]->(cta:Cuenta)
              -[:ORIGEN_DE]->(t:Transaccion)
        WITH cli, sum(t.importe) AS volumen, count(t) AS n_txs
        WHERE n_txs > 0
        RETURN cli.id     AS cliente,
               cli.nombre AS nombre,
               cli.pais   AS pais,
               cli.pep    AS es_pep,
               n_txs      AS transacciones,
               round(volumen) AS volumen_total_eur
        ORDER BY volumen DESC
        LIMIT 5
    """)
    for rec in r:
        pep_flag = " ⚠ PEP" if rec['es_pep'] else ""
        print(f"  {rec['cliente']} [{rec['pais']}]{pep_flag}: "
              f"{rec['transacciones']} txs · "
              f"{rec['volumen_total_eur']:,.0f}€")

    # ── ESTADÍSTICA FINAL ────────────────────────────────────
    print("\nResumen del grafo AML:")
    r = s.run("""
        MATCH (t:Transaccion)
        WHERE t.patron_riesgo <> 'normal'
        RETURN t.patron_riesgo AS patron, count(*) AS n,
               round(sum(t.importe)) AS volumen_eur
        ORDER BY volumen_eur DESC
    """)
    for rec in r:
        print(f"  {rec['patron']:<15} {rec['n']:>3} txs · "
              f"{rec['volumen_eur']:>12,.0f}€")

driver.close()
print("\nGrafo Neo4j listo. Pipeline de datos AML operativo.")
