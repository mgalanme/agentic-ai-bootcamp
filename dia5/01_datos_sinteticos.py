"""
ARGOS-FCC · Script 01: Generador de datos sintéticos del banco retail
=====================================================================
Genera un dataset realista de un banco retail español con patrones
de riesgo FCC incrustados: structuring, smurfing y layering.

Como DA: esto es tu capa Bronze. Datos raw, inmutables, que
alimentarán el pipeline Medallion y el grafo Neo4j.
"""
from faker import Faker
from faker.providers import bank, person, address
import random
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List

random.seed(42)
fake = Faker("es_ES")
Faker.seed(42)

os.makedirs("./data/fcc", exist_ok=True)

# ============================================================
# ESQUEMA DE ENTIDADES
# ============================================================

@dataclass
class Cliente:
    id: str
    nombre: str
    tipo: str          # particular | empresa
    pais_residencia: str
    segmento_riesgo: str   # bajo | medio | alto
    pep: bool
    fecha_alta: str
    nif: str

@dataclass
class Cuenta:
    id: str
    id_cliente: str
    tipo: str          # corriente | ahorro | empresa
    divisa: str
    fecha_apertura: str
    estado: str        # activa | bloqueada | cerrada
    saldo_inicial: float

@dataclass
class Transaccion:
    id: str
    id_cuenta_origen: str
    id_cuenta_destino: str
    importe: float
    divisa: str
    fecha: str
    canal: str         # online | oficina | cajero | banca_movil
    pais_origen: str
    pais_destino: str
    tipo: str          # transferencia | pago | efectivo | divisa
    descripcion: str
    patron_riesgo: str # normal | structuring | smurfing | layering | ninguno

# ============================================================
# GENERADORES
# ============================================================

PAISES_RIESGO = ["PA", "VG", "KY", "LR", "IR", "KP", "SY"]
PAISES_NORMALES = ["ES", "ES", "ES", "ES", "FR", "DE", "GB", "IT", "PT"]

def generar_clientes(n: int = 80) -> List[Cliente]:
    clientes = []
    for i in range(n):
        tipo = random.choices(["particular", "empresa"], weights=[70, 30])[0]
        pais = random.choices(
            PAISES_NORMALES + PAISES_RIESGO,
            weights=[10]*len(PAISES_NORMALES) + [1]*len(PAISES_RIESGO)
        )[0]
        riesgo = "alto" if pais in PAISES_RIESGO else random.choices(
            ["bajo", "medio", "alto"], weights=[60, 30, 10])[0]

        clientes.append(Cliente(
            id=f"CLI-{i+1:04d}",
            nombre=fake.company() if tipo == "empresa" else fake.name(),
            tipo=tipo,
            pais_residencia=pais,
            segmento_riesgo=riesgo,
            pep=random.random() < 0.03,  # 3% son PEPs
            fecha_alta=fake.date_between(
                start_date="-5y", end_date="-1m").isoformat(),
            nif=fake.nif() if tipo == "particular" else fake.cif()
        ))
    return clientes

def generar_cuentas(clientes: List[Cliente]) -> List[Cuenta]:
    cuentas = []
    for cliente in clientes:
        # Cada cliente tiene entre 1 y 3 cuentas
        n_cuentas = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
        for j in range(n_cuentas):
            tipo = "empresa" if cliente.tipo == "empresa" else random.choice(
                ["corriente", "ahorro"])
            cuentas.append(Cuenta(
                id=f"CTA-{len(cuentas)+1:05d}",
                id_cliente=cliente.id,
                tipo=tipo,
                divisa=random.choices(["EUR", "USD", "GBP"], weights=[85, 10, 5])[0],
                fecha_apertura=cliente.fecha_alta,
                estado=random.choices(
                    ["activa", "bloqueada", "cerrada"],
                    weights=[90, 5, 5])[0],
                saldo_inicial=round(random.uniform(500, 150_000), 2)
            ))
    return cuentas

def generar_transacciones_normales(
        cuentas: List[Cuenta], n: int = 600) -> List[Transaccion]:
    """Transacciones cotidianas sin patrones de riesgo."""
    txs = []
    ids_cuentas = [c.id for c in cuentas]
    for i in range(n):
        origen = random.choice(ids_cuentas)
        destino = random.choice([c for c in ids_cuentas if c != origen])
        fecha = fake.date_time_between(start_date="-6m", end_date="now")
        txs.append(Transaccion(
            id=f"TX-{len(txs)+1:06d}",
            id_cuenta_origen=origen,
            id_cuenta_destino=destino,
            importe=round(random.uniform(10, 8_000), 2),
            divisa="EUR",
            fecha=fecha.isoformat(),
            canal=random.choices(
                ["online", "oficina", "cajero", "banca_movil"],
                weights=[40, 20, 15, 25])[0],
            pais_origen="ES",
            pais_destino=random.choices(
                PAISES_NORMALES, weights=[8]*len(PAISES_NORMALES))[0],
            tipo=random.choices(
                ["transferencia", "pago", "efectivo"],
                weights=[50, 40, 10])[0],
            descripcion=fake.sentence(nb_words=4),
            patron_riesgo="normal"
        ))
    return txs

def generar_structuring(cuentas: List[Cuenta]) -> List[Transaccion]:
    """
    Structuring (pitufeo): fragmentar una cantidad grande
    en múltiples transacciones pequeñas por debajo del umbral
    de reporte (10.000€ en España) para evitar la detección.
    Patrón: mismo origen, destinos distintos, importes similares
    ligeramente por debajo de 10.000€, en pocos días.
    """
    txs = []
    ids_cuentas = [c.id for c in cuentas]
    # Seleccionar 3 cuentas origen sospechosas
    origenes = random.sample(ids_cuentas, 3)
    base_fecha = datetime.now() - timedelta(days=30)

    for origen in origenes:
        # 5-8 transferencias de ~9.000-9.900€ en 10 días
        n_txs = random.randint(5, 8)
        for j in range(n_txs):
            destino = random.choice([c for c in ids_cuentas if c != origen])
            fecha = base_fecha + timedelta(days=j*1.5, hours=random.randint(0, 12))
            txs.append(Transaccion(
                id=f"TX-{len(txs)+900:06d}",
                id_cuenta_origen=origen,
                id_cuenta_destino=destino,
                importe=round(random.uniform(8_500, 9_800), 2),
                divisa="EUR",
                fecha=fecha.isoformat(),
                canal="online",
                pais_origen="ES",
                pais_destino="ES",
                tipo="transferencia",
                descripcion="Pago servicios",
                patron_riesgo="structuring"
            ))
    return txs

def generar_smurfing(cuentas: List[Cuenta]) -> List[Transaccion]:
    """
    Smurfing: múltiples personas (pitufos) realizan pequeñas
    transacciones hacia una cuenta destino central.
    Patrón: muchos orígenes distintos, mismo destino,
    importes pequeños, en poco tiempo.
    """
    txs = []
    ids_cuentas = [c.id for c in cuentas]
    # Una cuenta destino central recibe de muchos orígenes
    destino_central = random.choice(ids_cuentas)
    origenes = random.sample(
        [c for c in ids_cuentas if c != destino_central], 12)
    base_fecha = datetime.now() - timedelta(days=15)

    for j, origen in enumerate(origenes):
        fecha = base_fecha + timedelta(hours=j*3)
        txs.append(Transaccion(
            id=f"TX-{len(txs)+800:06d}",
            id_cuenta_origen=origen,
            id_cuenta_destino=destino_central,
            importe=round(random.uniform(800, 2_500), 2),
            divisa="EUR",
            fecha=fecha.isoformat(),
            canal=random.choice(["online", "banca_movil"]),
            pais_origen="ES",
            pais_destino="ES",
            tipo="transferencia",
            descripcion="Transferencia personal",
            patron_riesgo="smurfing"
        ))
    return txs

def generar_layering(cuentas: List[Cuenta]) -> List[Transaccion]:
    """
    Layering: cadena de transferencias entre cuentas para
    ocultar el origen de los fondos. Patrón: A->B->C->D
    con importes que se van reduciendo ligeramente
    (simulando comisiones de intermediarios) e involucran
    cuentas en jurisdicciones de riesgo.
    """
    txs = []
    ids_cuentas = [c.id for c in cuentas]
    # Crear 2 cadenas de layering
    for cadena in range(2):
        eslabones = random.sample(ids_cuentas, 5)
        importe = round(random.uniform(25_000, 80_000), 2)
        base_fecha = datetime.now() - timedelta(days=45 + cadena*15)

        for j in range(len(eslabones) - 1):
            importe = round(importe * random.uniform(0.92, 0.98), 2)
            fecha = base_fecha + timedelta(days=j*3 + random.randint(0, 2))
            pais_destino = random.choice(PAISES_RIESGO) if j >= 2 else "ES"
            txs.append(Transaccion(
                id=f"TX-{len(txs)+700:06d}",
                id_cuenta_origen=eslabones[j],
                id_cuenta_destino=eslabones[j+1],
                importe=importe,
                divisa="EUR",
                fecha=fecha.isoformat(),
                canal="online",
                pais_origen="ES",
                pais_destino=pais_destino,
                tipo="transferencia",
                descripcion="Inversión internacional",
                patron_riesgo="layering"
            ))
    return txs

# ============================================================
# GENERAR Y GUARDAR
# ============================================================

print("ARGOS-FCC · Generando datos sintéticos del banco retail")
print("=" * 55)

clientes = generar_clientes(80)
cuentas = generar_cuentas(clientes)

txs_normales   = generar_transacciones_normales(cuentas, 600)
txs_structuring = generar_structuring(cuentas)
txs_smurfing   = generar_smurfing(cuentas)
txs_layering   = generar_layering(cuentas)

todas_txs = txs_normales + txs_structuring + txs_smurfing + txs_layering
random.shuffle(todas_txs)

# Guardar en JSON (capa Bronze)
def guardar(nombre, datos):
    ruta = f"./data/fcc/{nombre}.json"
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump([asdict(d) for d in datos], f,
                  ensure_ascii=False, indent=2)
    return ruta, len(datos)

r, n = guardar("clientes", clientes)
print(f"Clientes:      {n:>4} registros → {r}")

r, n = guardar("cuentas", cuentas)
print(f"Cuentas:       {n:>4} registros → {r}")

r, n = guardar("transacciones", todas_txs)
print(f"Transacciones: {n:>4} registros → {r}")

# Estadísticas
print("\nEstadísticas del dataset:")
print(f"  Clientes particulares: {sum(1 for c in clientes if c.tipo=='particular')}")
print(f"  Clientes empresa:      {sum(1 for c in clientes if c.tipo=='empresa')}")
print(f"  PEPs identificados:    {sum(1 for c in clientes if c.pep)}")
print(f"  Clientes riesgo alto:  {sum(1 for c in clientes if c.segmento_riesgo=='alto')}")
print(f"  Cuentas activas:       {sum(1 for c in cuentas if c.estado=='activa')}")
print(f"\nPatrones de riesgo incrustados:")
print(f"  Transacciones normales:    {len(txs_normales)}")
print(f"  Structuring (pitufeo):     {len(txs_structuring)}")
print(f"  Smurfing (muchos->uno):    {len(txs_smurfing)}")
print(f"  Layering (cadena opaca):   {len(txs_layering)}")
print(f"\nTotal transacciones: {len(todas_txs)}")
print("\nDatos Bronze listos para el pipeline Medallion y Neo4j.")
