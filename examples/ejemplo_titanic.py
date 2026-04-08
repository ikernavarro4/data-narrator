"""
Ejemplo de uso de datanarrator con el dataset del Titanic.

Este script demuestra todas las funcionalidades de la librería
usando uno de los datasets más conocidos en ciencia de datos.

Para correr este ejemplo:
    python examples/ejemplo_titanic.py

O con Docker:
    docker compose up --build
"""

import pandas as pd
from datanarrator import Narrator

print("=" * 50)
print("datanarrator — Ejemplo con Titanic")
print("=" * 50)

# Cargamos el dataset del Titanic desde un repositorio público
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print(f"\nDataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# Inicializamos el Narrator en español
n = Narrator(df, lang="es")

# 1. Análisis completo — muestra todas las secciones
print("\n📊 ANÁLISIS COMPLETO:")
print(n.describe())

# 2. Resumen ejecutivo — útil para reportes rápidos
print("\n📋 RESUMEN EJECUTIVO:")
print(n.executive_summary())

# 3. Solo alertas — problemas que necesitan atención antes de modelar
print("\n⚠️  SOLO ALERTAS:")
print(n.alerts_only())

# 4. Sugerencias de modelado — qué modelo usar y cómo preprocesar
print("\n💡 SUGERENCIAS DE MODELADO:")
print(n.suggest())

# 5. Comparación de datasets — simulamos un dataset de producción
# con menos registros y edades modificadas para ver el data drift
print("\n🔄 COMPARACIÓN DE DATASETS:")
df_produccion = df.sample(frac=0.7, random_state=42).copy()
df_produccion["Age"] = df_produccion["Age"] * 1.5
print(n.compare(df_produccion))

# 6. Exportar el análisis a un archivo de texto
n.export("reporte_titanic.md")

print("\n✅ Demo completada.")
