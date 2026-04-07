import pandas as pd
from datanarrator import Narrator

print("=" * 50)
print("datanarrator — Ejemplo con Titanic")
print("=" * 50)

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

n = Narrator(df, lang="es")

print("\n📊 ANÁLISIS COMPLETO:")
print(n.describe())

print("\n📋 RESUMEN EJECUTIVO:")
print(n.executive_summary())

print("\n⚠️  SOLO ALERTAS:")
print(n.alerts_only())

print("\n🔄 COMPARACIÓN DE DATASETS:")
df2 = df.sample(frac=0.7, random_state=42).copy()
df2["Age"] = df2["Age"] * 1.5
print(n.compare(df2))

print("\n✅ Demo completada.")
