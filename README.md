# datanarrator 📊

[![PyPI version](https://img.shields.io/pypi/v/datanarrator)](https://pypi.org/project/datanarrator/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/ikernavarro4/data-narrator/actions/workflows/tests.yml/badge.svg)](https://github.com/ikernavarro4/data-narrator/actions/workflows/tests.yml)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ikernavarro4/data-narrator/blob/main/notebooks/tutorial.ipynb)

**datanarrator** convierte cualquier DataFrame de pandas en un análisis en lenguaje natural. En lugar de leer tablas de números, obtienes texto interpretado con hallazgos, alertas y recomendaciones automáticas y reportes HTML interactivos.

---

## Instalación
```bash
pip install datanarrator
```

---


## Requisitos
- Python 3.9+
- Las dependencias (pandas, numpy, matplotlib) se instalan automáticamente con pip.

---
## Uso rápido
```python
import pandas as pd
from datanarrator import Narrator

df = pd.read_csv("datos.csv")
n = Narrator(df, lang="es")

print(n.describe())
```

---

## Métodos disponibles

| Método | Descripción |
|--------|-------------|
| `describe()` | Análisis completo en lenguaje natural |
| `executive_summary()` | Resumen ejecutivo de 2-3 oraciones |
| `alerts_only()` | Solo alertas y recomendaciones |
| `quality_score()` | Score de calidad del dataset de 0 a 100 con grado A–F |
| `narrative()` | Análisis narrativo interpretativo en párrafos |
| `narrate(audience=)` | Narrativa adaptada por audiencia: ejecutivo, técnico o no técnico |
| `suggest()` | Sugiere modelos ML y pasos de preprocesamiento |
| `compare(df2)` | Compara dos datasets y detecta data drift |
| `export(filepath)` | Exporta a `.txt`, `.md` o reporte `.html` interactivo |

---

## Ejemplos de uso

### Análisis completo
```python
n = Narrator(df, lang="es")
print(n.describe())
```

```
--- Resumen general ---
El dataset contiene 891 registros con 12 columnas: 7 numéricas, 5 categóricas.
Valores nulos: 866 (8.1% del total).
Memoria utilizada: 285.6 KB.
--- Columnas numéricas ---
Age: media=29.7, mediana=28.0, std=14.53, rango=[0.42 – 80.0], nulos=177 (19.9%). Se detectaron 11 posibles outliers (IQR).
Fare: media=32.2, mediana=14.45, std=49.69, rango=[0.0 – 512.33]. Se detectaron 116 posibles outliers (IQR). Distribución con sesgo positivo (4.79).
--- Correlaciones relevantes ---
Pclass ↔ Fare: correlación moderada negativa (-0.55)
--- Alertas y recomendaciones ---
⚠  'Cabin' tiene 77.1% de valores nulos.
→ Considera imputar o eliminar esta columna.
```

### Sugerencias de modelado
```python
print(n.suggest())
```

```
--- Sugerencias para modelado ---
Tipo de problema detectado: clasificacion binaria
Variable objetivo probable: Survived
Modelos recomendados:
  → Logistic Regression — buen baseline para clasificacion
  → Random Forest — robusto con variables mixtas
  → XGBoost — recomendado si priorizas accuracy
Columnas a excluir o tratar: Name, Ticket, Cabin
  → Alta cardinalidad, usa target encoding o eliminalas
Preprocesamiento recomendado:
  → Imputar nulos en: Age
  → Aplicar log transform en: SibSp, Parch, Fare (sesgo alto)
  → Revisar outliers en: Age, SibSp, Parch, Fare
  → Encodear variables categóricas
```

### Comparar dos datasets
```python
n = Narrator(df_train, lang="es")
print(n.compare(df_produccion))
```

```
--- Comparación de datasets ---
El segundo dataset tiene 267 registros menos.
'Age': media subió de 29.7 a 44.34 (+49.3%).
⚠  Posible data drift en: Age.
→ Dispersión cambió más del 30%. Revisa antes de producción.
```

### Score de calidad

```python
resultado = n.quality_score()
print(resultado["resumen"])
print(resultado["penalizaciones"])
```

```
El dataset obtuvo un score de 68/100 (grado D).
{'nulos': 12.15, 'duplicados': 0.0, 'constantes': 0, 'cardinalidad': 15, 'cols_nulas': 5}
```

### Análisis narrativo

```python
print(n.narrative())
```

```
Con 891 registros y 12 columnas, el dataset es de tamaño moderado — adecuado
para la mayoría de algoritmos de machine learning. Contiene 7 variables numéricas
y 5 categóricas. El problema de calidad más crítico es la columna "Cabin", que
concentra el 77.1% de valores nulos — un nivel tan alto sugiere que esta
información simplemente no estaba disponible para la mayoría de los registros
y probablemente deba eliminarse. Score de calidad global: 68/100 (grado D).
```

### Narrativa por audiencia — `narrate(audience=)`

El mismo DataFrame, tres narrativas completamente distintas según quién las lee.
Ninguna otra librería de análisis de datos hace esto.

```python
n = Narrator(df, lang="es")

for audience in ["executive", "technical", "non-technical"]:
    print(f"\n=== {audience.upper()} ===")
    print(n.narrate(audience=audience))
```

```
=== EXECUTIVE ===
[ Resumen Ejecutivo ]
El dataset cuenta con 891 registros y se encuentra con problemas de calidad que requieren atención (score de calidad: 68/100). El problema más urgente es la columna 'Cabin', que tiene 77.1% de datos faltantes y debe tratarse antes de cualquier análisis. La variable 'Survived' es el probable indicador de resultado: el 38.0% de los registros tienen resultado positivo. Se recomienda una limpieza básica antes de usar estos datos para tomar decisiones.

=== TECHNICAL ===
[ Análisis Técnico — Para Científicos de Datos ]
Con 891 registros y 12 columnas, el dataset es de tamaño moderado — adecuado para la mayoría de algoritmos de machine learning. Contiene 7 variables numéricas y 5 categóricas. El problema de calidad más crítico es la columna "Cabin", que concentra el 77.1% de valores nulos — un nivel tan alto sugiere que esta información simplemente no estaba disponible para la mayoría de los registros y probablemente deba eliminarse. En total, el 8.1% de todas las celdas del dataset están vacías. Score de calidad global: 68/100 (grado D).

La columna PassengerId parece ser un identificador único sin valor predictivo — su distribución perfectamente uniforme confirma que es solo un índice y debe excluirse del modelado.

La variable Survived es binaria y representa una candidata natural como variable objetivo para clasificación. El 38.0% de los registros tienen valor positivo frente al 62.0% negativo — una distribución relativamente balanceada. El balance entre clases es favorable para el entrenamiento.

La variable Pclass tiene una ligera asimetría (media=2.31, mediana=3.0).

La variable Age tiene una ligera asimetría (media=29.7, mediana=28.0). Presenta 11 valores atípicos (1.2%) que conviene revisar antes de modelar. El 19.9% de nulos es manejable mediante imputación con la mediana.

La variable SibSp muestra un sesgo positivo pronunciado (media=0.52, mediana=0.0): la mayoría de los valores se concentran en valores bajos pero hay casos extremos hacia arriba que inflan el promedio. Presenta 46 valores atípicos (5.2%) que conviene revisar antes de modelar.

La variable Parch muestra un sesgo positivo pronunciado (media=0.38, mediana=0.0): la mayoría de los valores se concentran en valores bajos pero hay casos extremos hacia arriba que inflan el promedio. El 23.9% de outliers es alto — podría indicar subpoblaciones distintas o errores de captura.

La variable Fare muestra un sesgo positivo pronunciado (media=32.2, mediana=14.45): la mayoría de los valores se concentran en valores bajos pero hay casos extremos hacia arriba que inflan el promedio. El 13.0% de outliers es alto — podría indicar subpoblaciones distintas o errores de captura.

El análisis de correlaciones revela 1 relación(es) estadísticamente relevante(s):
  Pclass ↔ Fare: correlación moderada negativa (-0.55), es decir, a mayor Pclass, menor Fare. Una señal útil para el modelo pero sin riesgo significativo de multicolinealidad.

Se identificaron 4 problema(s) de calidad. En orden de impacto potencial en el modelado:
  • 'Cabin' tiene 77.1% de valores nulos. Considera imputar o eliminar esta columna.
  • 'Name' tiene 891 valores únicos. Evita label encoding directo. Considera target encoding.
  • 'Ticket' tiene 681 valores únicos. Evita label encoding directo. Considera target encoding.
  • 'Cabin' tiene 147 valores únicos. Evita label encoding directo. Considera target encoding.

Resumen de pasos recomendados (6 en total):
  1. Eliminar la columna "Cabin" (77.1% de nulos) — con más del 50% de datos faltantes, la imputación introduciría más ruido que información.
  2. Imputar nulos en "Age" (19.9%) con la media — su distribución simétrica hace que sea la opción más robusta.
  3. Aplicar transformación logarítmica en "SibSp", "Parch", "Fare" para reducir el sesgo y el impacto de los outliers antes de escalar o modelar.
  4. Para "Name", "Ticket", "Cabin" (alta cardinalidad), usar target encoding o embeddings — el one-hot encoding generaría cientos de columnas dispersas que degradan el rendimiento del modelo.
  5. Aplicar one-hot encoding a "Sex", "Embarked" — su baja cardinalidad hace que sea seguro y eficiente.
  6. Realizar la división train/test antes de aplicar cualquier transformación o imputación para evitar data leakage.

=== NON-TECHNICAL ===
[ Explicación Simple — Para Todos ]
Este conjunto de datos tiene información sobre 891 elementos, organizados en 12 categorías diferentes.

Hay información que falta, especialmente en 'Cabin'. Imagina un formulario donde muchas personas dejaron ese campo en blanco — eso es lo que ocurre aquí.

En columnas como 'SibSp', 'Parch', la mayoría de los valores son similares entre sí, pero hay algunos casos muy distintos — como en un salón donde casi todos sacan entre 7 y 9, pero alguien saca 10 y otro 2.

En 4 columnas hay valores inusualmente altos o bajos — no necesariamente errores, pero vale la pena revisarlos.

Hay una relación interesante: cuando 'Pclass' sube, 'Fare' tiende a bajar. Esto puede ser una pista clave para entender el comportamiento de los datos.

En general, estos datos están en condiciones aceptables, aunque tienen aspectos que podrían mejorarse.
```

También puedes llamar a una sola audiencia directamente:

```python
print(n.narrate(audience="executive"))
```

El reporte HTML exportado con `export("reporte.html")` incluye una sección
**"👥 Audiencias"** con los 3 textos en tabs interactivos.

### Exportar reporte HTML interactivo

```python
n.export("reporte_titanic.html")  # reporte visual con gráficas
n.export("reporte_titanic.txt")   # texto plano
```

El reporte HTML incluye navegación por secciones, semáforo de salud por columna,
histogramas, gráficas interactivas con Chart.js, score de calidad, sugerencias de ML
y la nueva sección **👥 Audiencias** con los 3 tipos de narrativa en tabs interactivos.

### Soporte multilenguaje
```python
n_es = Narrator(df, lang="es")  # Español
n_en = Narrator(df, lang="en")  # English
```

---

## Correr con Docker
```bash
# Clonar el repositorio
git clone https://github.com/ikernavarro4/data-narrator.git
cd data-narrator

# Construir y correr el contenedor
docker compose up --build
```

El contenedor instala la librería desde PyPI y corre automáticamente el script de ejemplo con el dataset del Titanic.

---

## Correr los tests
```bash
pip install pytest
pytest tests/ -v
```

---

## Repositorio y links

- **GitHub:** https://github.com/ikernavarro4/data-narrator
- **PyPI:** https://pypi.org/project/datanarrator/
- **Tutorial en Colab:** https://colab.research.google.com/github/ikernavarro4/data-narrator/blob/main/notebooks/tutorial.ipynb
