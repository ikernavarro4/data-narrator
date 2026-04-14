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
  → Encodear var
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

### Exportar reporte HTML interactivo

```python
n.export("reporte_titanic.html")  # reporte visual con gráficas
n.export("reporte_titanic.txt")   # texto plano
```

El reporte HTML incluye navegación por secciones, semáforo de salud por columna,
histogramas, gráficas interactivas con Chart.js, score de calidad y sugerencias de ML.

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
