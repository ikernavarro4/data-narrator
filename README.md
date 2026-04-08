# datanarrator 📊

[![PyPI version](https://badge.fury.io/py/datanarrator.svg)](https://pypi.org/project/datanarrator/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ikernavarro4/data-narrator/blob/main/notebooks/tutorial.ipynb)

**datanarrator** convierte cualquier DataFrame de pandas en un análisis en lenguaje natural. En lugar de leer tablas de números, obtienes texto interpretado con hallazgos, alertas y recomendaciones automáticas.

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
| `export(filepath)` | Exporta el análisis a .txt o .md |
| `compare(df2)` | Compara dos datasets y detecta data drift |
| `suggest()` | Sugiere modelos ML y preprocesamiento |

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
--- Columnas numéricas ---
Age: media=29.7, mediana=28.0, std=14.53. Se detectaron 11 posibles outliers.
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
Preprocesamiento recomendado:
→ Imputar nulos en: Age
→ Encodear variables categoricas: Name, Sex, Ticket, Cabin, Embarked
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
