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

### Sugerencias de modelado
```python
print(n.suggest())
```

### Comparar dos datasets
```python
n = Narrator(df_train, lang="es")
print(n.compare(df_produccion))
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
