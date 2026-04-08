"""
datanarrator — Convierte DataFrames en análisis de lenguaje natural.

Librería de Python para generar descripciones automáticas de datasets
de pandas. En lugar de leer tablas de números, obtienes texto
interpretado con hallazgos, alertas y recomendaciones.

Uso básico
----------
>>> from datanarrator import Narrator
>>> import pandas as pd
>>> df = pd.read_csv("datos.csv")
>>> n = Narrator(df, lang="es")
>>> print(n.describe())
"""

from datanarrator.narrator import Narrator

__version__ = "0.1.2"
__all__ = ["Narrator"]
