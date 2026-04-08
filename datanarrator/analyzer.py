"""
Módulo de análisis estadístico para datanarrator.

Contiene la clase DataAnalyzer, que es el motor interno de la
librería. Se encarga de extraer estadísticas de un DataFrame
de pandas para que el Narrator pueda generar texto descriptivo.

No se recomienda usar esta clase directamente. En su lugar
usa la clase Narrator que es la interfaz pública.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class DataAnalyzer:
    """Analiza un DataFrame y extrae estadísticas relevantes.

    Esta clase es el motor interno de datanarrator. Recibe un
    DataFrame de pandas y calcula todo lo necesario para que
    el Narrator pueda generar texto descriptivo.

    Parameters
    ----------
    df : pd.DataFrame
        El DataFrame que se quiere analizar. No puede estar vacío.

    Raises
    ------
    TypeError
        Si el input no es un DataFrame de pandas.
    ValueError
        Si el DataFrame está vacío.

    Examples
    --------
    >>> import pandas as pd
    >>> from datanarrator.analyzer import DataAnalyzer
    >>> df = pd.DataFrame({"edad": [25, 30, 35], "ciudad": ["cdmx", "mty", "gdl"]})
    >>> analyzer = DataAnalyzer(df)
    >>> results = analyzer.analyze()
    """

    def __init__(self, df: pd.DataFrame):
        """Inicializa el analizador con un DataFrame de pandas.

        Parameters
        ----------
        df : pd.DataFrame
            El DataFrame que se quiere analizar.

        Raises
        ------
        TypeError
            Si el input no es un DataFrame de pandas.
        ValueError
            Si el DataFrame está vacío.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("El input debe ser un DataFrame de pandas.")
        if df.empty:
            raise ValueError("El DataFrame no puede estar vacío.")
        self.df = df
        self._results = None

    def analyze(self) -> dict:
        """Corre el análisis completo del DataFrame.

        Ejecuta todos los sub-análisis internos y los agrupa
        en un solo diccionario. Este es el método principal
        que usa el Narrator para generar el texto.

        Returns
        -------
        dict
            Diccionario con las siguientes secciones:
            - overview: resumen general del dataset
            - numeric: estadísticas de columnas numéricas
            - categorical: estadísticas de columnas categóricas
            - datetime: estadísticas de columnas de fecha
            - correlations: pares de columnas correlacionadas
            - alerts: problemas detectados en el dataset
        """
        self._results = {
            "overview": self._overview(),
            "numeric": self._analyze_numeric(),
            "categorical": self._analyze_categorical(),
            "datetime": self._analyze_datetime(),
            "correlations": self._correlations(),
            "alerts": self._alerts(),
        }
        return self._results

    def _overview(self) -> dict:
        """Genera un resumen general del DataFrame.

        Cuenta filas, columnas, nulos, duplicados y memoria.
        También intenta detectar columnas de fecha que vengan
        como texto.

        Returns
        -------
        dict
            Diccionario con métricas generales del dataset.
        """
        df = self.df
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

        # Intentar detectar fechas en columnas string
        for col in categorical_cols.copy():
            try:
                pd.to_datetime(df[col])
                datetime_cols.append(col)
                categorical_cols.remove(col)
            except Exception:
                pass

        total_nulls = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]

        return {
            "rows": df.shape[0],
            "cols": df.shape[1],
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "total_nulls": int(total_nulls),
            "null_pct": round(total_nulls / total_cells * 100, 2),
            "duplicates": int(df.duplicated().sum()),
            "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
        }

    def _analyze_numeric(self) -> list:
        """Calcula estadísticas para cada columna numérica.

        Para cada columna calcula media, mediana, desviación
        estándar, rango, nulos, sesgo y outliers usando el
        método IQR (rango intercuartílico).

        Returns
        -------
        list
            Lista de diccionarios, uno por columna numérica.
        """
        results = []
        numeric_cols = self.df.select_dtypes(include="number").columns
        for col in numeric_cols:
            series = self.df[col].dropna()
            if len(series) == 0:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
            results.append({
                "col": col,
                "mean": round(series.mean(), 2),
                "median": round(series.median(), 2),
                "std": round(series.std(), 2),
                "min": round(series.min(), 2),
                "max": round(series.max(), 2),
                "nulls": int(self.df[col].isnull().sum()),
                "null_pct": round(self.df[col].isnull().sum() / len(self.df) * 100, 1),
                "skew": round(float(series.skew()), 2),
                "outlier_count": len(outliers),
            })
        return results

    def _analyze_categorical(self) -> list:
        """Calcula estadísticas para cada columna categórica.

        Para cada columna obtiene el número de valores únicos,
        el valor más frecuente y el porcentaje de nulos.
        También detecta columnas con alta cardinalidad.

        Returns
        -------
        list
            Lista de diccionarios, uno por columna categórica.
        """
        results = []
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            series = self.df[col].dropna()
            value_counts = series.value_counts()
            results.append({
                "col": col,
                "unique": int(series.nunique()),
                "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "top_freq": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "top_pct": round(value_counts.iloc[0] / len(series) * 100, 1) if len(value_counts) > 0 else 0,
                "nulls": int(self.df[col].isnull().sum()),
                "null_pct": round(self.df[col].isnull().sum() / len(self.df) * 100, 1),
                "high_cardinality": series.nunique() > 20,
            })
        return results

    def _analyze_datetime(self) -> list:
        """Calcula estadísticas para columnas de fecha.

        Para cada columna de tipo datetime obtiene la fecha
        mínima, máxima y el rango total en días.

        Returns
        -------
        list
            Lista de diccionarios, uno por columna de fecha.
        """
        results = []
        dt_cols = self.df.select_dtypes(include=["datetime"]).columns
        for col in dt_cols:
            series = pd.to_datetime(self.df[col], errors="coerce").dropna()
            if len(series) == 0:
                continue
            results.append({
                "col": col,
                "min_date": str(series.min().date()),
                "max_date": str(series.max().date()),
                "range_days": (series.max() - series.min()).days,
                "nulls": int(self.df[col].isnull().sum()),
            })
        return results

    def _correlations(self) -> list:
        """Detecta pares de columnas con correlación significativa.

        Solo reporta correlaciones con valor absoluto mayor o
        igual a 0.5. Las clasifica como moderadas (0.5-0.75)
        o altas (mayor a 0.75).

        Returns
        -------
        list
            Lista de pares de columnas correlacionadas,
            ordenada de mayor a menor correlación absoluta.
        """
        numeric = self.df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return []
        corr_matrix = numeric.corr()
        pairs = []
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if not np.isnan(val) and abs(val) >= 0.5:
                    pairs.append({
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "correlation": round(val, 2),
                        "strength": "alta" if abs(val) >= 0.75 else "moderada",
                        "direction": "positiva" if val > 0 else "negativa",
                    })
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return pairs

    def _alerts(self) -> list:
        """Detecta problemas comunes en el dataset.

        Revisa cuatro tipos de problemas:
        - Registros duplicados
        - Columnas con más del 20% de nulos
        - Columnas categóricas con alta cardinalidad (más de 50 valores)
        - Columnas con un solo valor único (sin varianza)

        Returns
        -------
        list
            Lista de diccionarios con el tipo de alerta,
            mensaje y sugerencia de solución.
        """
        alerts = []
        df = self.df

        # Duplicados
        dups = df.duplicated().sum()
        if dups > 0:
            alerts.append({
                "type": "duplicates",
                "message": f"{dups} registros duplicados detectados.",
                "suggestion": "Considera eliminarlos antes de modelar.",
            })

        # Columnas con muchos nulos
        for col in df.columns:
            null_pct = df[col].isnull().sum() / len(df) * 100
            if null_pct >= 20:
                alerts.append({
                    "type": "high_nulls",
                    "col": col,
                    "message": f"'{col}' tiene {null_pct:.1f}% de valores nulos.",
                    "suggestion": "Considera imputar o eliminar esta columna.",
                })

        # Alta cardinalidad
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if df[col].nunique() > 50:
                alerts.append({
                    "type": "high_cardinality",
                    "col": col,
                    "message": f"'{col}' tiene {df[col].nunique()} valores únicos.",
                    "suggestion": "Evita label encoding directo. Considera target encoding.",
                })

        # Columnas con un solo valor
        for col in df.columns:
            if df[col].nunique() == 1:
                alerts.append({
                    "type": "constant_column",
                    "col": col,
                    "message": f"'{col}' tiene un solo valor único.",
                    "suggestion": "Esta columna no aporta información. Considera eliminarla.",
                })

        return alerts
