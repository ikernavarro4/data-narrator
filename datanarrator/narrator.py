from __future__ import annotations
import pandas as pd
from datanarrator.analyzer import DataAnalyzer


class Narrator:
    """
    Convierte un DataFrame de pandas en un análisis en lenguaje natural.

    Ejemplo:
        >>> from datanarrator import Narrator
        >>> import pandas as pd
        >>> df = pd.read_csv("titanic.csv")
        >>> n = Narrator(df, lang="es")
        >>> print(n.describe())
    """

    SUPPORTED_LANGS = ("es", "en")

    def __init__(self, df: pd.DataFrame, lang: str = "es"):
        if lang not in self.SUPPORTED_LANGS:
            raise ValueError(f"Idioma '{lang}' no soportado. Usa: {self.SUPPORTED_LANGS}")
        self.df = df
        self.lang = lang
        self._analyzer = DataAnalyzer(df)
        self._data = self._analyzer.analyze()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Genera el análisis completo en lenguaje natural."""
        sections = [
            self._section_overview(),
            self._section_numeric(),
            self._section_categorical(),
            self._section_correlations(),
            self._section_alerts(),
        ]
        return "\n\n".join(s for s in sections if s)

    def executive_summary(self) -> str:
        """Resumen ejecutivo de 2-3 oraciones."""
        ov = self._data["overview"]
        parts = []

        if self.lang == "es":
            parts.append(
                f"El dataset contiene {ov['rows']:,} registros y {ov['cols']} columnas "
                f"({len(ov['numeric_cols'])} numéricas, {len(ov['categorical_cols'])} categóricas)."
            )
            if ov["null_pct"] > 0:
                parts.append(f"Presenta un {ov['null_pct']}% de valores nulos en total.")
            corrs = self._data["correlations"]
            if corrs:
                top = corrs[0]
                parts.append(
                    f"La correlación más fuerte es entre '{top['col_a']}' y '{top['col_b']}' ({top['correlation']})."
                )
        else:
            parts.append(
                f"The dataset has {ov['rows']:,} rows and {ov['cols']} columns "
                f"({len(ov['numeric_cols'])} numeric, {len(ov['categorical_cols'])} categorical)."
            )
            if ov["null_pct"] > 0:
                parts.append(f"Overall null rate is {ov['null_pct']}%.")
            corrs = self._data["correlations"]
            if corrs:
                top = corrs[0]
                parts.append(
                    f"Strongest correlation: '{top['col_a']}' and '{top['col_b']}' ({top['correlation']})."
                )

        return " ".join(parts)

    def alerts_only(self) -> str:
        """Devuelve solo las alertas y recomendaciones detectadas."""
        alerts = self._data["alerts"]
        if not alerts:
            if self.lang == "es":
                return "No se detectaron alertas en este dataset."
            return "No alerts detected in this dataset."

        lines = []
        header = "Alertas detectadas:" if self.lang == "es" else "Alerts detected:"
        lines.append(header)
        for a in alerts:
            if self.lang == "en":
                msg = a["message"].replace("registros duplicados detectados", "duplicate rows detected")
                msg = msg.replace("tiene", "has")
                msg = msg.replace("valores nulos", "null values")
                msg = msg.replace("valores únicos", "unique values")
                sug = a["suggestion"].replace("Considera imputar o eliminar esta columna", "Consider imputing or dropping this column")
                sug = sug.replace("Evita label encoding directo. Considera target encoding", "Avoid direct label encoding. Consider target encoding")
                sug = sug.replace("Considera eliminarlos antes de modelar", "Consider dropping them before modeling")
                sug = sug.replace("Esta columna no aporta información. Considera eliminarla", "This column has no information. Consider dropping it")
            else:
                msg = a["message"]
                sug = a["suggestion"]
            lines.append(f"  → {msg} {sug}")
        return "\n".join(lines)

    def export(self, filepath: str) -> None:
        """Exporta el análisis completo a un archivo .txt o .md."""
        content = self.describe()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Análisis exportado a: {filepath}")

    def compare(self, df2):
        if not isinstance(df2, __import__('pandas').DataFrame):
            raise TypeError("El input debe ser un DataFrame de pandas.")
        if df2.empty:
            raise ValueError("El segundo DataFrame no puede estar vacío.")
        df1 = self.df
        out = []
        if self.lang == "es":
            out.append("--- Comparación de datasets ---")
            diff = df2.shape[0] - df1.shape[0]
            if diff > 0:
                out.append(f"El segundo dataset tiene {diff} registros más.")
            elif diff < 0:
                out.append(f"El segundo dataset tiene {abs(diff)} registros menos.")
            else:
                out.append("Ambos datasets tienen el mismo número de registros.")
            cols1, cols2 = set(df1.columns), set(df2.columns)
            if cols1 - cols2:
                out.append(f"Columnas solo en el primero: {', '.join(cols1 - cols2)}.")
            if cols2 - cols1:
                out.append(f"Columnas solo en el segundo: {', '.join(cols2 - cols1)}.")
            common = [c for c in df1.select_dtypes(include="number").columns if c in df2.select_dtypes(include="number").columns]
            for col in common:
                m1, m2 = df1[col].mean(), df2[col].mean()
                if m1 == 0:
                    continue
                pct = round((m2 - m1) / abs(m1) * 100, 1)
                if abs(pct) >= 10:
                    d = "subió" if pct > 0 else "bajó"
                    out.append(f"'{col}': media {d} de {round(m1,2)} a {round(m2,2)} ({pct:+.1f}%).")
            n1 = df1.isnull().sum().sum() / (df1.shape[0] * df1.shape[1]) * 100
            n2 = df2.isnull().sum().sum() / (df2.shape[0] * df2.shape[1]) * 100
            dn = round(n2 - n1, 1)
            if abs(dn) >= 5:
                d = "más" if dn > 0 else "menos"
                out.append(f"⚠  El segundo dataset tiene {abs(dn)}% {d} valores nulos.")
                out.append("     → Posible degradación en calidad de datos.")
            drift = []
            for col in common:
                s1, s2 = df1[col].std(), df2[col].std()
                if s1 == 0:
                    continue
                if abs(s2 - s1) / s1 * 100 >= 30:
                    drift.append(col)
            if drift:
                out.append(f"⚠  Posible data drift en: {', '.join(drift)}.")
                out.append("     → Dispersión cambió más del 30%. Revisa antes de producción.")
        else:
            out.append("--- Dataset comparison ---")
            diff = df2.shape[0] - df1.shape[0]
            if diff > 0:
                out.append(f"The second dataset has {diff} more rows.")
            elif diff < 0:
                out.append(f"The second dataset has {abs(diff)} fewer rows.")
            else:
                out.append("Both datasets have the same number of rows.")
            cols1, cols2 = set(df1.columns), set(df2.columns)
            if cols1 - cols2:
                out.append(f"Columns only in first: {', '.join(cols1 - cols2)}.")
            if cols2 - cols1:
                out.append(f"Columns only in second: {', '.join(cols2 - cols1)}.")
            common = [c for c in df1.select_dtypes(include="number").columns if c in df2.select_dtypes(include="number").columns]
            for col in common:
                m1, m2 = df1[col].mean(), df2[col].mean()
                if m1 == 0:
                    continue
                pct = round((m2 - m1) / abs(m1) * 100, 1)
                if abs(pct) >= 10:
                    d = "increased" if pct > 0 else "decreased"
                    out.append(f"'{col}': mean {d} from {round(m1,2)} to {round(m2,2)} ({pct:+.1f}%).")
            n1 = df1.isnull().sum().sum() / (df1.shape[0] * df1.shape[1]) * 100
            n2 = df2.isnull().sum().sum() / (df2.shape[0] * df2.shape[1]) * 100
            dn = round(n2 - n1, 1)
            if abs(dn) >= 5:
                d = "more" if dn > 0 else "fewer"
                out.append(f"⚠  Second dataset has {abs(dn)}% {d} null values.")
                out.append("     → Possible data quality degradation.")
            drift = []
            for col in common:
                s1, s2 = df1[col].std(), df2[col].std()
                if s1 == 0:
                    continue
                if abs(s2 - s1) / s1 * 100 >= 30:
                    drift.append(col)
            if drift:
                out.append(f"⚠  Possible data drift in: {', '.join(drift)}.")
                out.append("     → Dispersion changed over 30%. Review before production.")
        return "\n".join(out)

    # ------------------------------------------------------------------
    # Secciones internas
    # ------------------------------------------------------------------

    def _section_overview(self) -> str:
        ov = self._data["overview"]
        if self.lang == "es":
            lines = [
                "--- Resumen general ---",
                f"El dataset contiene {ov['rows']:,} registros con {ov['cols']} columnas: "
                f"{len(ov['numeric_cols'])} numéricas, {len(ov['categorical_cols'])} categóricas"
                + (f" y {len(ov['datetime_cols'])} de fecha." if ov["datetime_cols"] else "."),
            ]
            if ov["total_nulls"] > 0:
                lines.append(f"Valores nulos: {ov['total_nulls']:,} ({ov['null_pct']}% del total).")
            else:
                lines.append("No se encontraron valores nulos.")
            if ov["duplicates"] > 0:
                lines.append(f"Se detectaron {ov['duplicates']} registros duplicados.")
            lines.append(f"Memoria utilizada: {ov['memory_kb']} KB.")
        else:
            lines = [
                "--- Overview ---",
                f"The dataset has {ov['rows']:,} rows and {ov['cols']} columns: "
                f"{len(ov['numeric_cols'])} numeric, {len(ov['categorical_cols'])} categorical"
                + (f", and {len(ov['datetime_cols'])} datetime." if ov["datetime_cols"] else "."),
            ]
            if ov["total_nulls"] > 0:
                lines.append(f"Null values: {ov['total_nulls']:,} ({ov['null_pct']}% of all cells).")
            else:
                lines.append("No null values found.")
            if ov["duplicates"] > 0:
                lines.append(f"{ov['duplicates']} duplicate rows detected.")
            lines.append(f"Memory usage: {ov['memory_kb']} KB.")
        return "\n".join(lines)

    def _section_numeric(self) -> str:
        cols = self._data["numeric"]
        if not cols:
            return ""
        header = "--- Columnas numéricas ---" if self.lang == "es" else "--- Numeric columns ---"
        lines = [header]
        for c in cols:
            if self.lang == "es":
                line = (
                    f"{c['col']}: media={c['mean']}, mediana={c['median']}, "
                    f"std={c['std']}, rango=[{c['min']} – {c['max']}]"
                )
                if c["nulls"] > 0:
                    line += f", nulos={c['nulls']} ({c['null_pct']}%)"
                if c["outlier_count"] > 0:
                    line += f". Se detectaron {c['outlier_count']} posibles outliers (IQR)."
                if abs(c["skew"]) > 1:
                    direction = "positivo" if c["skew"] > 0 else "negativo"
                    line += f" Distribución con sesgo {direction} ({c['skew']})."
            else:
                line = (
                    f"{c['col']}: mean={c['mean']}, median={c['median']}, "
                    f"std={c['std']}, range=[{c['min']} – {c['max']}]"
                )
                if c["nulls"] > 0:
                    line += f", nulls={c['nulls']} ({c['null_pct']}%)"
                if c["outlier_count"] > 0:
                    line += f". {c['outlier_count']} potential outliers detected (IQR)."
                if abs(c["skew"]) > 1:
                    direction = "positive" if c["skew"] > 0 else "negative"
                    line += f" {direction.capitalize()} skew ({c['skew']})."
            lines.append(f"  {line}")
        return "\n".join(lines)

    def _section_categorical(self) -> str:
        cols = self._data["categorical"]
        if not cols:
            return ""
        header = "--- Columnas categóricas ---" if self.lang == "es" else "--- Categorical columns ---"
        lines = [header]
        for c in cols:
            if self.lang == "es":
                line = (
                    f"{c['col']}: {c['unique']} valores únicos. "
                    f"El más frecuente es '{c['top_value']}' ({c['top_pct']}% de los registros)."
                )
                if c["nulls"] > 0:
                    line += f" Nulos: {c['null_pct']}%."
            else:
                line = (
                    f"{c['col']}: {c['unique']} unique values. "
                    f"Most frequent: '{c['top_value']}' ({c['top_pct']}% of records)."
                )
                if c["nulls"] > 0:
                    line += f" Nulls: {c['null_pct']}%."
            lines.append(f"  {line}")
        return "\n".join(lines)

    def _section_correlations(self) -> str:
        corrs = self._data["correlations"]
        if not corrs:
            return ""
        header = "--- Correlaciones relevantes ---" if self.lang == "es" else "--- Relevant correlations ---"
        lines = [header]
        for c in corrs:
            if self.lang == "es":
                lines.append(
                    f"  {c['col_a']} ↔ {c['col_b']}: correlación {c['strength']} {c['direction']} ({c['correlation']})"
                )
            else:
                lines.append(
                    f"  {c['col_a']} ↔ {c['col_b']}: {c['strength']} {c['direction']} correlation ({c['correlation']})"
                )
        return "\n".join(lines)

    def _section_alerts(self) -> str:
        alerts = self._data["alerts"]
        if not alerts:
            if self.lang == "es":
                return "No se detectaron alertas en este dataset."
            return "No alerts detected in this dataset."
        header = "--- Alertas y recomendaciones ---" if self.lang == "es" else "--- Alerts & recommendations ---"
        lines = [header]
        for a in alerts:
            if self.lang == "en":
                msg = a["message"].replace("registros duplicados detectados", "duplicate rows detected")
                msg = msg.replace("tiene", "has")
                msg = msg.replace("valores nulos", "null values")
                msg = msg.replace("valores únicos", "unique values")
                sug = a["suggestion"].replace("Considera imputar o eliminar esta columna", "Consider imputing or dropping this column")
                sug = sug.replace("Evita label encoding directo. Considera target encoding", "Avoid direct label encoding. Consider target encoding")
                sug = sug.replace("Considera eliminarlos antes de modelar", "Consider dropping them before modeling")
                sug = sug.replace("Esta columna no aporta información. Considera eliminarla", "This column has no information. Consider dropping it")
            else:
                msg = a["message"]
                sug = a["suggestion"]
            lines.append(f"  ⚠  {msg}")
            lines.append(f"     → {sug}")
        return "\n".join(lines)
