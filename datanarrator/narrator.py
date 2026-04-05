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
            lines.append(f"  → {a['message']} {a['suggestion']}")
        return "\n".join(lines)

    def export(self, filepath: str) -> None:
        """Exporta el análisis completo a un archivo .txt o .md."""
        content = self.describe()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Análisis exportado a: {filepath}")

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
            return ""
        header = "--- Alertas y recomendaciones ---" if self.lang == "es" else "--- Alerts & recommendations ---"
        lines = [header]
        for a in alerts:
            lines.append(f"  ⚠  {a['message']}")
            lines.append(f"     → {a['suggestion']}")
        return "\n".join(lines)
