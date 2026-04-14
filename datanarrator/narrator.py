"""
Módulo principal de datanarrator.

Contiene la clase Narrator, que es la interfaz pública de la
librería. Recibe un DataFrame de pandas y genera texto descriptivo
usando el DataAnalyzer como motor interno.

Uso básico
----------
>>> from datanarrator import Narrator
>>> import pandas as pd
>>> df = pd.read_csv("datos.csv")
>>> n = Narrator(df, lang="es")
>>> print(n.describe())
"""

from __future__ import annotations
import pandas as pd
from datanarrator.analyzer import DataAnalyzer


class Narrator:
    """Convierte un DataFrame de pandas en un análisis en lenguaje natural.

    En lugar de leer tablas de números, obtienes texto interpretado
    con hallazgos, alertas y recomendaciones automáticas. Útil para
    exploración rápida de datos y generación de reportes.

    Parameters
    ----------
    df : pd.DataFrame
        El DataFrame que se quiere analizar.
    lang : str, optional
        Idioma del análisis. Puede ser 'es' (español) o 'en' (inglés).
        Por defecto es 'es'.

    Raises
    ------
    TypeError
        Si el input no es un DataFrame de pandas.
    ValueError
        Si el idioma no está soportado o el DataFrame está vacío.

    Examples
    --------
    >>> from datanarrator import Narrator
    >>> import pandas as pd
    >>> df = pd.read_csv("titanic.csv")
    >>> n = Narrator(df, lang="es")
    >>> print(n.describe())
    >>> print(n.executive_summary())
    """

    SUPPORTED_LANGS = ("es", "en")

    def __init__(self, df: pd.DataFrame, lang: str = "es"):
        """Inicializa el Narrator con un DataFrame y un idioma.

        Parameters
        ----------
        df : pd.DataFrame
            El DataFrame que se quiere analizar.
        lang : str, optional
            Idioma del análisis. 'es' para español, 'en' para inglés.
            Por defecto es 'es'.

        Raises
        ------
        ValueError
            Si el idioma no está en la lista de soportados.
        """
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
        """Genera el análisis completo en lenguaje natural.

        Combina todas las secciones del análisis en un solo texto:
        resumen general, columnas numéricas, columnas categóricas,
        correlaciones y alertas.

        Returns
        -------
        str
            Texto con el análisis completo del DataFrame.

        Examples
        --------
        >>> n = Narrator(df, lang="es")
        >>> print(n.describe())
        """
        sections = [
            self._section_overview(),
            self._section_numeric(),
            self._section_categorical(),
            self._section_correlations(),
            self._section_alerts(),
        ]
        return "\n\n".join(s for s in sections if s)

    def executive_summary(self) -> str:
        """Genera un resumen ejecutivo del dataset en 2-3 oraciones.

        Útil cuando necesitas una descripción rápida sin entrar
        en detalles. Incluye el tamaño del dataset, porcentaje
        de nulos y la correlación más fuerte detectada.

        Returns
        -------
        str
            Resumen ejecutivo en una o dos oraciones.

        Examples
        --------
        >>> n = Narrator(df, lang="es")
        >>> print(n.executive_summary())
        """
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
        """Devuelve solo las alertas y recomendaciones del dataset.

        Útil cuando solo te interesa saber qué problemas tiene
        el dataset antes de modelar. Detecta nulos, duplicados,
        alta cardinalidad y columnas sin varianza.

        Returns
        -------
        str
            Texto con las alertas detectadas y sugerencias.
            Si no hay alertas, lo indica explícitamente.

        Examples
        --------
        >>> n = Narrator(df, lang="es")
        >>> print(n.alerts_only())
        """
        alerts = self._data["alerts"]
        if not alerts:
            if self.lang == "es":
                return "No se detectaron alertas en este dataset."
            return "No alerts detected in this dataset."

        lines = []
        header = "Alertas detectadas:" if self.lang == "es" else "Alerts detected:"
        lines.append(header)
	# Usamos _translate_alert() para centralizar la traducción
        # en lugar de duplicar la lógica de str.replace() aquí
        for a in alerts:
            msg, sug = self._translate_alert(a)
            lines.append(f"  → {msg} {sug}")
        return "\n".join(lines)

    def export(self, filepath: str) -> None:
        """Exporta el análisis a un archivo de texto o reporte HTML.

        Si la extensión del archivo es .html, genera un reporte visual
        con tablas y gráficas embebidas en base64. Para cualquier otra
        extensión (.txt, .md) exporta el texto plano de describe().

        Parameters
        ----------
        filepath : str
            Ruta del archivo donde se guardará el análisis.
            Usa extensión .html para reporte visual, .txt o .md
            para texto plano.

        Examples
        --------
        >>> n = Narrator(df, lang="es")
        >>> n.export("reporte.html")   # reporte visual
        >>> n.export("reporte.txt")    # texto plano
        """
        if filepath.endswith(".html"):
            self._export_html(filepath)
        else:
            content = self.describe()
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Análisis exportado a: {filepath}")

    def _export_html(self, filepath: str) -> None:
        """Genera un reporte HTML interactivo completo con Chart.js.

        Incluye navegación por secciones, semáforo de salud por columna,
        gráficas interactivas, histogramas, variables categóricas,
        sugerencias de ML y exportación a PDF.

        Parameters
        ----------
        filepath : str
            Ruta donde se guardará el archivo HTML.
        """
        import base64
        import io
        import json
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ov = self._data["overview"]
        numeric = self._data["numeric"]
        categorical = self._data["categorical"]
        alerts = self._data["alerts"]
        corrs = self._data["correlations"]
        qs = self.quality_score()
        narrative_text = self.narrative().replace("\n", "<br>")
        suggest_text = self.suggest()

        # --- Score donut ---
        fig, ax = plt.subplots(figsize=(3, 3))
        score = qs["score"]
        color = "#22c55e" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
        ax.pie([score, 100 - score], colors=[color, "#e5e7eb"],
               startangle=90, counterclock=False)
        ax.text(0, 0, f"{score}", ha="center", va="center",
                fontsize=28, fontweight="bold", color=color)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, transparent=True)
        plt.close()
        buf.seek(0)
        score_b64 = base64.b64encode(buf.read()).decode("utf-8")

        # --- Histogramas por columna numérica (base64) ---
        hist_b64 = {}
        for c in numeric:
            series = self.df[c["col"]].dropna()
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.hist(series, bins=20, color="#4F46E5", alpha=0.8, edgecolor="white")
            ax.set_xlabel(c["col"], fontsize=8)
            ax.set_ylabel("")
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, transparent=True)
            plt.close()
            buf.seek(0)
            hist_b64[c["col"]] = base64.b64encode(buf.read()).decode("utf-8")

        # --- Semáforo por columna ---
        def get_semaforo(col_name):
            col_alerts = [a for a in alerts if a.get("col") == col_name]
            for a in col_alerts:
                if a["type"] in ("high_nulls", "constant_column"):
                    return "rojo", "#ef4444"
                if a["type"] == "high_cardinality":
                    return "amarillo", "#f59e0b"
            num_col = next((c for c in numeric if c["col"] == col_name), None)
            if num_col:
                if num_col["outlier_count"] > 0 or abs(num_col["skew"]) > 1:
                    return "amarillo", "#f59e0b"
            return "verde", "#22c55e"

        # --- Datos Chart.js ---
        col_types_labels = []
        col_types_data = []
        if ov["numeric_cols"]:
            col_types_labels.append("Numéricas" if self.lang == "es" else "Numeric")
            col_types_data.append(len(ov["numeric_cols"]))
        if ov["categorical_cols"]:
            col_types_labels.append("Categóricas" if self.lang == "es" else "Categorical")
            col_types_data.append(len(ov["categorical_cols"]))
        if ov["datetime_cols"]:
            col_types_labels.append("Fechas" if self.lang == "es" else "Datetime")
            col_types_data.append(len(ov["datetime_cols"]))

        null_labels = []
        null_values = []
        for col in self.df.columns:
            pct = round(self.df[col].isnull().sum() / len(self.df) * 100, 1)
            if pct > 0:
                null_labels.append(col)
                null_values.append(pct)

        outlier_labels = [c["col"] for c in numeric if c["outlier_count"] > 0]
        outlier_values = [c["outlier_count"] for c in numeric if c["outlier_count"] > 0]

        corr_labels = [f"{c['col_a']} ↔ {c['col_b']}" for c in corrs]
        corr_values = [c["correlation"] for c in corrs]
        corr_colors = ["#22c55e" if v > 0 else "#ef4444" for v in corr_values]

        # --- Semáforo HTML (tabla resumen de columnas) ---
        semaforo_rows = ""
        for col in self.df.columns:
            estado, color_hex = get_semaforo(col)
            col_alerts = [a for a in alerts if a.get("col") == col]
            notas = ", ".join(a["type"] for a in col_alerts) if col_alerts else ("ok" if self.lang == "es" else "ok")
            dtype = str(self.df[col].dtype)
            null_pct = round(self.df[col].isnull().sum() / len(self.df) * 100, 1)
            semaforo_rows += f"""
            <tr>
                <td><strong>{col}</strong></td>
                <td><code style="background:#f1f5f9;padding:2px 6px;border-radius:4px;font-size:12px">{dtype}</code></td>
                <td>{null_pct}%</td>
                <td><span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:{color_hex};vertical-align:middle;margin-right:6px"></span>{notas}</td>
            </tr>"""

        # --- Tabla numérica con histogramas ---
        numeric_rows = ""
        for c in numeric:
            estado, color_hex = get_semaforo(c["col"])
            skew_badge = ""
            if abs(c["skew"]) > 1:
                label = f'sesgo {c["skew"]}' if self.lang == "es" else f'skew {c["skew"]}'
                skew_badge = f'<span style="background:#fef3c7;color:#92400e;padding:2px 6px;border-radius:4px;font-size:11px">{label}</span>'
            null_badge = ""
            if c["nulls"] > 0:
                label = f'{c["null_pct"]}% nulos' if self.lang == "es" else f'{c["null_pct"]}% nulls'
                null_badge = f'<span style="background:#fee2e2;color:#991b1b;padding:2px 6px;border-radius:4px;font-size:11px">{label}</span>'
            hist_img = ""
            if c["col"] in hist_b64:
                hist_img = f'<img src="data:image/png;base64,{hist_b64[c["col"]]}" style="width:100%;max-width:220px;display:block;margin-top:6px">'
            numeric_rows += f"""
            <tr>
                <td>
                  <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color_hex};margin-right:6px;vertical-align:middle"></span>
                  <strong>{c["col"]}</strong>
                  {hist_img}
                </td>
                <td>{c["mean"]}</td>
                <td>{c["median"]}</td>
                <td>{c["std"]}</td>
                <td>{c["min"]} – {c["max"]}</td>
                <td>{c["outlier_count"]}</td>
                <td>{skew_badge} {null_badge}</td>
            </tr>"""

        # --- Tabla categórica ---
        cat_rows = ""
        for c in categorical:
            estado, color_hex = get_semaforo(c["col"])
            card_badge = ""
            if c["high_cardinality"]:
                label = "alta cardinalidad" if self.lang == "es" else "high cardinality"
                card_badge = f'<span style="background:#ede9fe;color:#5b21b6;padding:2px 6px;border-radius:4px;font-size:11px">{label}</span>'
            null_badge = ""
            if c["nulls"] > 0:
                label = f'{c["null_pct"]}% nulos' if self.lang == "es" else f'{c["null_pct"]}% nulls'
                null_badge = f'<span style="background:#fee2e2;color:#991b1b;padding:2px 6px;border-radius:4px;font-size:11px">{label}</span>'
            cat_rows += f"""
            <tr>
                <td>
                  <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color_hex};margin-right:6px;vertical-align:middle"></span>
                  <strong>{c["col"]}</strong>
                </td>
                <td>{c["unique"]}</td>
                <td><strong>{c["top_value"]}</strong> ({c["top_pct"]}%)</td>
                <td>{card_badge} {null_badge}</td>
            </tr>"""

        # --- Alertas ---
        alert_rows = ""
        type_colors = {
            "high_nulls": ("#fee2e2", "#991b1b"),
            "duplicates": ("#fef3c7", "#92400e"),
            "high_cardinality": ("#ede9fe", "#5b21b6"),
            "constant_column": ("#f1f5f9", "#475569"),
        }
        for a in alerts:
            msg, sug = self._translate_alert(a)
            bg, fg = type_colors.get(a["type"], ("#f1f5f9", "#475569"))
            alert_rows += f"""
            <tr>
                <td><span style="background:{bg};color:{fg};padding:3px 8px;border-radius:6px;font-size:12px">{a["type"]}</span></td>
                <td>{msg}</td>
                <td>{sug}</td>
            </tr>"""

        # --- Suggest cards ---
        suggest_lines = suggest_text.split("\n")
        suggest_html = ""
        for line in suggest_lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("---"):
                suggest_html += f'<h3 style="margin:1.5rem 0 1rem;color:#334155">{line.replace("-","").strip()}</h3>'
            elif line.startswith("→"):
                suggest_html += f'<div style="background:#f8fafc;border-left:4px solid #4F46E5;border-radius:0 8px 8px 0;padding:0.75rem 1rem;margin:0.5rem 0;font-size:0.9rem">{line[1:].strip()}</div>'
            else:
                suggest_html += f'<p style="margin:0.5rem 0;color:#475569">{line}</p>'

        # Títulos
        if self.lang == "es":
            t_nav = ["Resumen", "Numéricas", "Categóricas", "Calidad", "Alertas", "ML", "Narrativo"]
            t_title = "Reporte de Análisis"
            t_overview = "Resumen General"
            t_semaforo = "Estado de columnas"
            t_numeric = "Variables Numéricas"
            t_categorical = "Variables Categóricas"
            t_quality = "Score de Calidad"
            t_nulls = "Nulos por columna (%)"
            t_outliers = "Outliers por variable"
            t_corr = "Correlaciones"
            t_alerts = "Alertas y Recomendaciones"
            t_suggest = "Sugerencias de Modelado"
            t_narrative = "Análisis Narrativo"
            t_generated = "Generado con datanarrator"
            t_rows = "Registros"
            t_cols = "Columnas"
            t_nullpct = "% Nulos"
            t_dups = "Duplicados"
            t_col = "Columna"
            t_type = "Tipo"
            t_unique = "Únicos"
            t_top = "Más frecuente"
            t_flags = "Flags"
            t_pdf = "Exportar PDF"
            t_no_alerts = "✓ No se detectaron alertas."
            t_no_nulls = "✓ Sin valores nulos"
            t_pen = "Desglose de penalizaciones"
        else:
            t_nav = ["Overview", "Numeric", "Categorical", "Quality", "Alerts", "ML", "Narrative"]
            t_title = "Analysis Report"
            t_overview = "Overview"
            t_semaforo = "Column health"
            t_numeric = "Numeric Variables"
            t_categorical = "Categorical Variables"
            t_quality = "Quality Score"
            t_nulls = "Nulls per column (%)"
            t_outliers = "Outliers per variable"
            t_corr = "Correlations"
            t_alerts = "Alerts & Recommendations"
            t_suggest = "Modeling Suggestions"
            t_narrative = "Narrative Analysis"
            t_generated = "Generated with datanarrator"
            t_rows = "Records"
            t_cols = "Columns"
            t_nullpct = "% Nulls"
            t_dups = "Duplicates"
            t_col = "Column"
            t_type = "Type"
            t_unique = "Unique"
            t_top = "Most frequent"
            t_flags = "Flags"
            t_pdf = "Export PDF"
            t_no_alerts = "✓ No alerts detected."
            t_no_nulls = "✓ No null values"
            t_pen = "Penalty breakdown"

        # Pre-calculamos secciones condicionales para evitar
        # backslashes dentro de f-strings (incompatible con Python 3.10)
        if outlier_labels:
            outliers_section = (
                '<div class="chart-box" style="margin-bottom:1.5rem"><h3>'
                + t_outliers
                + '</h3><div class="chart-container">'
                + '<canvas id="outliersChart"></canvas></div></div>'
            )
        else:
            outliers_section = ""

        if corrs:
            corr_section = (
                '<h3>' + t_corr + '</h3>'
                '<div class="chart-box" style="margin-top:1rem">'
                '<div class="chart-container">'
                '<canvas id="corrChart"></canvas></div></div>'
            )
            corr_chart_js = (
                'new Chart(document.getElementById("corrChart"),{'
                'type:"bar",data:{labels:' + json.dumps(corr_labels)
                + ',datasets:[{label:"r",data:' + json.dumps(corr_values)
                + ',backgroundColor:' + json.dumps(corr_colors)
                + ',borderRadius:4}]},options:{responsive:true,'
                'maintainAspectRatio:false,indexAxis:"y",'
                'scales:{x:{min:-1,max:1}},plugins:{legend:{display:false}}}});'
            )
        else:
            corr_section = ""
            corr_chart_js = ""

        if null_labels:
            nulls_chart_js = (
                'new Chart(document.getElementById("nullsChart"),{'
                'type:"bar",data:{labels:' + json.dumps(null_labels)
                + ',datasets:[{label:"% nulls",data:' + json.dumps(null_values)
                + ',backgroundColor:"#ef4444",borderRadius:4}]},'
                'options:{responsive:true,maintainAspectRatio:false,'
                'scales:{y:{beginAtZero:true,max:100}},'
                'plugins:{legend:{display:false}}}});'
            )
        else:
            nulls_chart_js = ""

        if outlier_labels:
            outliers_chart_js = (
                'new Chart(document.getElementById("outliersChart"),{'
                'type:"bar",data:{labels:' + json.dumps(outlier_labels)
                + ',datasets:[{label:"Outliers",data:' + json.dumps(outlier_values)
                + ',backgroundColor:"#f59e0b",borderRadius:4}]},'
                'options:{responsive:true,maintainAspectRatio:false,'
                'plugins:{legend:{display:false}}}});'
            )
        else:
            outliers_chart_js = ""

        html = f"""<!DOCTYPE html>
<html lang="{self.lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>datanarrator — {t_title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f8fafc;color:#1e293b;min-height:100vh}}
  nav{{background:#4F46E5;padding:0 1.5rem;display:flex;align-items:center;gap:0;position:sticky;top:0;z-index:100;box-shadow:0 2px 8px rgba(79,70,229,0.3);flex-wrap:wrap}}
  .brand{{color:white;font-weight:700;font-size:1rem;padding:0.875rem 1rem 0.875rem 0;border-right:1px solid rgba(255,255,255,0.2);margin-right:0.25rem}}
  nav button{{background:none;border:none;color:rgba(255,255,255,0.75);padding:0.875rem 0.9rem;cursor:pointer;font-size:0.82rem;transition:all 0.2s;border-bottom:3px solid transparent;white-space:nowrap}}
  nav button:hover{{color:white}}
  nav button.active{{color:white;border-bottom:3px solid white;font-weight:600}}
  .pdf-btn{{margin-left:auto;background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.3)!important;border-radius:6px;color:white!important;padding:0.5rem 1rem!important;font-size:0.82rem;cursor:pointer}}
  .pdf-btn:hover{{background:rgba(255,255,255,0.25)!important}}
  .section{{display:none;padding:2rem;max-width:1100px;margin:0 auto}}
  .section.active{{display:block}}
  h2{{font-size:1.3rem;font-weight:700;color:#1e293b;margin-bottom:1.5rem;padding-bottom:0.75rem;border-bottom:2px solid #e2e8f0}}
  h3{{font-size:1rem;font-weight:600;color:#334155;margin:1.5rem 0 1rem}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:1rem;margin-bottom:2rem}}
  .card{{background:white;border-radius:12px;padding:1.25rem;box-shadow:0 1px 3px rgba(0,0,0,0.08);text-align:center}}
  .card-value{{font-size:1.8rem;font-weight:700;color:#4F46E5}}
  .card-label{{font-size:0.78rem;color:#64748b;margin-top:0.25rem}}
  .charts-grid{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:2rem}}
  .chart-box{{background:white;border-radius:12px;padding:1.5rem;box-shadow:0 1px 3px rgba(0,0,0,0.08)}}
  .chart-box h3{{margin-top:0;margin-bottom:1rem}}
  .chart-container{{position:relative;height:200px}}
  table{{width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.08);margin-bottom:1.5rem}}
  th{{background:#4F46E5;color:white;padding:0.7rem 1rem;text-align:left;font-weight:600;font-size:0.82rem;cursor:pointer;user-select:none}}
  th:hover{{background:#4338ca}}
  td{{padding:0.65rem 1rem;border-bottom:1px solid #f1f5f9;font-size:0.88rem;vertical-align:top}}
  tr:last-child td{{border-bottom:none}}
  tr:hover td{{background:#f8fafc}}
  .score-box{{background:white;border-radius:12px;padding:1.5rem;box-shadow:0 1px 3px rgba(0,0,0,0.08);display:flex;align-items:center;gap:2rem;margin-bottom:2rem}}
  .score-box img{{width:130px;flex-shrink:0}}
  .pen-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin-top:1rem}}
  .pen-card{{background:#f8fafc;border-radius:8px;padding:1rem;border-left:4px solid #4F46E5}}
  .pen-label{{font-size:0.78rem;color:#64748b;text-transform:uppercase}}
  .pen-value{{font-size:1.3rem;font-weight:700;color:#4F46E5;margin-top:0.25rem}}
  .narrative-box{{background:white;border-radius:12px;padding:2rem;box-shadow:0 1px 3px rgba(0,0,0,0.08);line-height:1.8;color:#334155}}
  footer{{text-align:center;padding:2rem;color:#94a3b8;font-size:0.8rem}}
  @media print{{nav{{display:none}}.section{{display:block!important;page-break-after:always}}}}
  @media(max-width:700px){{.charts-grid{{grid-template-columns:1fr}}.score-box{{flex-direction:column}}}}
</style>
</head>
<body>
<nav>
  <div class="brand">datanarrator</div>
  <button class="active" onclick="showSection(0)">{t_nav[0]}</button>
  <button onclick="showSection(1)">{t_nav[1]}</button>
  <button onclick="showSection(2)">{t_nav[2]}</button>
  <button onclick="showSection(3)">{t_nav[3]}</button>
  <button onclick="showSection(4)">{t_nav[4]}</button>
  <button onclick="showSection(5)">{t_nav[5]}</button>
  <button onclick="showSection(6)">{t_nav[6]}</button>
  <button class="pdf-btn" onclick="window.print()">{t_pdf}</button>
</nav>

<!-- S0: RESUMEN -->
<div class="section active" id="s0">
  <h2>{t_overview}</h2>
  <div class="cards">
    <div class="card"><div class="card-value">{ov["rows"]:,}</div><div class="card-label">{t_rows}</div></div>
    <div class="card"><div class="card-value">{ov["cols"]}</div><div class="card-label">{t_cols}</div></div>
    <div class="card"><div class="card-value">{ov["null_pct"]}%</div><div class="card-label">{t_nullpct}</div></div>
    <div class="card"><div class="card-value">{ov["duplicates"]}</div><div class="card-label">{t_dups}</div></div>
    <div class="card"><div class="card-value">{qs["score"]}</div><div class="card-label">Quality Score</div></div>
    <div class="card"><div class="card-value">{qs["grade"]}</div><div class="card-label">Grade</div></div>
  </div>
  <div class="charts-grid">
    <div class="chart-box">
      <h3>{"Tipos de columnas" if self.lang == "es" else "Column types"}</h3>
      <div class="chart-container"><canvas id="colTypesChart"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>{t_nulls}</h3>
      {"<div class=\"chart-container\"><canvas id=\"nullsChart\"></canvas></div>" if null_labels else f'<p style="color:#22c55e;padding:1rem">{t_no_nulls}</p>'}
    </div>
  </div>
  <h3>{t_semaforo}</h3>
  <table>
    <thead><tr>
      <th>{t_col}</th>
      <th>{t_type}</th>
      <th>% Nulls</th>
      <th>Estado</th>
    </tr></thead>
    <tbody>{semaforo_rows}</tbody>
  </table>
</div>

<!-- S1: NUMÉRICAS -->
<div class="section" id="s1">
  <h2>{t_numeric}</h2>
  {outliers_section}
  <table id="numericTable">
    <thead><tr>
      <th onclick="sortTable('numericTable',0)">{t_col} ↕</th>
      <th onclick="sortTable('numericTable',1)">{"Media" if self.lang == "es" else "Mean"} ↕</th>
      <th onclick="sortTable('numericTable',2)">{"Mediana" if self.lang == "es" else "Median"} ↕</th>
      <th onclick="sortTable('numericTable',3)">Std ↕</th>
      <th>{"Rango" if self.lang == "es" else "Range"}</th>
      <th onclick="sortTable('numericTable',5)">Outliers ↕</th>
      <th>{t_flags}</th>
    </tr></thead>
    <tbody>{numeric_rows}</tbody>
  </table>
</div>

<!-- S2: CATEGÓRICAS -->
<div class="section" id="s2">
  <h2>{t_categorical}</h2>
  {"<p style=\"color:#64748b;padding:1rem\">No hay variables categóricas.</p>" if not categorical else f'<table id="catTable"><thead><tr><th onclick="sortTable(\'catTable\',0)">{t_col} ↕</th><th onclick="sortTable(\'catTable\',1)">{t_unique} ↕</th><th>{t_top}</th><th>{t_flags}</th></tr></thead><tbody>{cat_rows}</tbody></table>'}
</div>

<!-- S3: CALIDAD -->
<div class="section" id="s3">
  <h2>{t_quality}</h2>
  <div class="score-box">
    <img src="data:image/png;base64,{score_b64}" alt="score">
    <div>
      <h3 style="margin:0 0 0.5rem">{qs["resumen"]}</h3>
      <p style="color:#475569">{t_quality}: <strong>{qs["score"]}/100</strong> — Grade <strong>{qs["grade"]}</strong></p>
    </div>
  </div>
  <h3>{t_pen}</h3>
  <div class="pen-grid">
    {"".join(f'<div class="pen-card"><div class="pen-label">{k}</div><div class="pen-value">-{round(v,1)}</div></div>' for k, v in qs["penalizaciones"].items() if v > 0)}
  </div>
  {corr_section}
</div>

<!-- S4: ALERTAS -->
<div class="section" id="s4">
  <h2>{t_alerts}</h2>
  {"<p style=\"color:#22c55e;padding:1rem;background:white;border-radius:12px\">" + t_no_alerts + "</p>" if not alerts else f'<table><thead><tr><th>Tipo</th><th>Mensaje</th><th>{"Sugerencia" if self.lang == "es" else "Suggestion"}</th></tr></thead><tbody>{alert_rows}</tbody></table>'}
</div>

<!-- S5: ML -->
<div class="section" id="s5">
  <h2>{t_suggest}</h2>
  <div style="background:white;border-radius:12px;padding:1.5rem;box-shadow:0 1px 3px rgba(0,0,0,0.08)">
    {suggest_html}
  </div>
</div>

<!-- S6: NARRATIVO -->
<div class="section" id="s6">
  <h2>{t_narrative}</h2>
  <div class="narrative-box">{narrative_text}</div>
</div>

<footer>{t_generated} v{self._get_version()} · <a href="https://pypi.org/project/datanarrator/" style="color:#4F46E5">PyPI</a></footer>

<script>
function showSection(idx){{
  document.querySelectorAll(".section").forEach((s,i)=>s.classList.toggle("active",i===idx));
  document.querySelectorAll("nav button:not(.pdf-btn)").forEach((b,i)=>b.classList.toggle("active",i===idx));
}}

function sortTable(id,col){{
  const table=document.getElementById(id);
  const rows=Array.from(table.querySelectorAll("tbody tr"));
  const asc=table.dataset.sort==col&&table.dataset.dir=="asc"?false:true;
  table.dataset.sort=col;table.dataset.dir=asc?"asc":"desc";
  rows.sort((a,b)=>{{
    const va=a.cells[col].textContent.trim();
    const vb=b.cells[col].textContent.trim();
    const na=parseFloat(va),nb=parseFloat(vb);
    if(!isNaN(na)&&!isNaN(nb))return asc?na-nb:nb-na;
    return asc?va.localeCompare(vb):vb.localeCompare(va);
  }});
  rows.forEach(r=>table.querySelector("tbody").appendChild(r));
}}

const COLORS=["#4F46E5","#06b6d4","#f59e0b","#22c55e","#ef4444","#a855f7"];

new Chart(document.getElementById("colTypesChart"),{{
  type:"doughnut",
  data:{{labels:{json.dumps(col_types_labels)},datasets:[{{data:{json.dumps(col_types_data)},backgroundColor:COLORS}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:"bottom"}}}}}}
}});

{nulls_chart_js}

{outliers_chart_js}

{corr_chart_js}
</script>
</body>
</html>"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Reporte HTML exportado a: {filepath}")

    def _get_version(self) -> str:
        """Retorna la versión instalada de datanarrator.

        Returns
        -------
        str
            Versión de la librería o 'dev' si no está instalada.
        """
        try:
            from importlib.metadata import version
            return version("datanarrator")
        except Exception:
            return "dev"

    def compare(self, df2):
        """Compara el DataFrame original con un segundo DataFrame.

        Detecta diferencias en tamaño, columnas, distribuciones
        y posible data drift entre dos datasets. Útil para comparar
        datos de entrenamiento vs producción.

        Parameters
        ----------
        df2 : pd.DataFrame
            El segundo DataFrame con el que se quiere comparar.

        Returns
        -------
        str
            Texto describiendo las diferencias encontradas entre
            ambos datasets, incluyendo alertas de data drift.

        Raises
        ------
        TypeError
            Si df2 no es un DataFrame de pandas.
        ValueError
            Si df2 está vacío.

        Examples
        --------
        >>> n = Narrator(df_train, lang="es")
        >>> print(n.compare(df_produccion))
        """
        if not isinstance(df2, pd.DataFrame):
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
                # Si la desviación estándar cambió más del 30% consideramos que hay drift
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
                # Si la desviación estándar cambió más del 30% consideramos que hay drift
                if abs(s2 - s1) / s1 * 100 >= 30:
                    drift.append(col)
            if drift:
                out.append(f"⚠  Possible data drift in: {', '.join(drift)}.")
                out.append("     → Dispersion changed over 30%. Review before production.")
        return "\n".join(out)

    def narrative(self) -> str:
        """Genera un análisis narrativo del dataset en lenguaje natural.

        A diferencia de describe(), que presenta estadísticas en formato
        estructurado, narrative() genera párrafos analíticos con
        interpretaciones, contexto estadístico y recomendaciones
        específicas basadas en los datos reales del dataset.

        Returns
        -------
        str
            Texto narrativo completo con párrafos analíticos. Respeta
            el idioma configurado en lang.

        Examples
        --------
        >>> import pandas as pd
        >>> from datanarrator import Narrator
        >>> df = pd.read_csv("titanic.csv")
        >>> n = Narrator(df, lang="es")
        >>> print(n.narrative())
        """
        sections = [
            self._narrative_overview(),
            self._narrative_numeric(),
            self._narrative_correlations(),
            self._narrative_alerts(),
            self._narrative_recommendations(),
        ]
        return "\n\n".join(s for s in sections if s)

    def _narrative_overview(self) -> str:
        """Genera el párrafo de contexto general del dataset.

        Returns
        -------
        str
            Párrafo interpretando tamaño, composición y calidad general.
        """
        ov = self._data["overview"]
        qs = self.quality_score()
        n_rows = ov["rows"]
        n_cols = ov["cols"]
        n_num = len(ov["numeric_cols"])
        n_cat = len(ov["categorical_cols"])
        null_pct = ov["null_pct"]
        dups = ov["duplicates"]
        score = qs["score"]
        grade = qs["grade"]
        alerts = self._data["alerts"]

        # Columna con más nulos para mencionarla explícitamente
        worst_null = max(
            alerts, key=lambda a: float(a["message"].split()[2].replace("%",""))
            if a["type"] == "high_nulls" else 0,
            default=None
        )

        if self.lang == "es":
            if n_rows < 500:
                size_desc = "pequeño"
                size_note = "lo que puede limitar la generalización de modelos complejos"
            elif n_rows < 10000:
                size_desc = "de tamaño moderado"
                size_note = "adecuado para la mayoría de algoritmos de machine learning"
            else:
                size_desc = "grande"
                size_note = "con volumen suficiente para entrenar modelos robustos"

            parrafo = (
                f"Con {n_rows:,} registros y {n_cols} columnas, el dataset es "
                f"{size_desc} — {size_note}. "
                f"Contiene {n_num} variables numéricas y {n_cat} categóricas"
            )

            if null_pct == 0 and dups == 0:
                parrafo += (
                    f", y se encuentra completamente limpio: sin valores nulos "
                    f"ni duplicados, lo que es poco común y representa una ventaja "
                    f"significativa. Score de calidad: {score}/100 (grado {grade})."
                )
            else:
                parrafo += f". "
                if worst_null:
                    col_problematica = worst_null.get("col", "")
                    pct_problematica = worst_null["message"].split()[2]
                    parrafo += (
                        f"El problema de calidad más crítico es la columna "
                        f"\"{col_problematica}\", que concentra el {pct_problematica} "
                        f"de valores nulos — un nivel tan alto sugiere que esta "
                        f"información simplemente no estaba disponible para la mayoría "
                        f"de los registros y probablemente deba eliminarse. "
                    )
                if null_pct > 0:
                    parrafo += (
                        f"En total, el {null_pct}% de todas las celdas del dataset "
                        f"están vacías. "
                    )
                if dups > 0:
                    parrafo += (
                        f"Además, se detectaron {dups} registros duplicados que "
                        f"introducen sesgo si no se eliminan antes de modelar. "
                    )
                parrafo += f"Score de calidad global: {score}/100 (grado {grade})."
        else:
            if n_rows < 500:
                size_desc = "small"
                size_note = "which may limit generalization of complex models"
            elif n_rows < 10000:
                size_desc = "moderate-sized"
                size_note = "suitable for most machine learning algorithms"
            else:
                size_desc = "large"
                size_note = "with sufficient volume to train robust models"

            parrafo = (
                f"With {n_rows:,} records and {n_cols} columns, the dataset is "
                f"{size_desc} — {size_note}. "
                f"It contains {n_num} numeric and {n_cat} categorical variables"
            )

            if null_pct == 0 and dups == 0:
                parrafo += (
                    f", and is completely clean: no null values or duplicates, "
                    f"which is uncommon and a significant advantage. "
                    f"Quality score: {score}/100 (grade {grade})."
                )
            else:
                parrafo += f". "
                if worst_null:
                    col_problematica = worst_null.get("col", "")
                    pct_problematica = worst_null["message"].split()[2]
                    parrafo += (
                        f"The most critical quality issue is column "
                        f"\"{col_problematica}\", with {pct_problematica} null values — "
                        f"a level this high suggests the information was simply "
                        f"unavailable for most records and should likely be dropped. "
                    )
                if null_pct > 0:
                    parrafo += (
                        f"Overall, {null_pct}% of all dataset cells are empty. "
                    )
                if dups > 0:
                    parrafo += (
                        f"Additionally, {dups} duplicate records were found that "
                        f"introduce bias if not removed before modeling. "
                    )
                parrafo += f"Overall quality score: {score}/100 (grade {grade})."

        return parrafo

    def _narrative_numeric(self) -> str:
        """Genera párrafos analíticos sobre las variables numéricas.

        En lugar de listar estadísticas columna por columna, agrupa
        observaciones por patrón: variables identificadoras, variables
        objetivo, distribuciones sesgadas y variables con problemas.

        Returns
        -------
        str
            Párrafos interpretativos de las variables numéricas.
        """
        cols = self._data["numeric"]
        if not cols:
            return ""

        parrafos = []

        # Detectamos variables que parecen identificadores (ID)
        id_cols = [
            c for c in cols
            if c["mean"] == c["median"]
            and c["col"].lower() in ["id", "passengerid", "customerid",
                                      "userid", "index", "rowid"]
        ]

        # Detectamos posibles variables objetivo binarias
        binary_cols = [
            c for c in cols
            if set(self.df[c["col"]].dropna().unique()) <= {0, 1}
            and c["col"].lower() not in ["id", "passengerid"]
        ]

        # Detectamos variables con sesgo significativo
        skewed_cols = [c for c in cols if abs(c["skew"]) > 1]

        # Detectamos variables con muchos outliers
        outlier_heavy = [
            c for c in cols
            if c["outlier_count"] / len(self.df) > 0.10
        ]

        # Detectamos variables con nulos significativos
        null_heavy = [c for c in cols if c["null_pct"] > 5]

        if self.lang == "es":
            if id_cols:
                cols_str = ", ".join(c["col"] for c in id_cols)
                parrafos.append(
                    f"La columna {cols_str} parece ser un identificador único "
                    f"sin valor predictivo — su distribución perfectamente uniforme "
                    f"confirma que es solo un índice y debe excluirse del modelado."
                )

            if binary_cols:
                for c in binary_cols:
                    pct_positivo = round(c["mean"] * 100, 1)
                    pct_negativo = round((1 - c["mean"]) * 100, 1)
                    balance = "desbalanceada" if abs(pct_positivo - 50) > 15 else "relativamente balanceada"
                    parrafos.append(
                        f"La variable {c['col']} es binaria y representa una "
                        f"candidata natural como variable objetivo para clasificación. "
                        f"El {pct_positivo}% de los registros tienen valor positivo "
                        f"frente al {pct_negativo}% negativo — una distribución "
                        f"{balance}. "
                        + (f"El desbalance de clases sugiere usar métricas como "
                           f"F1-score o AUC-ROC en lugar de accuracy simple."
                           if balance == "desbalanceada" else
                           f"El balance entre clases es favorable para el entrenamiento.")
                    )

            regular_cols = [
                c for c in cols
                if c not in id_cols and c not in binary_cols
            ]
            if regular_cols:
                # Separamos las que tienen comportamiento interesante
                interesting = [
                    c for c in regular_cols
                    if abs(c["skew"]) > 0.5
                    or c["outlier_count"] > 0
                    or c["nulls"] > 0
                ]
                normal = [c for c in regular_cols if c not in interesting]

                if normal:
                    cols_str = ", ".join(c["col"] for c in normal)
                    parrafos.append(
                        f"Las variables {cols_str} presentan distribuciones "
                        f"relativamente simétricas y sin problemas mayores, "
                        f"listas para usarse directamente en el modelado."
                    )

                for c in interesting:
                    linea = f"La variable {c['col']} "
                    if abs(c["skew"]) > 1:
                        direccion = "positivo" if c["skew"] > 0 else "negativo"
                        linea += (
                            f"muestra un sesgo {direccion} pronunciado "
                            f"(media={c['mean']}, mediana={c['median']}): "
                        )
                        if c["skew"] > 0:
                            linea += (
                                f"la mayoría de los valores se concentran en "
                                f"valores bajos pero hay casos extremos hacia arriba "
                                f"que inflan el promedio. "
                            )
                        else:
                            linea += (
                                f"hay valores extremos hacia abajo que "
                                f"reducen el promedio por debajo de la mediana. "
                            )
                    elif abs(c["mean"] - c["median"]) / (abs(c["median"]) + 1e-9) > 0.05:
                        linea += (
                            f"tiene una ligera asimetría "
                            f"(media={c['mean']}, mediana={c['median']}). "
                        )
                    else:
                        linea += (
                            f"es relativamente simétrica "
                            f"(media={c['mean']}, mediana={c['median']}). "
                        )

                    if c["outlier_count"] > 0:
                        pct = round(c["outlier_count"] / len(self.df) * 100, 1)
                        if pct > 10:
                            linea += (
                                f"El {pct}% de outliers es alto — "
                                f"podría indicar subpoblaciones distintas "
                                f"o errores de captura. "
                            )
                        else:
                            linea += (
                                f"Presenta {c['outlier_count']} valores atípicos "
                                f"({pct}%) que conviene revisar antes de modelar. "
                            )

                    if c["nulls"] > 0:
                        if c["null_pct"] > 20:
                            linea += (
                                f"El {c['null_pct']}% de nulos es crítico — "
                                f"evalúa si la ausencia de dato es informativa "
                                f"en sí misma antes de imputar."
                            )
                        else:
                            linea += (
                                f"El {c['null_pct']}% de nulos es manejable "
                                f"mediante imputación con la mediana."
                            )
                    parrafos.append(linea)
        else:
            if id_cols:
                cols_str = ", ".join(c["col"] for c in id_cols)
                parrafos.append(
                    f"The column {cols_str} appears to be a unique identifier "
                    f"with no predictive value — its perfectly uniform distribution "
                    f"confirms it is just an index and should be excluded from modeling."
                )

            if binary_cols:
                for c in binary_cols:
                    pct_pos = round(c["mean"] * 100, 1)
                    pct_neg = round((1 - c["mean"]) * 100, 1)
                    balance = "imbalanced" if abs(pct_pos - 50) > 15 else "relatively balanced"
                    parrafos.append(
                        f"Variable {c['col']} is binary and a natural candidate "
                        f"as a target variable for classification. "
                        f"{pct_pos}% of records are positive vs {pct_neg}% negative — "
                        f"a {balance} distribution. "
                        + (f"Class imbalance suggests using F1-score or AUC-ROC "
                           f"instead of plain accuracy."
                           if balance == "imbalanced" else
                           f"The class balance is favorable for training.")
                    )

            regular_cols = [
                c for c in cols
                if c not in id_cols and c not in binary_cols
            ]
            if regular_cols:
                interesting = [
                    c for c in regular_cols
                    if abs(c["skew"]) > 0.5
                    or c["outlier_count"] > 0
                    or c["nulls"] > 0
                ]
                normal = [c for c in regular_cols if c not in interesting]

                if normal:
                    cols_str = ", ".join(c["col"] for c in normal)
                    parrafos.append(
                        f"Variables {cols_str} show relatively symmetric "
                        f"distributions with no major issues, ready for direct use."
                    )

                for c in interesting:
                    linea = f"Variable {c['col']} "
                    if abs(c["skew"]) > 1:
                        direction = "positive" if c["skew"] > 0 else "negative"
                        linea += (
                            f"shows a pronounced {direction} skew "
                            f"(mean={c['mean']}, median={c['median']}): "
                        )
                        if c["skew"] > 0:
                            linea += (
                                f"most values cluster at the low end but extreme "
                                f"high cases inflate the average. "
                            )
                        else:
                            linea += (
                                f"extreme low values pull the average below the median. "
                            )
                    else:
                        linea += (
                            f"is relatively symmetric "
                            f"(mean={c['mean']}, median={c['median']}). "
                        )

                    if c["outlier_count"] > 0:
                        pct = round(c["outlier_count"] / len(self.df) * 100, 1)
                        if pct > 10:
                            linea += (
                                f"{pct}% outlier rate is high — could indicate "
                                f"distinct subpopulations or data entry errors. "
                            )
                        else:
                            linea += (
                                f"{c['outlier_count']} outliers ({pct}%) worth "
                                f"reviewing before modeling. "
                            )

                    if c["nulls"] > 0:
                        if c["null_pct"] > 20:
                            linea += (
                                f"{c['null_pct']}% nulls is critical — consider "
                                f"whether missingness itself is informative before imputing."
                            )
                        else:
                            linea += (
                                f"{c['null_pct']}% nulls is manageable "
                                f"via median imputation."
                            )
                    parrafos.append(linea)

        return "\n\n".join(parrafos)

    def _narrative_correlations(self) -> str:
        """Genera párrafos interpretativos sobre correlaciones relevantes.

        Returns
        -------
        str
            Párrafo interpretando correlaciones con significado analítico.
        """
        corrs = self._data["correlations"]
        if not corrs:
            return ""

        parrafos = []

        if self.lang == "es":
            intro = (
                f"El análisis de correlaciones revela "
                f"{len(corrs)} relación(es) estadísticamente relevante(s):"
            )
            parrafos.append(intro)
            for c in corrs:
                val = c["correlation"]
                a, b = c["col_a"], c["col_b"]
                abs_val = abs(val)

                if abs_val >= 0.9:
                    fuerza = "muy alta"
                    riesgo = (
                        "Esta correlación es tan fuerte que una de las dos "
                        "variables podría ser redundante — considera eliminar "
                        "una de ellas para evitar multicolinealidad severa."
                    )
                elif abs_val >= 0.75:
                    fuerza = "alta"
                    riesgo = (
                        "En modelos lineales esto puede introducir "
                        "multicolinealidad — vale la pena evaluar si ambas "
                        "variables son necesarias."
                    )
                else:
                    fuerza = "moderada"
                    riesgo = (
                        "Una señal útil para el modelo pero sin riesgo "
                        "significativo de multicolinealidad."
                    )

                direccion = "positiva" if val > 0 else "negativa"
                interpretacion = (
                    f"a mayor {a}, mayor {b}" if val > 0
                    else f"a mayor {a}, menor {b}"
                )

                parrafos.append(
                    f"  {a} ↔ {b}: correlación {fuerza} {direccion} ({val}), "
                    f"es decir, {interpretacion}. {riesgo}"
                )
        else:
            intro = (
                f"Correlation analysis reveals "
                f"{len(corrs)} statistically relevant relationship(s):"
            )
            parrafos.append(intro)
            for c in corrs:
                val = c["correlation"]
                a, b = c["col_a"], c["col_b"]
                abs_val = abs(val)

                if abs_val >= 0.9:
                    fuerza = "very high"
                    riesgo = (
                        "This correlation is so strong that one variable "
                        "may be redundant — consider dropping one to avoid "
                        "severe multicollinearity."
                    )
                elif abs_val >= 0.75:
                    fuerza = "high"
                    riesgo = (
                        "In linear models this may introduce multicollinearity "
                        "— worth evaluating whether both variables are needed."
                    )
                else:
                    fuerza = "moderate"
                    riesgo = (
                        "A useful signal for the model without significant "
                        "multicollinearity risk."
                    )

                direction = "positive" if val > 0 else "negative"
                interpretation = (
                    f"higher {a} tends to mean higher {b}" if val > 0
                    else f"higher {a} tends to mean lower {b}"
                )

                parrafos.append(
                    f"  {a} ↔ {b}: {fuerza} {direction} correlation ({val}), "
                    f"meaning {interpretation}. {riesgo}"
                )

        return "\n".join(parrafos)

    def _narrative_alerts(self) -> str:
        """Genera párrafo de problemas críticos priorizados por severidad.

        Returns
        -------
        str
            Párrafo describiendo problemas con contexto de impacto.
        """
        alerts = self._data["alerts"]
        if not alerts:
            if self.lang == "es":
                return (
                    "No se detectaron problemas críticos en el dataset. "
                    "La estructura y calidad de los datos es adecuada para "
                    "proceder directamente al análisis o modelado."
                )
            return (
                "No critical issues detected in the dataset. "
                "Data structure and quality are adequate to proceed "
                "directly to analysis or modeling."
            )

        priority = {
            "high_nulls": 0, "duplicates": 1,
            "high_cardinality": 2, "constant_column": 3
        }
        sorted_alerts = sorted(
            alerts, key=lambda a: priority.get(a["type"], 99)
        )

        if self.lang == "es":
            intro = (
                f"Se identificaron {len(alerts)} problema(s) de calidad. "
                f"En orden de impacto potencial en el modelado:"
            )
            lineas = [intro]
            for a in sorted_alerts:
                msg, sug = self._translate_alert(a)
                lineas.append(f"  • {msg} {sug}")
        else:
            intro = (
                f"{len(alerts)} data quality issue(s) identified. "
                f"In order of potential modeling impact:"
            )
            lineas = [intro]
            for a in sorted_alerts:
                msg, sug = self._translate_alert(a)
                lineas.append(f"  • {msg} {sug}")

        return "\n".join(lineas)

    def _narrative_recommendations(self) -> str:
        """Genera recomendaciones concretas y priorizadas de preprocesamiento.

        Returns
        -------
        str
            Párrafo con pasos específicos basados en los problemas
            y características detectadas en el dataset.
        """
        ov = self._data["overview"]
        numeric = self._data["numeric"]
        alerts = self._data["alerts"]
        corrs = self._data["correlations"]
        qs = self.quality_score()

        pasos = []

        # Paso 1: duplicados primero
        if ov["duplicates"] > 0:
            if self.lang == "es":
                pasos.append(
                    f"Eliminar los {ov['duplicates']} registros duplicados "
                    f"como primer paso, antes de cualquier transformación."
                )
            else:
                pasos.append(
                    f"Remove the {ov['duplicates']} duplicate records "
                    f"as a first step, before any transformation."
                )

        # Paso 2: columnas con nulos críticos (>20%)
        critical_null = [a for a in alerts if a["type"] == "high_nulls"]
        for a in critical_null:
            col = a.get("col", "")
            pct = a["message"].split()[2]
            if self.lang == "es":
                if float(pct.replace("%","")) > 50:
                    pasos.append(
                        f"Eliminar la columna \"{col}\" ({pct} de nulos) — "
                        f"con más del 50% de datos faltantes, la imputación "
                        f"introduciría más ruido que información."
                    )
                else:
                    pasos.append(
                        f"Imputar \"{col}\" ({pct} de nulos) — "
                        f"evalúa si usar la mediana, una constante especial "
                        f"o un modelo de imputación según el contexto."
                    )
            else:
                if float(pct.replace("%","")) > 50:
                    pasos.append(
                        f"Drop column \"{col}\" ({pct} nulls) — "
                        f"with over 50% missing data, imputation would "
                        f"introduce more noise than information."
                    )
                else:
                    pasos.append(
                        f"Impute \"{col}\" ({pct} nulls) — "
                        f"consider median, a special constant, or a model-based "
                        f"imputation depending on context."
                    )

        # Paso 3: imputar nulos en numéricas
        null_cols = [c for c in numeric if c["nulls"] > 0]
        for c in null_cols:
            already_covered = any(
                a.get("col") == c["col"] for a in critical_null
            )
            if not already_covered:
                if self.lang == "es":
                    estrategia = "mediana" if abs(c["skew"]) > 1 else "media"
                    pasos.append(
                        f"Imputar nulos en \"{c['col']}\" ({c['null_pct']}%) "
                        f"con la {estrategia} — su distribución "
                        f"{'sesgada' if abs(c['skew']) > 1 else 'simétrica'} "
                        f"hace que sea la opción más robusta."
                    )
                else:
                    strategy = "median" if abs(c["skew"]) > 1 else "mean"
                    pasos.append(
                        f"Impute nulls in \"{c['col']}\" ({c['null_pct']}%) "
                        f"with the {strategy} — its "
                        f"{'skewed' if abs(c['skew']) > 1 else 'symmetric'} "
                        f"distribution makes this the most robust option."
                    )

        # Paso 4: outliers en columnas con sesgo alto
        skewed_with_outliers = [
            c for c in numeric
            if abs(c["skew"]) > 1 and c["outlier_count"] > 0
        ]
        if skewed_with_outliers:
            cols_str = ", ".join(f'\"{c["col"]}\"' for c in skewed_with_outliers)
            if self.lang == "es":
                pasos.append(
                    f"Aplicar transformación logarítmica en {cols_str} "
                    f"para reducir el sesgo y el impacto de los outliers "
                    f"antes de escalar o modelar."
                )
            else:
                pasos.append(
                    f"Apply log transformation to {cols_str} "
                    f"to reduce skew and outlier impact before scaling or modeling."
                )

        # Paso 5: encoding
        high_card = [a["col"] for a in alerts if a["type"] == "high_cardinality"]
        normal_cat = [c for c in ov["categorical_cols"] if c not in high_card]

        if high_card:
            cols_str = ", ".join(f'\"{c}\"' for c in high_card)
            if self.lang == "es":
                pasos.append(
                    f"Para {cols_str} (alta cardinalidad), usar target encoding "
                    f"o embeddings — el one-hot encoding generaría cientos de "
                    f"columnas dispersas que degradan el rendimiento del modelo."
                )
            else:
                pasos.append(
                    f"For {cols_str} (high cardinality), use target encoding "
                    f"or embeddings — one-hot encoding would generate hundreds "
                    f"of sparse columns degrading model performance."
                )

        if normal_cat:
            cols_str = ", ".join(f'\"{c}\"' for c in normal_cat)
            if self.lang == "es":
                pasos.append(
                    f"Aplicar one-hot encoding a {cols_str} "
                    f"— su baja cardinalidad hace que sea seguro y eficiente."
                )
            else:
                pasos.append(
                    f"Apply one-hot encoding to {cols_str} "
                    f"— their low cardinality makes it safe and efficient."
                )

        # Paso 6: multicolinealidad
        high_corr = [c for c in corrs if abs(c["correlation"]) >= 0.75]
        if high_corr:
            pairs = ", ".join(
                f'\"{c["col_a"]}\" y \"{c["col_b"]}\"' for c in high_corr
            )
            if self.lang == "es":
                pasos.append(
                    f"Evaluar multicolinealidad entre {pairs}: "
                    f"considera eliminar una variable de cada par o aplicar PCA "
                    f"si vas a usar regresión lineal o logística."
                )
            else:
                pairs_en = ", ".join(
                    f'\"{c["col_a"]}\" and \"{c["col_b"]}\"' for c in high_corr
                )
                pasos.append(
                    f"Evaluate multicollinearity between {pairs_en}: "
                    f"consider dropping one variable from each pair or applying PCA "
                    f"for linear or logistic regression."
                )

        # Paso final: split train/test siempre
        if self.lang == "es":
            pasos.append(
                f"Realizar la división train/test antes de aplicar cualquier "
                f"transformación o imputación para evitar data leakage."
            )
        else:
            pasos.append(
                f"Perform the train/test split before applying any "
                f"transformation or imputation to avoid data leakage."
            )

        if self.lang == "es":
            header = f"Resumen de pasos recomendados ({len(pasos)} en total):"
        else:
            header = f"Recommended steps summary ({len(pasos)} total):"

        pasos_numerados = [
            f"  {i+1}. {p}" for i, p in enumerate(pasos)
        ]
        return header + "\n" + "\n".join(pasos_numerados)

    def quality_score(self) -> dict:
        """Calcula un score de calidad del dataset de 0 a 100.

        Evalúa la salud general del dataset penalizando por distintos
        tipos de problemas detectados por el DataAnalyzer. El score
        es útil para comparar datasets rápidamente o monitorear la
        degradación de calidad en producción.

        El score parte de 100 y resta puntos según estos criterios:

        - Nulos globales: hasta 30 puntos (1.5 puntos por cada 1%)
        - Duplicados: hasta 20 puntos (2 puntos por cada 1%)
        - Columnas constantes: hasta 20 puntos (10 por columna)
        - Alta cardinalidad: hasta 15 puntos (5 por columna)
        - Columnas con más del 20% de nulos: hasta 15 puntos (5 por columna)

        La escala de grados es:
        - A: 90-100 (dataset limpio y listo para modelar)
        - B: 80-89  (calidad buena, problemas menores)
        - C: 70-79  (calidad aceptable, requiere limpieza)
        - D: 60-69  (calidad baja, limpieza importante)
        - F: 0-59   (dataset con problemas graves)

        Returns
        -------
        dict
            Diccionario con las siguientes claves:
            - score: int, puntuación de 0 a 100
            - grade: str, grado de A a F
            - resumen: str, texto descriptivo del resultado
            - penalizaciones: dict, desglose de puntos restados
              por categoría

        Examples
        --------
        >>> import pandas as pd
        >>> from datanarrator import Narrator
        >>> df = pd.read_csv("titanic.csv")
        >>> n = Narrator(df, lang="es")
        >>> resultado = n.quality_score()
        >>> print(resultado["resumen"])
        El dataset obtuvo un score de 72/100 (grado C).
        >>> print(resultado["penalizaciones"])
        {'nulos': 12.1, 'duplicados': 0, 'constantes': 0,
         'cardinalidad': 5, 'cols_nulas': 10}
        """
        ov = self._data["overview"]
        alerts = self._data["alerts"]

        penalizaciones = {}

        # Penalizar por porcentaje global de nulos (máximo 30 puntos).
        # Multiplicamos por 1.5 para que un dataset con 20% de nulos
        # ya pierda 30 puntos completos
        penalizaciones["nulos"] = min(ov["null_pct"] * 1.5, 30)

        # Penalizar por duplicados como porcentaje del total de filas
        # (máximo 20 puntos)
        dup_pct = (
            ov["duplicates"] / ov["rows"] * 100 if ov["rows"] > 0 else 0
        )
        penalizaciones["duplicados"] = min(dup_pct * 2, 20)

        # Penalizar por columnas sin varianza — no aportan información
        # al modelo (10 puntos cada una, máximo 20)
        constantes = sum(
            1 for a in alerts if a["type"] == "constant_column"
        )
        penalizaciones["constantes"] = min(constantes * 10, 20)

        # Penalizar por columnas con alta cardinalidad — difíciles de
        # encodear correctamente (5 puntos cada una, máximo 15)
        cardinalidad = sum(
            1 for a in alerts if a["type"] == "high_cardinality"
        )
        penalizaciones["cardinalidad"] = min(cardinalidad * 5, 15)

        # Penalizar por columnas con más del 20% de nulos — señal de
        # problemas graves de recolección de datos (5 puntos cada una,
        # máximo 15)
        cols_nulas = sum(
            1 for a in alerts if a["type"] == "high_nulls"
        )
        penalizaciones["cols_nulas"] = min(cols_nulas * 5, 15)

        # Calculamos el score final restando todas las penalizaciones
        # y asegurándonos de que no baje de 0
        total_penalizacion = sum(penalizaciones.values())
        score = max(0, round(100 - total_penalizacion))

        # Asignamos grado según la escala estándar de calificaciones
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"

        if self.lang == "es":
            return {
                "score": score,
                "grade": grade,
                "resumen": (
                    f"El dataset obtuvo un score de {score}/100 "
                    f"(grado {grade})."
                ),
                "penalizaciones": penalizaciones,
            }
        return {
            "score": score,
            "grade": grade,
            "resumen": f"Dataset scored {score}/100 (grade {grade}).",
            "penalizaciones": penalizaciones,
        }

    def suggest(self) -> str:
        """Sugiere modelos de ML y pasos de preprocesamiento.

        Basándose en las características del dataset detecta el
        tipo de problema (clasificación, regresión o clustering)
        y recomienda modelos y transformaciones adecuadas.

        Returns
        -------
        str
            Texto con sugerencias de modelos y preprocesamiento.

        Examples
        --------
        >>> n = Narrator(df, lang="es")
        >>> print(n.suggest())
        """
        ov = self._data["overview"]
        alerts = self._data["alerts"]
        numeric = self._data["numeric"]
        categorical = self._data["categorical"]
        out = []

        if self.lang == "es":
            out.append("--- Sugerencias para modelado ---")
            # Detectamos clasificación binaria: columna con valores 0 y 1 únicamente
            binary_cols = [c for c in numeric if set(self.df[c["col"]].dropna().unique()) <= {0, 1}]
            if binary_cols:
                out.append(f"Tipo de problema detectado: clasificacion binaria")
                out.append(f"Variable objetivo probable: {binary_cols[0]['col']}")
                out.append("")
                out.append("Modelos recomendados:")
                out.append("  → Logistic Regression — buen baseline para clasificacion")
                out.append("  → Random Forest — robusto con variables mixtas")
                out.append("  → XGBoost — recomendado si priorizas accuracy")
            elif len(ov["numeric_cols"]) > len(ov["categorical_cols"]):
                out.append("Tipo de problema detectado: posible regresion")
                out.append("")
                out.append("Modelos recomendados:")
                out.append("  → Linear Regression — baseline simple")
                out.append("  → Random Forest Regressor — robusto con outliers")
                out.append("  → Gradient Boosting — para mayor precision")
            else:
                out.append("Tipo de problema: clustering o clasificacion multiclase")
                out.append("")
                out.append("Modelos recomendados:")
                out.append("  → KMeans — si no tienes variable objetivo")
                out.append("  → Random Forest Classifier — si tienes etiquetas")
            high_card = [a["col"] for a in alerts if a["type"] == "high_cardinality"]
            constant = [a["col"] for a in alerts if a["type"] == "constant_column"]
            if high_card:
                out.append("")
                out.append(f"Columnas a excluir o tratar: {', '.join(high_card)}")
                out.append("  → Alta cardinalidad, usa target encoding o eliminalas")
            if constant:
                out.append(f"Columnas a eliminar: {', '.join(constant)}")
                out.append("  → Sin varianza, no aportan informacion al modelo")
            out.append("")
            out.append("Preprocesamiento recomendado:")
            null_cols = [c["col"] for c in numeric if c["nulls"] > 0]
            if null_cols:
                out.append(f"  → Imputar nulos en: {', '.join(null_cols)}")
            skewed = [c["col"] for c in numeric if abs(c["skew"]) > 1]
            if skewed:
                out.append(f"  → Aplicar log transform en: {', '.join(skewed)} (sesgo alto)")
            outlier_cols = [c["col"] for c in numeric if c["outlier_count"] > 0]
            if outlier_cols:
                out.append(f"  → Revisar outliers en: {', '.join(outlier_cols)}")
            if ov["categorical_cols"]:
                out.append(f"  → Encodear variables categoricas: {', '.join(ov['categorical_cols'])}")
        else:
            out.append("--- Modeling suggestions ---")
            # Detectamos clasificación binaria: columna con valores 0 y 1 únicamente
            binary_cols = [c for c in numeric if set(self.df[c["col"]].dropna().unique()) <= {0, 1}]
            if binary_cols:
                out.append(f"Detected problem type: binary classification")
                out.append(f"Likely target variable: {binary_cols[0]['col']}")
                out.append("")
                out.append("Recommended models:")
                out.append("  → Logistic Regression — good baseline")
                out.append("  → Random Forest — robust with mixed features")
                out.append("  → XGBoost — recommended for higher accuracy")
            elif len(ov["numeric_cols"]) > len(ov["categorical_cols"]):
                out.append("Detected problem type: possible regression")
                out.append("")
                out.append("Recommended models:")
                out.append("  → Linear Regression — simple baseline")
                out.append("  → Random Forest Regressor — robust with outliers")
                out.append("  → Gradient Boosting — for higher precision")
            else:
                out.append("Problem type: clustering or multiclass classification")
                out.append("")
                out.append("Recommended models:")
                out.append("  → KMeans — if no target variable")
                out.append("  → Random Forest Classifier — if labels available")
            high_card = [a["col"] for a in alerts if a["type"] == "high_cardinality"]
            constant = [a["col"] for a in alerts if a["type"] == "constant_column"]
            if high_card:
                out.append("")
                out.append(f"Columns to exclude or treat: {', '.join(high_card)}")
                out.append("  → High cardinality, use target encoding or drop them")
            if constant:
                out.append(f"Columns to drop: {', '.join(constant)}")
                out.append("  → No variance, no information for the model")
            out.append("")
            out.append("Recommended preprocessing:")
            null_cols = [c["col"] for c in numeric if c["nulls"] > 0]
            if null_cols:
                out.append(f"  → Impute nulls in: {', '.join(null_cols)}")
            skewed = [c["col"] for c in numeric if abs(c["skew"]) > 1]
            if skewed:
                out.append(f"  → Apply log transform to: {', '.join(skewed)} (high skew)")
            outlier_cols = [c["col"] for c in numeric if c["outlier_count"] > 0]
            if outlier_cols:
                out.append(f"  → Review outliers in: {', '.join(outlier_cols)}")
            if ov["categorical_cols"]:
                out.append(f"  → Encode categorical columns: {', '.join(ov['categorical_cols'])}")
        return "\n".join(out)

    # ------------------------------------------------------------------
    # Secciones internas
    # ------------------------------------------------------------------

    def _translate_alert(self, alert: dict) -> tuple:
        """Traduce el mensaje y sugerencia de una alerta al idioma activo.

        Centraliza la lógica de traducción de alertas que antes estaba
        duplicada en alerts_only() y _section_alerts(). Usa diccionarios
        de plantillas por tipo de alerta en lugar de str.replace() en
        cadena, lo que hace la traducción robusta a cambios futuros en
        los mensajes en español.

        Parameters
        ----------
        alert : dict
            Diccionario de alerta generado por DataAnalyzer._alerts().
            Debe contener al menos las claves 'type', 'message' y
            'suggestion'. Las alertas de tipo 'high_nulls',
            'high_cardinality' y 'constant_column' también requieren
            la clave 'col'.

        Returns
        -------
        tuple
            Par (mensaje, sugerencia) ya traducidos al idioma activo.
            Si el idioma es 'es', retorna los valores originales sin
            modificación. Si es 'en', aplica la plantilla correspondiente
            al tipo de alerta.

        Examples
        --------
        >>> n = Narrator(df, lang="en")
        >>> alert = {"type": "high_nulls", "col": "age",
        ...          "message": "'age' tiene 30.0% de valores nulos.",
        ...          "suggestion": "Considera imputar o eliminar esta columna."}
        >>> msg, sug = n._translate_alert(alert)
        >>> print(msg)
        'age' has 30.0% null values.
        """
        # Si el idioma es español devolvemos los valores tal como vienen
        # del analyzer sin ninguna transformación
        if self.lang == "es":
            return alert["message"], alert["suggestion"]

        # Plantillas de mensajes en inglés por tipo de alerta.
        # Cada lambda extrae los datos necesarios directamente del
        # diccionario de alerta, evitando depender de la redacción
        # exacta del mensaje en español
        MENSAJES = {
            "duplicates": lambda a: (
                f"{a['message'].split()[0]} duplicate rows detected."
            ),
            "high_nulls": lambda a: (
                f"'{a['col']}' has {a['message'].split()[2]} null values."
            ),
            "high_cardinality": lambda a: (
                f"'{a['col']}' has {a['message'].split()[2]} unique values."
            ),
            "constant_column": lambda a: (
                f"'{a['col']}' has only one unique value."
            ),
        }

        # Plantillas de sugerencias en inglés por tipo de alerta
        SUGERENCIAS = {
            "duplicates": "Consider dropping them before modeling.",
            "high_nulls": "Consider imputing or dropping this column.",
            "high_cardinality": (
                "Avoid direct label encoding. Consider target encoding."
            ),
            "constant_column": (
                "This column has no information. Consider dropping it."
            ),
        }

        # Aplicamos la plantilla correspondiente al tipo de alerta.
        # Si el tipo no está en el diccionario, usamos el mensaje
        # original como fallback para no perder información
        tipo = alert["type"]
        msg = MENSAJES.get(tipo, lambda a: a["message"])(alert)
        sug = SUGERENCIAS.get(tipo, alert["suggestion"])
        return msg, sug

    def _section_overview(self) -> str:
        # Obtenemos el resumen general calculado por el analyzer
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
        # Si no hay columnas numéricas no generamos esta sección
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
        # Si no hay columnas categóricas no generamos esta sección
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
        # Solo mostramos correlaciones si el analyzer encontró alguna >= 0.5
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
	# Reutilizamos _translate_alert() para no duplicar
        # la lógica de traducción entre métodos públicos
        for a in alerts:
            msg, sug = self._translate_alert(a)
            lines.append(f"  ⚠  {msg}")
            lines.append(f"     → {sug}")
