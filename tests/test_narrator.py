"""
Tests para la librería datanarrator.

Cubre todos los métodos públicos de la clase Narrator:
describe(), executive_summary(), alerts_only(), export(),
compare() y suggest(). También prueba el DataAnalyzer interno.

Para correr los tests:
    pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from datanarrator import Narrator
from datanarrator.analyzer import DataAnalyzer


# ------------------------------------------------------------------
# Fixtures — datasets reutilizables en todos los tests
# ------------------------------------------------------------------

@pytest.fixture
def df_basic():
    """Dataset básico con columnas numéricas y categóricas."""
    return pd.DataFrame({
        "edad": [25, 30, 35, 40, 45, 200],
        "salario": [30000, 45000, 50000, 60000, 70000, 80000],
        "ciudad": ["cdmx", "cdmx", "monterrey", "guadalajara", "cdmx", "monterrey"],
        "activo": ["si", "no", "si", "si", "no", "si"],
    })


@pytest.fixture
def df_with_nulls():
    """Dataset con valores nulos."""
    return pd.DataFrame({
        "edad": [25, None, 35, None, 45],
        "nombre": ["Ana", "Luis", None, "María", "Pedro"],
        "salario": [30000, 45000, 50000, None, 70000],
    })


@pytest.fixture
def df_numeric_only():
    """Dataset solo con columnas numéricas."""
    return pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, 50],
        "c": [100, 200, 300, 400, 500],
    })


@pytest.fixture
def df_categorical_only():
    """Dataset solo con columnas categóricas."""
    return pd.DataFrame({
        "color": ["rojo", "azul", "rojo", "verde", "azul"],
        "talla": ["s", "m", "l", "m", "s"],
    })


# ------------------------------------------------------------------
# Tests de inicialización
# ------------------------------------------------------------------

def test_narrator_init_valid(df_basic):
    n = Narrator(df_basic)
    assert n is not None


def test_narrator_default_lang(df_basic):
    n = Narrator(df_basic)
    assert n.lang == "es"


def test_narrator_english_lang(df_basic):
    n = Narrator(df_basic, lang="en")
    assert n.lang == "en"


def test_narrator_invalid_lang(df_basic):
    with pytest.raises(ValueError):
        Narrator(df_basic, lang="fr")


def test_narrator_invalid_input():
    with pytest.raises(TypeError):
        Narrator("no soy un dataframe")


def test_narrator_empty_dataframe():
    with pytest.raises(ValueError):
        Narrator(pd.DataFrame())


# ------------------------------------------------------------------
# Tests de describe()
# ------------------------------------------------------------------

def test_describe_returns_string(df_basic):
    n = Narrator(df_basic)
    result = n.describe()
    assert isinstance(result, str)


def test_describe_not_empty(df_basic):
    n = Narrator(df_basic)
    result = n.describe()
    assert len(result) > 0


def test_describe_contains_resumen(df_basic):
    n = Narrator(df_basic, lang="es")
    result = n.describe()
    assert "Resumen general" in result


def test_describe_english(df_basic):
    n = Narrator(df_basic, lang="en")
    result = n.describe()
    assert "Overview" in result


def test_describe_numeric_only(df_numeric_only):
    n = Narrator(df_numeric_only)
    result = n.describe()
    assert isinstance(result, str)
    assert len(result) > 0


def test_describe_categorical_only(df_categorical_only):
    n = Narrator(df_categorical_only)
    result = n.describe()
    assert isinstance(result, str)
    assert len(result) > 0


# ------------------------------------------------------------------
# Tests de executive_summary()
# ------------------------------------------------------------------

def test_executive_summary_returns_string(df_basic):
    n = Narrator(df_basic)
    result = n.executive_summary()
    assert isinstance(result, str)


def test_executive_summary_not_empty(df_basic):
    n = Narrator(df_basic)
    result = n.executive_summary()
    assert len(result) > 0


def test_executive_summary_english(df_basic):
    n = Narrator(df_basic, lang="en")
    result = n.executive_summary()
    assert isinstance(result, str)


# ------------------------------------------------------------------
# Tests de alerts_only()
# ------------------------------------------------------------------

def test_alerts_only_returns_string(df_basic):
    n = Narrator(df_basic)
    result = n.alerts_only()
    assert isinstance(result, str)


def test_alerts_with_nulls(df_with_nulls):
    n = Narrator(df_with_nulls)
    result = n.alerts_only()
    assert isinstance(result, str)


def test_no_alerts_clean_dataset(df_numeric_only):
    n = Narrator(df_numeric_only)
    result = n.alerts_only()
    assert "No se detectaron alertas" in result


# ------------------------------------------------------------------
# Tests de export()
# ------------------------------------------------------------------

def test_export_creates_file(df_basic, tmp_path):
    n = Narrator(df_basic)
    filepath = tmp_path / "reporte.txt"
    n.export(str(filepath))
    assert filepath.exists()


def test_export_file_not_empty(df_basic, tmp_path):
    n = Narrator(df_basic)
    filepath = tmp_path / "reporte.md"
    n.export(str(filepath))
    assert filepath.stat().st_size > 0


# ------------------------------------------------------------------
# Tests del analyzer
# ------------------------------------------------------------------

def test_analyzer_overview_rows(df_basic):
    analyzer = DataAnalyzer(df_basic)
    result = analyzer.analyze()
    assert result["overview"]["rows"] == 6


def test_analyzer_overview_cols(df_basic):
    analyzer = DataAnalyzer(df_basic)
    result = analyzer.analyze()
    assert result["overview"]["cols"] == 4


def test_analyzer_detects_nulls(df_with_nulls):
    analyzer = DataAnalyzer(df_with_nulls)
    result = analyzer.analyze()
    assert result["overview"]["total_nulls"] > 0


def test_analyzer_correlations(df_numeric_only):
    analyzer = DataAnalyzer(df_numeric_only)
    result = analyzer.analyze()
    assert isinstance(result["correlations"], list)


def test_analyzer_alerts_high_nulls(df_with_nulls):
    analyzer = DataAnalyzer(df_with_nulls)
    result = analyzer.analyze()
    alert_types = [a["type"] for a in result["alerts"]]
    assert "high_nulls" in alert_types


# ------------------------------------------------------------------
# Tests de compare()
# ------------------------------------------------------------------

def test_compare_returns_string(df_basic):
    df2 = df_basic.copy()
    df2["edad"] = df2["edad"] * 2
    n = Narrator(df_basic)
    result = n.compare(df2)
    assert isinstance(result, str)

def test_compare_invalid_input(df_basic):
    n = Narrator(df_basic)
    with pytest.raises(TypeError):
        n.compare("no soy un dataframe")

def test_compare_empty_dataframe(df_basic):
    n = Narrator(df_basic)
    with pytest.raises(ValueError):
        n.compare(pd.DataFrame())

def test_compare_english(df_basic):
    df2 = df_basic.copy()
    n = Narrator(df_basic, lang="en")
    result = n.compare(df2)
    assert "comparison" in result

# ------------------------------------------------------------------
# Tests de suggest()
# ------------------------------------------------------------------

def test_suggest_returns_string(df_basic):
    n = Narrator(df_basic)
    result = n.suggest()
    assert isinstance(result, str)

def test_suggest_not_empty(df_basic):
    n = Narrator(df_basic)
    result = n.suggest()
    assert len(result) > 0

def test_suggest_english(df_basic):
    n = Narrator(df_basic, lang="en")
    result = n.suggest()
    assert "suggestions" in result.lower()

def test_suggest_numeric_only(df_numeric_only):
    n = Narrator(df_numeric_only)
    result = n.suggest()
    assert isinstance(result, str)

# -----------------------------------------------------------------------
# Tests de quality_score()
# -----------------------------------------------------------------------

def test_quality_score_returns_dict(df_basic):
    """Verifica que quality_score() retorna un diccionario."""
    n = Narrator(df_basic, lang="es")
    result = n.quality_score()
    assert isinstance(result, dict)


def test_quality_score_keys(df_basic):
    """Verifica que el diccionario tiene todas las claves esperadas."""
    n = Narrator(df_basic, lang="es")
    result = n.quality_score()
    assert "score" in result
    assert "grade" in result
    assert "resumen" in result
    assert "penalizaciones" in result


def test_quality_score_range(df_basic):
    """Verifica que el score está entre 0 y 100."""
    n = Narrator(df_basic, lang="es")
    result = n.quality_score()
    assert 0 <= result["score"] <= 100


def test_quality_score_grade_valid(df_basic):
    """Verifica que el grado es uno de los valores válidos."""
    n = Narrator(df_basic, lang="es")
    result = n.quality_score()
    assert result["grade"] in ("A", "B", "C", "D", "F")


def test_quality_score_clean_dataset(df_numeric_only):
    """Un dataset limpio sin nulos ni duplicados debe tener score alto."""
    n = Narrator(df_numeric_only, lang="es")
    result = n.quality_score()
    assert result["score"] >= 80


def test_quality_score_dirty_dataset(df_with_nulls, df_numeric_only):
    """Un dataset con muchos nulos debe tener score más bajo."""
    n_clean = Narrator(df_numeric_only, lang="es")
    n_dirty = Narrator(df_with_nulls, lang="es")
    assert n_dirty.quality_score()["score"] <= n_clean.quality_score()["score"]


def test_quality_score_english(df_basic):
    """Verifica que el resumen en inglés no contiene palabras en español."""
    n = Narrator(df_basic, lang="en")
    result = n.quality_score()
    assert "scored" in result["resumen"]
    assert "grade" in result["resumen"]


def test_quality_score_spanish(df_basic):
    """Verifica que el resumen en español contiene las palabras correctas."""
    n = Narrator(df_basic, lang="es")
    result = n.quality_score()
    assert "score" in result["resumen"]
    assert "grado" in result["resumen"]


def test_quality_score_penalizaciones_keys(df_basic):
    """Verifica que el desglose de penalizaciones tiene todas las claves."""
    n = Narrator(df_basic, lang="es")
    pen = n.quality_score()["penalizaciones"]
    assert "nulos" in pen
    assert "duplicados" in pen
    assert "constantes" in pen
    assert "cardinalidad" in pen
    assert "cols_nulas" in pen


# -----------------------------------------------------------------------
# Tests de _translate_alert()
# -----------------------------------------------------------------------

def test_translate_alert_spanish_returns_original(df_basic):
    """En español debe retornar el mensaje original sin cambios."""
    n = Narrator(df_basic, lang="es")
    alert = {
        "type": "high_nulls",
        "col": "edad",
        "message": "'edad' tiene 30.0% de valores nulos.",
        "suggestion": "Considera imputar o eliminar esta columna.",
    }
    msg, sug = n._translate_alert(alert)
    assert msg == alert["message"]
    assert sug == alert["suggestion"]


def test_translate_alert_english_high_nulls(df_basic):
    """Verifica traducción correcta de alerta high_nulls al inglés."""
    n = Narrator(df_basic, lang="en")
    alert = {
        "type": "high_nulls",
        "col": "edad",
        "message": "'edad' tiene 30.0% de valores nulos.",
        "suggestion": "Considera imputar o eliminar esta columna.",
    }
    msg, sug = n._translate_alert(alert)
    assert "edad" in msg
    assert "null values" in msg
    assert "imputing" in sug


def test_translate_alert_english_duplicates(df_basic):
    """Verifica traducción correcta de alerta duplicates al inglés."""
    n = Narrator(df_basic, lang="en")
    alert = {
        "type": "duplicates",
        "message": "5 registros duplicados detectados.",
        "suggestion": "Considera eliminarlos antes de modelar.",
    }
    msg, sug = n._translate_alert(alert)
    assert "duplicate" in msg
    assert "modeling" in sug


def test_translate_alert_english_constant_column(df_basic):
    """Verifica traducción correcta de alerta constant_column al inglés."""
    n = Narrator(df_basic, lang="en")
    alert = {
        "type": "constant_column",
        "col": "pais",
        "message": "'pais' tiene un solo valor único.",
        "suggestion": "Esta columna no aporta información. Considera eliminarla.",
    }
    msg, sug = n._translate_alert(alert)
    assert "pais" in msg
    assert "dropping" in sug


def test_translate_alert_english_high_cardinality(df_basic):
    """Verifica traducción correcta de alerta high_cardinality al inglés."""
    n = Narrator(df_basic, lang="en")
    alert = {
        "type": "high_cardinality",
        "col": "ciudad",
        "message": "'nombre' tiene 500 valores únicos.",
        "suggestion": "Evita label encoding directo. Considera target encoding.",
    }
    msg, sug = n._translate_alert(alert)
    assert "ciudad" in msg
    assert "unique values" in msg
    assert "target encoding" in sug

# -----------------------------------------------------------------------
# Tests de narrative()
# -----------------------------------------------------------------------

def test_narrative_returns_string(df_basic):
    """Verifica que narrative() retorna un string."""
    n = Narrator(df_basic, lang="es")
    assert isinstance(n.narrative(), str)


def test_narrative_not_empty(df_basic):
    """Verifica que narrative() no retorna un string vacío."""
    n = Narrator(df_basic, lang="es")
    assert len(n.narrative()) > 0


def test_narrative_english(df_basic):
    """Verifica que narrative() en inglés no contiene palabras en español."""
    n = Narrator(df_basic, lang="en")
    result = n.narrative()
    assert "registros" not in result
    assert "columnas" not in result


def test_narrative_spanish(df_basic):
    """Verifica que narrative() en español contiene palabras clave."""
    n = Narrator(df_basic, lang="es")
    result = n.narrative()
    assert "dataset" in result.lower()


def test_narrative_numeric_only(df_numeric_only):
    """Verifica que narrative() funciona con dataset solo numérico."""
    n = Narrator(df_numeric_only, lang="es")
    assert isinstance(n.narrative(), str)


def test_narrative_with_nulls(df_with_nulls):
    """Verifica que narrative() menciona nulos cuando los hay."""
    n = Narrator(df_with_nulls, lang="es")
    result = n.narrative()
    assert "nulos" in result or "null" in result

# -----------------------------------------------------------------------
# Tests de narrate()
# -----------------------------------------------------------------------

def test_narrate_returns_string(df_basic):
    """Verifica que narrate() retorna un string."""
    n = Narrator(df_basic, lang="es")
    assert isinstance(n.narrate(audience="technical"), str)


def test_narrate_invalid_audience(df_basic):
    """Verifica que una audiencia inválida lanza ValueError."""
    n = Narrator(df_basic, lang="es")
    with pytest.raises(ValueError):
        n.narrate(audience="invalid")


def test_narrate_default_audience(df_basic):
    """Verifica que la audiencia por defecto es technical."""
    n = Narrator(df_basic, lang="es")
    result_default = n.narrate()
    result_technical = n.narrate(audience="technical")
    assert result_default == result_technical


def test_narrate_executive_not_empty(df_basic):
    """Verifica que la narrativa ejecutiva no está vacía."""
    n = Narrator(df_basic, lang="es")
    result = n.narrate(audience="executive")
    assert len(result) > 0


def test_narrate_technical_not_empty(df_basic):
    """Verifica que la narrativa técnica no está vacía."""
    n = Narrator(df_basic, lang="es")
    result = n.narrate(audience="technical")
    assert len(result) > 0


def test_narrate_non_technical_not_empty(df_basic):
    """Verifica que la narrativa no técnica no está vacía."""
    n = Narrator(df_basic, lang="es")
    result = n.narrate(audience="non-technical")
    assert len(result) > 0


def test_narrate_executive_contains_header(df_basic):
    """Verifica que la narrativa ejecutiva tiene su header."""
    n = Narrator(df_basic, lang="es")
    result = n.narrate(audience="executive")
    assert "Resumen Ejecutivo" in result


def test_narrate_non_technical_contains_header(df_basic):
    """Verifica que la narrativa no técnica tiene su header."""
    n = Narrator(df_basic, lang="es")
    result = n.narrate(audience="non-technical")
    assert "Explicación Simple" in result


def test_narrate_audiences_are_different(df_basic):
    """Verifica que las tres audiencias generan textos distintos."""
    n = Narrator(df_basic, lang="es")
    executive = n.narrate(audience="executive")
    technical = n.narrate(audience="technical")
    non_technical = n.narrate(audience="non-technical")
    assert executive != technical
    assert technical != non_technical
    assert executive != non_technical


def test_narrate_english_executive(df_basic):
    """Verifica que la narrativa ejecutiva en inglés funciona."""
    n = Narrator(df_basic, lang="en")
    result = n.narrate(audience="executive")
    assert "Executive Summary" in result


def test_narrate_english_non_technical(df_basic):
    """Verifica que la narrativa no técnica en inglés funciona."""
    n = Narrator(df_basic, lang="en")
    result = n.narrate(audience="non-technical")
    assert "Simple Explanation" in result


def test_narrate_with_nulls(df_with_nulls):
    """Verifica que narrate() funciona con dataset con nulos."""
    n = Narrator(df_with_nulls, lang="es")
    for audience in ["executive", "technical", "non-technical"]:
        assert isinstance(n.narrate(audience=audience), str)


def test_narrate_numeric_only(df_numeric_only):
    """Verifica que narrate() funciona con dataset solo numérico."""
    n = Narrator(df_numeric_only, lang="es")
    for audience in ["executive", "technical", "non-technical"]:
        assert isinstance(n.narrate(audience=audience), str)
