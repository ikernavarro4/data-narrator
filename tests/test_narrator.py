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
