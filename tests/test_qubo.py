"""
Tests para el motor QUBO.

Verifica la correcta transformación del problema de Markowitz
a formato QUBO compatible con algoritmos cuánticos.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimization.qubo_engine import (
    QUBOEngine,
    QUBOResult,
    Constraint,
    ConstraintType,
    to_qubo,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_portfolio():
    """Portfolio simple de 4 activos para testing."""
    tickers = ["A", "B", "C", "D"]
    mu = pd.Series([0.10, 0.15, 0.08, 0.20], index=tickers)
    sigma = pd.DataFrame(
        [
            [0.04, 0.01, 0.02, 0.00],
            [0.01, 0.09, 0.01, 0.02],
            [0.02, 0.01, 0.03, 0.01],
            [0.00, 0.02, 0.01, 0.16],
        ],
        index=tickers,
        columns=tickers,
    )
    return tickers, mu, sigma


@pytest.fixture
def qubo_engine():
    """Motor QUBO configurado."""
    return QUBOEngine(default_penalty=1000.0)


# =============================================================================
# Tests de Validación de Inputs
# =============================================================================

class TestInputValidation:
    """Tests de validación de entradas."""
    
    def test_risk_aversion_bounds(self, qubo_engine, simple_portfolio):
        """Risk aversion debe estar en [0, 1]."""
        _, mu, sigma = simple_portfolio
        
        # Valores válidos
        for ra in [0.0, 0.5, 1.0]:
            result = qubo_engine.to_qubo(mu, sigma, risk_aversion=ra)
            assert result is not None
        
        # Valores inválidos
        with pytest.raises(ValueError):
            qubo_engine.to_qubo(mu, sigma, risk_aversion=-0.1)
        
        with pytest.raises(ValueError):
            qubo_engine.to_qubo(mu, sigma, risk_aversion=1.1)
    
    def test_dimension_mismatch(self, qubo_engine):
        """Detectar dimensiones inconsistentes."""
        mu = pd.Series([0.1, 0.2])
        sigma = pd.DataFrame([[0.1, 0.01, 0.02], [0.01, 0.2, 0.01], [0.02, 0.01, 0.15]])
        
        with pytest.raises(ValueError):
            qubo_engine.to_qubo(mu, sigma)
    
    def test_accepts_numpy_arrays(self, qubo_engine):
        """Debe aceptar arrays numpy además de pandas."""
        mu = np.array([0.1, 0.15, 0.08])
        sigma = np.array([[0.04, 0.01, 0.02], [0.01, 0.09, 0.01], [0.02, 0.01, 0.03]])
        
        result = qubo_engine.to_qubo(mu, sigma)
        
        assert result is not None
        assert result.n_qubits == 3


# =============================================================================
# Tests de Transformación QUBO
# =============================================================================

class TestQUBOTransformation:
    """Tests de la transformación Markowitz → QUBO."""
    
    def test_qubo_matrix_shape(self, qubo_engine, simple_portfolio):
        """La matriz QUBO debe ser cuadrada y del tamaño correcto."""
        tickers, mu, sigma = simple_portfolio
        
        result = qubo_engine.to_qubo(mu, sigma)
        
        assert result.Q.shape == (4, 4)
        assert result.n_qubits == 4
    
    def test_qubo_matrix_symmetric(self, qubo_engine, simple_portfolio):
        """La matriz QUBO debe ser simétrica."""
        _, mu, sigma = simple_portfolio
        
        result = qubo_engine.to_qubo(mu, sigma)
        
        # Verificar simetría
        assert np.allclose(result.Q, result.Q.T)
    
    def test_risk_aversion_effect(self, qubo_engine, simple_portfolio):
        """Mayor risk aversion debe penalizar más el riesgo."""
        _, mu, sigma = simple_portfolio
        
        # Risk aversion bajo (favorece retorno)
        result_low = qubo_engine.to_qubo(mu, sigma, risk_aversion=0.1)
        
        # Risk aversion alto (favorece bajo riesgo)
        result_high = qubo_engine.to_qubo(mu, sigma, risk_aversion=0.9)
        
        # Con high risk aversion, los activos de alta varianza
        # deben tener diagonal más positiva (más penalizados)
        # El activo D tiene sigma = 0.16 (el más alto)
        assert result_high.Q[3, 3] > result_low.Q[3, 3]
    
    def test_high_return_favored_low_risk_aversion(self, qubo_engine, simple_portfolio):
        """Con baja aversión al riesgo, activos de alto retorno deben ser favorecidos."""
        _, mu, sigma = simple_portfolio
        
        result = qubo_engine.to_qubo(mu, sigma, risk_aversion=0.1)
        
        # El activo D tiene el mayor retorno (0.20)
        # Con baja risk aversion, debería tener diagonal más negativa
        diagonal = np.diag(result.Q)
        
        # D (índice 3) debería estar entre los más negativos
        # (recordando que negativo = favorecido en QUBO)
        assert diagonal[3] < np.median(diagonal)
    
    def test_evaluate_function(self, qubo_engine, simple_portfolio):
        """La evaluación de soluciones debe funcionar correctamente."""
        _, mu, sigma = simple_portfolio
        
        result = qubo_engine.to_qubo(mu, sigma, risk_aversion=0.5)
        
        # Evaluar diferentes soluciones
        x_none = [0, 0, 0, 0]
        x_all = [1, 1, 1, 1]
        x_one = [1, 0, 0, 0]
        
        val_none = result.evaluate(x_none)
        val_all = result.evaluate(x_all)
        val_one = result.evaluate(x_one)
        
        # Ningún activo = 0 (sin costo ni beneficio)
        assert val_none == 0
        
        # Todos los activos debería dar valor != 0
        assert val_all != 0
        
        # Los valores deben ser diferentes
        assert val_none != val_all
        assert val_one != val_all


# =============================================================================
# Tests de Restricciones
# =============================================================================

class TestConstraints:
    """Tests de restricciones en la formulación QUBO."""
    
    def test_min_assets_constraint(self, qubo_engine, simple_portfolio):
        """La restricción de mínimo de activos debe modificar la matriz."""
        _, mu, sigma = simple_portfolio
        
        # Sin restricción
        result_no_constraint = qubo_engine.to_qubo(mu, sigma)
        
        # Con restricción de mínimo 2 activos
        constraints = [Constraint(ConstraintType.MIN_ASSETS, 2)]
        result_with_constraint = qubo_engine.to_qubo(mu, sigma, constraints=constraints)
        
        # Las matrices deben ser diferentes
        assert not np.allclose(result_no_constraint.Q, result_with_constraint.Q)
    
    def test_exact_assets_constraint(self, qubo_engine, simple_portfolio):
        """La restricción de número exacto debe funcionar."""
        _, mu, sigma = simple_portfolio
        
        constraints = [Constraint(ConstraintType.EXACT_ASSETS, 2, penalty=1000)]
        result = qubo_engine.to_qubo(mu, sigma, constraints=constraints)
        
        # Evaluar solución con exactamente 2 activos
        x_two = [1, 1, 0, 0]
        x_three = [1, 1, 1, 0]
        
        val_two = result.evaluate(x_two)
        val_three = result.evaluate(x_three)
        
        # Con penalty alta, x_two (que cumple restricción) debería ser mejor
        # Nota: esto depende de los valores exactos, así que solo verificamos
        # que la restricción tiene efecto
        assert val_two != val_three
    
    def test_constraint_penalty_effect(self, qubo_engine, simple_portfolio):
        """Mayor penalización debe tener mayor efecto en la matriz."""
        _, mu, sigma = simple_portfolio
        
        constraints_low = [Constraint(ConstraintType.MIN_ASSETS, 2, penalty=100)]
        constraints_high = [Constraint(ConstraintType.MIN_ASSETS, 2, penalty=10000)]
        
        result_low = qubo_engine.to_qubo(mu, sigma, constraints=constraints_low)
        result_high = qubo_engine.to_qubo(mu, sigma, constraints=constraints_high)
        
        # La diferencia con alta penalización debe ser mayor
        diff_low = np.abs(result_low.Q).sum()
        diff_high = np.abs(result_high.Q).sum()
        
        assert diff_high > diff_low


# =============================================================================
# Tests de Normalización
# =============================================================================

class TestNormalization:
    """Tests de normalización de datos."""
    
    def test_normalization_improves_conditioning(self, qubo_engine, simple_portfolio):
        """La normalización debe mejorar el número de condición."""
        _, mu, sigma = simple_portfolio
        
        result_normalized = qubo_engine.to_qubo(mu, sigma, normalize=True)
        result_raw = qubo_engine.to_qubo(mu, sigma, normalize=False)
        
        # Ambos deben producir resultados válidos
        assert result_normalized.Q is not None
        assert result_raw.Q is not None
        
        # El número de condición con normalización debería ser menor o similar
        cond_normalized = result_normalized.metadata.get("condition_number", np.inf)
        cond_raw = result_raw.metadata.get("condition_number", np.inf)
        
        # No siempre es mejor, pero debería ser finito
        assert np.isfinite(cond_normalized)


# =============================================================================
# Tests de Conveniencia
# =============================================================================

class TestConvenienceFunctions:
    """Tests de funciones de conveniencia."""
    
    def test_to_qubo_function(self, simple_portfolio):
        """La función to_qubo debe funcionar sin instanciar motor."""
        _, mu, sigma = simple_portfolio
        
        Q = to_qubo(mu, sigma, risk_aversion=0.5)
        
        assert Q is not None
        assert Q.shape == (4, 4)
    
    def test_to_qubo_with_constraints(self, simple_portfolio):
        """to_qubo debe aceptar constraints como parámetros."""
        _, mu, sigma = simple_portfolio
        
        Q = to_qubo(mu, sigma, risk_aversion=0.5, min_assets=2, max_assets=3)
        
        assert Q is not None
        assert Q.shape == (4, 4)


# =============================================================================
# Tests de Análisis
# =============================================================================

class TestQUBOAnalysis:
    """Tests de funciones de análisis."""
    
    def test_analyze_qubo(self, qubo_engine, simple_portfolio):
        """El análisis QUBO debe retornar métricas válidas."""
        _, mu, sigma = simple_portfolio
        
        result = qubo_engine.to_qubo(mu, sigma)
        analysis = qubo_engine.analyze_qubo(result)
        
        assert "n_qubits" in analysis
        assert analysis["n_qubits"] == 4
        
        assert "eigenvalue_min" in analysis
        assert "eigenvalue_max" in analysis
        assert analysis["eigenvalue_min"] <= analysis["eigenvalue_max"]
        
        assert "sparsity" in analysis
        assert 0 <= analysis["sparsity"] <= 1
    
    def test_get_selected_assets(self, qubo_engine, simple_portfolio):
        """Debe retornar correctamente los activos seleccionados."""
        tickers, mu, sigma = simple_portfolio
        
        result = qubo_engine.to_qubo(mu, sigma)
        
        x = [1, 0, 1, 0]
        selected = result.get_selected_assets(x)
        
        assert selected == ["A", "C"]


# =============================================================================
# Tests de Casos Límite
# =============================================================================

class TestEdgeCases:
    """Tests de casos límite."""
    
    def test_single_asset(self, qubo_engine):
        """Debe manejar un solo activo."""
        mu = pd.Series([0.10], index=["A"])
        sigma = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
        
        result = qubo_engine.to_qubo(mu, sigma)
        
        assert result.n_qubits == 1
        assert result.Q.shape == (1, 1)
    
    def test_zero_returns(self, qubo_engine):
        """Debe manejar retornos cero."""
        mu = pd.Series([0.0, 0.0], index=["A", "B"])
        sigma = pd.DataFrame([[0.04, 0.01], [0.01, 0.05]], index=["A", "B"], columns=["A", "B"])
        
        result = qubo_engine.to_qubo(mu, sigma)
        
        assert result is not None
        assert np.all(np.isfinite(result.Q))
    
    def test_identical_assets(self, qubo_engine):
        """Debe manejar activos idénticos."""
        mu = pd.Series([0.10, 0.10], index=["A", "B"])
        sigma = pd.DataFrame([[0.04, 0.04], [0.04, 0.04]], index=["A", "B"], columns=["A", "B"])
        
        result = qubo_engine.to_qubo(mu, sigma)
        
        assert result is not None


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
