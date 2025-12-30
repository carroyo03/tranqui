"""
Tests para comparación de solvers.

Verifica que QAOA produce soluciones competitivas
respecto al baseline clásico.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimization.qubo_engine import QUBOEngine, Constraint, ConstraintType
from src.optimization.quantum_solver import QuantumSolver, QAOAConfig
from src.optimization.classical_solver import ClassicalSolver, ClassicalMethod, compare_solvers


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def small_problem():
    """Problema pequeño (4 activos) para tests rápidos."""
    np.random.seed(42)
    n = 4
    tickers = [f"Asset_{i}" for i in range(n)]
    
    mu = pd.Series(np.random.uniform(0.05, 0.15, n), index=tickers)
    
    A = np.random.randn(n, n) * 0.1
    sigma_arr = A @ A.T + np.eye(n) * 0.03
    sigma = pd.DataFrame(sigma_arr, index=tickers, columns=tickers)
    
    engine = QUBOEngine()
    qubo = engine.to_qubo(mu, sigma, risk_aversion=0.5)
    
    return qubo


@pytest.fixture
def medium_problem():
    """Problema mediano (8 activos)."""
    np.random.seed(42)
    n = 8
    tickers = [f"Asset_{i}" for i in range(n)]
    
    mu = pd.Series(np.random.uniform(0.05, 0.15, n), index=tickers)
    
    A = np.random.randn(n, n) * 0.1
    sigma_arr = A @ A.T + np.eye(n) * 0.03
    sigma = pd.DataFrame(sigma_arr, index=tickers, columns=tickers)
    
    engine = QUBOEngine()
    qubo = engine.to_qubo(mu, sigma, risk_aversion=0.5)
    
    return qubo


# =============================================================================
# Tests de Solvers Individuales
# =============================================================================

class TestQuantumSolver:
    """Tests para el solver cuántico (QAOA)."""
    
    def test_qaoa_returns_valid_solution(self, small_problem):
        """QAOA debe retornar una solución binaria válida."""
        config = QAOAConfig(reps=1, max_iter=50)
        solver = QuantumSolver(config)
        
        result = solver.solve(small_problem, verbose=False)
        
        # Verificar que es binaria
        assert all(x in [0, 1] for x in result.selection)
        
        # Verificar longitud correcta
        assert len(result.selection) == small_problem.n_qubits
        
        # Verificar que hay activos seleccionados
        assert result.n_selected > 0
    
    def test_qaoa_reps_effect(self, small_problem):
        """Más capas QAOA deberían mejorar o mantener la solución."""
        results = []
        
        for reps in [1, 2]:
            config = QAOAConfig(reps=reps, max_iter=50)
            solver = QuantumSolver(config)
            result = solver.solve(small_problem, verbose=False)
            results.append(result.objective_value)
        
        # reps=2 debería ser igual o mejor que reps=1
        # (menor valor objetivo = mejor)
        assert results[1] <= results[0] * 1.1  # 10% tolerance
    
    def test_qaoa_execution_time_recorded(self, small_problem):
        """El tiempo de ejecución debe ser registrado."""
        config = QAOAConfig(reps=1, max_iter=50)
        solver = QuantumSolver(config)
        
        result = solver.solve(small_problem, verbose=False)
        
        assert result.execution_time > 0
        assert result.execution_time < 60  # No debería tardar más de 1 minuto


class TestClassicalSolver:
    """Tests para los solvers clásicos."""
    
    def test_brute_force_finds_optimal(self, small_problem):
        """Fuerza bruta debe encontrar el óptimo global."""
        solver = ClassicalSolver(ClassicalMethod.BRUTE_FORCE)
        result = solver.solve(small_problem, verbose=False)
        
        # Verificar que es óptimo
        assert result.is_optimal
        
        # Verificar que evaluó todas las soluciones
        assert result.n_evaluations == 2 ** small_problem.n_qubits
    
    def test_greedy_produces_solution(self, small_problem):
        """Greedy debe producir una solución (no necesariamente óptima)."""
        solver = ClassicalSolver(ClassicalMethod.GREEDY)
        result = solver.solve(small_problem, verbose=False)
        
        # No es óptimo garantizado
        assert not result.is_optimal
        
        # Pero debe producir una solución válida
        assert len(result.selection) == small_problem.n_qubits
        assert all(x in [0, 1] for x in result.selection)
    
    def test_simulated_annealing_produces_solution(self, small_problem):
        """SA debe producir una solución."""
        solver = ClassicalSolver(ClassicalMethod.SIMULATED_ANNEALING)
        result = solver.solve(small_problem, verbose=False)
        
        assert len(result.selection) == small_problem.n_qubits
        assert result.objective_value is not None
    
    def test_classical_methods_consistent(self, small_problem):
        """Los métodos clásicos deben producir resultados consistentes."""
        results = {}
        
        for method in [ClassicalMethod.BRUTE_FORCE, ClassicalMethod.GREEDY]:
            solver = ClassicalSolver(method)
            result = solver.solve(small_problem, verbose=False)
            results[method.value] = result.objective_value
        
        # Greedy no debe ser dramáticamente peor que brute force
        optimal = results["brute_force"]
        greedy = results["greedy"]
        
        # Greedy debería estar dentro del 50% del óptimo
        if optimal != 0:
            gap = (greedy - optimal) / abs(optimal)
            assert gap < 0.5, f"Greedy gap too large: {gap:.2%}"


# =============================================================================
# Tests de Comparación
# =============================================================================

class TestSolverComparison:
    """Tests de comparación QAOA vs Clásico."""
    
    def test_compare_solvers_runs(self, small_problem):
        """La comparación debe ejecutarse correctamente."""
        comparison = compare_solvers(
            small_problem,
            qaoa_reps=1,
            classical_method=ClassicalMethod.BRUTE_FORCE,
            verbose=False,
        )
        
        assert comparison is not None
        assert "selection" in comparison.qaoa_result
        assert "selection" in comparison.classical_result
    
    def test_qaoa_competitive_with_optimal(self, small_problem):
        """QAOA debe ser competitivo con la solución óptima."""
        comparison = compare_solvers(
            small_problem,
            qaoa_reps=1,
            classical_method=ClassicalMethod.BRUTE_FORCE,
            verbose=False,
        )
        
        # Gap menor al 10% para problemas pequeños
        assert comparison.objective_gap < 0.10, (
            f"QAOA gap too large: {comparison.objective_gap:.2%}"
        )
    
    def test_qaoa_vs_greedy(self, small_problem):
        """QAOA debe ser al menos tan bueno como greedy."""
        # Ejecutar QAOA
        qaoa_config = QAOAConfig(reps=2, max_iter=100)
        qaoa_solver = QuantumSolver(qaoa_config)
        qaoa_result = qaoa_solver.solve(small_problem, verbose=False)
        
        # Ejecutar Greedy
        greedy_solver = ClassicalSolver(ClassicalMethod.GREEDY)
        greedy_result = greedy_solver.solve(small_problem, verbose=False)
        
        # QAOA debería ser competitivo
        # (puede ser ligeramente peor debido a la naturaleza aproximada)
        assert qaoa_result.objective_value <= greedy_result.objective_value * 1.2
    
    def test_time_ratio_reasonable(self, small_problem):
        """El ratio de tiempo debe ser razonable."""
        comparison = compare_solvers(
            small_problem,
            qaoa_reps=1,
            classical_method=ClassicalMethod.BRUTE_FORCE,
            verbose=False,
        )
        
        # Para problemas pequeños, QAOA puede ser más lento
        # pero no debería ser 100x más lento
        assert comparison.time_ratio < 100


# =============================================================================
# Tests de Escalabilidad
# =============================================================================

class TestScalability:
    """Tests de escalabilidad de los solvers."""
    
    def test_qaoa_scales_with_problem_size(self):
        """QAOA debe escalar razonablemente con el tamaño."""
        times = []
        
        for n in [4, 6, 8]:
            np.random.seed(42)
            tickers = [f"Asset_{i}" for i in range(n)]
            mu = pd.Series(np.random.uniform(0.05, 0.15, n), index=tickers)
            A = np.random.randn(n, n) * 0.1
            sigma_arr = A @ A.T + np.eye(n) * 0.03
            sigma = pd.DataFrame(sigma_arr, index=tickers, columns=tickers)
            
            qubo = QUBOEngine().to_qubo(mu, sigma)
            
            config = QAOAConfig(reps=1, max_iter=50)
            solver = QuantumSolver(config)
            result = solver.solve(qubo, verbose=False)
            
            times.append((n, result.execution_time))
        
        # El tiempo no debería crecer exponencialmente
        # (en simulación, debería ser polinómico)
        for i in range(1, len(times)):
            ratio = times[i][1] / times[i-1][1]
            # Ratio menor a 10x entre tamaños consecutivos
            assert ratio < 10, f"Time growth too fast: {ratio:.2f}x"
    
    def test_classical_exponential_scaling(self):
        """Brute force debe mostrar crecimiento exponencial."""
        times = []
        
        for n in [4, 6, 8]:
            np.random.seed(42)
            tickers = [f"Asset_{i}" for i in range(n)]
            mu = pd.Series(np.random.uniform(0.05, 0.15, n), index=tickers)
            A = np.random.randn(n, n) * 0.1
            sigma_arr = A @ A.T + np.eye(n) * 0.03
            sigma = pd.DataFrame(sigma_arr, index=tickers, columns=tickers)
            
            qubo = QUBOEngine().to_qubo(mu, sigma)
            
            solver = ClassicalSolver(ClassicalMethod.BRUTE_FORCE)
            result = solver.solve(qubo, verbose=False)
            
            times.append((n, result.execution_time, result.n_evaluations))
        
        # Las evaluaciones deben crecer como 2^n
        for i in range(1, len(times)):
            expected_ratio = 2 ** (times[i][0] - times[i-1][0])
            actual_ratio = times[i][2] / times[i-1][2]
            
            assert abs(actual_ratio - expected_ratio) < 0.1, (
                f"Expected {expected_ratio}x evaluations, got {actual_ratio:.2f}x"
            )


# =============================================================================
# Tests de Reproducibilidad
# =============================================================================

class TestReproducibility:
    """Tests de reproducibilidad de resultados."""
    
    def test_classical_deterministic(self, small_problem):
        """Los métodos clásicos deben ser deterministas."""
        solver = ClassicalSolver(ClassicalMethod.BRUTE_FORCE)
        
        results = []
        for _ in range(3):
            result = solver.solve(small_problem, verbose=False)
            results.append(result.objective_value)
        
        # Todos los resultados deben ser iguales
        assert all(r == results[0] for r in results)
    
    def test_qaoa_reasonably_consistent(self, small_problem):
        """QAOA debe ser razonablemente consistente."""
        config = QAOAConfig(reps=1, max_iter=100)
        
        results = []
        for _ in range(3):
            solver = QuantumSolver(config)
            result = solver.solve(small_problem, verbose=False)
            results.append(result.objective_value)
        
        # Los resultados no deben variar más del 20%
        mean_val = np.mean(results)
        for r in results:
            if mean_val != 0:
                deviation = abs(r - mean_val) / abs(mean_val)
                assert deviation < 0.20, f"QAOA too variable: {deviation:.2%}"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
