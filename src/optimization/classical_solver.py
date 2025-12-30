"""
Solver clásico para optimización de carteras.

Proporciona baseline de comparación para evaluar el rendimiento de QAOA.
Implementa múltiples métodos clásicos:
1. Fuerza bruta (exacto, para n ≤ 20)
2. Branch and bound (MILP)
3. Greedy heurístico
4. Simulated annealing

Esto es CRÍTICO para la validación de la tesis:
sin baseline clásico, no hay forma de evaluar si QAOA aporta valor.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from scipy.optimize import minimize

if TYPE_CHECKING:
    from src.optimization.qubo_engine import QUBOResult

logger = logging.getLogger(__name__)
console = Console()


# =============================================================================
# Enums
# =============================================================================

class ClassicalMethod(Enum):
    """Métodos clásicos disponibles."""
    BRUTE_FORCE = "brute_force"        # Exacto, O(2^n)
    GREEDY = "greedy"                  # Heurístico, O(n^2)
    SIMULATED_ANNEALING = "sa"         # Metaheurístico
    SCIPY_MINIMIZE = "scipy"           # Relajación continua


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ClassicalResult:
    """Resultado del solver clásico."""
    
    selection: list[int]
    objective_value: float
    selected_assets: list[str]
    rejected_assets: list[str]
    
    method: ClassicalMethod
    execution_time: float
    is_optimal: bool  # True si se garantiza óptimo global
    n_evaluations: int = 0
    
    @property
    def n_selected(self) -> int:
        return sum(self.selection)
    
    def to_dict(self) -> dict:
        return {
            "selection": self.selection,
            "objective_value": self.objective_value,
            "selected_assets": self.selected_assets,
            "rejected_assets": self.rejected_assets,
            "method": self.method.value,
            "execution_time": self.execution_time,
            "is_optimal": self.is_optimal,
            "n_evaluations": self.n_evaluations,
        }


# =============================================================================
# Classical Solver
# =============================================================================

class ClassicalSolver:
    """
    Solver clásico para problemas QUBO.
    
    Proporciona múltiples métodos con diferentes trade-offs
    entre optimalidad y velocidad.
    """
    
    # Límite de qubits para fuerza bruta
    BRUTE_FORCE_LIMIT = 20
    
    def __init__(self, method: ClassicalMethod = ClassicalMethod.BRUTE_FORCE):
        """
        Inicializar solver clásico.
        
        Args:
            method: Método de optimización a usar
        """
        self.method = method
        logger.info(f"ClassicalSolver inicializado con método: {method.value}")
    
    def solve(
        self,
        qubo_result: QUBOResult,
        verbose: bool = True,
    ) -> ClassicalResult:
        """
        Resolver problema QUBO con método clásico.
        
        Args:
            qubo_result: Problema QUBO
            verbose: Mostrar progreso
        
        Returns:
            ClassicalResult con la solución
        """
        Q = qubo_result.Q
        n = len(Q)
        tickers = qubo_result.tickers
        
        if verbose:
            console.print(
                f"[bold blue]🖥️ Resolviendo con método clásico: "
                f"{self.method.value} (n={n})[/bold blue]"
            )
        
        start_time = time.time()
        
        # Seleccionar método
        if self.method == ClassicalMethod.BRUTE_FORCE:
            selection, obj_val, n_evals, is_optimal = self._brute_force(Q, verbose)
        elif self.method == ClassicalMethod.GREEDY:
            selection, obj_val, n_evals, is_optimal = self._greedy(Q, verbose)
        elif self.method == ClassicalMethod.SIMULATED_ANNEALING:
            selection, obj_val, n_evals, is_optimal = self._simulated_annealing(Q, verbose)
        elif self.method == ClassicalMethod.SCIPY_MINIMIZE:
            selection, obj_val, n_evals, is_optimal = self._scipy_minimize(Q, verbose)
        else:
            raise ValueError(f"Método no soportado: {self.method}")
        
        execution_time = time.time() - start_time
        
        # Construir resultado
        selected = [t for t, s in zip(tickers, selection) if s == 1]
        rejected = [t for t, s in zip(tickers, selection) if s == 0]
        
        result = ClassicalResult(
            selection=selection,
            objective_value=obj_val,
            selected_assets=selected,
            rejected_assets=rejected,
            method=self.method,
            execution_time=execution_time,
            is_optimal=is_optimal,
            n_evaluations=n_evals,
        )
        
        if verbose:
            self._display_result(result)
        
        return result
    
    def _brute_force(
        self,
        Q: np.ndarray,
        verbose: bool,
    ) -> tuple[list[int], float, int, bool]:
        """
        Búsqueda exhaustiva de la solución óptima.
        
        Complejidad: O(2^n) - solo viable para n ≤ 20
        Garantiza encontrar el óptimo global.
        """
        n = len(Q)
        
        if n > self.BRUTE_FORCE_LIMIT:
            logger.warning(
                f"n={n} excede límite de fuerza bruta ({self.BRUTE_FORCE_LIMIT}). "
                "Usando greedy como fallback."
            )
            return self._greedy(Q, verbose)
        
        best_x = None
        best_val = float('inf')
        n_evals = 0
        
        total = 2 ** n
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            disable=not verbose,
        ) as progress:
            task = progress.add_task("Fuerza bruta...", total=total)
            
            # Iterar sobre todas las combinaciones binarias
            for i in range(total):
                # Convertir índice a vector binario
                x = np.array([int(b) for b in format(i, f'0{n}b')])
                
                # Evaluar función objetivo: x^T Q x
                val = x @ Q @ x
                n_evals += 1
                
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                
                progress.update(task, advance=1)
        
        return list(best_x), float(best_val), n_evals, True
    
    def _greedy(
        self,
        Q: np.ndarray,
        verbose: bool,
    ) -> tuple[list[int], float, int, bool]:
        """
        Algoritmo greedy basado en contribución marginal.
        
        Complejidad: O(n^2)
        No garantiza óptimo global.
        """
        n = len(Q)
        x = np.zeros(n, dtype=int)
        n_evals = 0
        
        # Valor inicial (ningún activo seleccionado)
        current_val = 0.0
        
        # Iterar mientras haya mejora
        improved = True
        while improved:
            improved = False
            best_flip = -1
            best_delta = 0.0
            
            for i in range(n):
                # Calcular cambio en objetivo si flippeamos bit i
                x_new = x.copy()
                x_new[i] = 1 - x_new[i]  # Flip
                
                new_val = x_new @ Q @ x_new
                delta = new_val - current_val
                n_evals += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_flip = i
                    improved = True
            
            if improved:
                x[best_flip] = 1 - x[best_flip]
                current_val += best_delta
        
        return list(x), float(current_val), n_evals, False
    
    def _simulated_annealing(
        self,
        Q: np.ndarray,
        verbose: bool,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.995,
        min_temp: float = 0.01,
        max_iter_per_temp: int = 100,
    ) -> tuple[list[int], float, int, bool]:
        """
        Simulated Annealing para optimización combinatoria.
        
        Permite explorar el espacio de soluciones de forma estocástica,
        escapando de óptimos locales.
        """
        n = len(Q)
        
        # Solución inicial aleatoria
        np.random.seed(42)  # Reproducibilidad
        x = np.random.randint(0, 2, n)
        current_val = x @ Q @ x
        
        best_x = x.copy()
        best_val = current_val
        
        temp = initial_temp
        n_evals = 0
        
        while temp > min_temp:
            for _ in range(max_iter_per_temp):
                # Generar vecino (flip aleatorio)
                i = np.random.randint(n)
                x_new = x.copy()
                x_new[i] = 1 - x_new[i]
                
                new_val = x_new @ Q @ x_new
                n_evals += 1
                
                # Criterio de aceptación Metropolis
                delta = new_val - current_val
                if delta < 0 or np.random.random() < np.exp(-delta / temp):
                    x = x_new
                    current_val = new_val
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_x = x.copy()
            
            temp *= cooling_rate
        
        return list(best_x), float(best_val), n_evals, False
    
    def _scipy_minimize(
        self,
        Q: np.ndarray,
        verbose: bool,
    ) -> tuple[list[int], float, int, bool]:
        """
        Relajación continua + redondeo.
        
        Relaja el problema binario a [0,1]^n, resuelve,
        y redondea a la solución binaria más cercana.
        """
        n = len(Q)
        n_evals = [0]  # Lista para permitir modificación en closure
        
        def objective(x):
            n_evals[0] += 1
            return x @ Q @ x
        
        def gradient(x):
            return 2 * Q @ x
        
        # Punto inicial
        x0 = np.ones(n) * 0.5
        
        # Bounds [0, 1]
        bounds = [(0, 1)] * n
        
        # Optimizar
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac=gradient,
            bounds=bounds,
        )
        
        # Redondear a binario
        x_continuous = result.x
        x_binary = (x_continuous > 0.5).astype(int)
        
        # Evaluar solución binaria
        final_val = x_binary @ Q @ x_binary
        
        return list(x_binary), float(final_val), n_evals[0], False
    
    def _display_result(self, result: ClassicalResult) -> None:
        """Mostrar resultado."""
        from rich.panel import Panel
        
        optimal_badge = "[green](ÓPTIMO)[/green]" if result.is_optimal else "[yellow](heurístico)[/yellow]"
        
        text = (
            f"[bold]Método:[/bold] {result.method.value} {optimal_badge}\n"
            f"[bold]Valor objetivo:[/bold] {result.objective_value:.6f}\n"
            f"[bold]Tiempo:[/bold] {result.execution_time:.4f}s\n"
            f"[bold]Evaluaciones:[/bold] {result.n_evaluations:,}\n\n"
            f"[bold cyan]Seleccionados:[/bold cyan] {', '.join(result.selected_assets)}\n"
            f"[bold red]Rechazados:[/bold red] {', '.join(result.rejected_assets)}"
        )
        
        console.print(Panel(text, title="🖥️ Resultado Clásico", border_style="blue"))


# =============================================================================
# Comparador Quantum vs Clásico
# =============================================================================

@dataclass
class ComparisonResult:
    """Resultado de comparación quantum vs clásico."""
    
    qaoa_result: dict
    classical_result: dict
    
    # Métricas de comparación
    objective_gap: float           # (qaoa - classical) / |classical|
    time_ratio: float              # qaoa_time / classical_time
    same_solution: bool            # Si encontraron la misma solución
    
    @property
    def qaoa_wins(self) -> bool:
        """Si QAOA encontró mejor solución."""
        return self.objective_gap < 0
    
    def summary(self) -> str:
        """Resumen textual de la comparación."""
        winner = "QAOA" if self.qaoa_wins else "Clásico"
        gap_pct = abs(self.objective_gap) * 100
        
        return (
            f"Ganador: {winner} | "
            f"Gap: {gap_pct:.2f}% | "
            f"Ratio tiempo: {self.time_ratio:.2f}x | "
            f"Misma solución: {self.same_solution}"
        )


def compare_solvers(
    qubo_result: QUBOResult,
    qaoa_reps: int = 1,
    classical_method: ClassicalMethod = ClassicalMethod.BRUTE_FORCE,
    verbose: bool = True,
) -> ComparisonResult:
    """
    Comparar QAOA con solver clásico.
    
    Esta función es ESENCIAL para la validación de la tesis.
    
    Args:
        qubo_result: Problema QUBO
        qaoa_reps: Capas QAOA
        classical_method: Método clásico a usar
        verbose: Mostrar resultados
    
    Returns:
        ComparisonResult con métricas de comparación
    """
    from src.optimization.quantum_solver import QuantumSolver, QAOAConfig
    
    if verbose:
        console.rule("[bold magenta]🔬 Comparación Quantum vs Clásico[/bold magenta]")
    
    # 1. Resolver con método clásico
    classical_solver = ClassicalSolver(method=classical_method)
    classical_result = classical_solver.solve(qubo_result, verbose=verbose)
    
    # 2. Resolver con QAOA
    qaoa_config = QAOAConfig(reps=qaoa_reps)
    qaoa_solver = QuantumSolver(qaoa_config)
    qaoa_result = qaoa_solver.solve(qubo_result, verbose=verbose)
    
    # 3. Calcular métricas
    classical_obj = classical_result.objective_value
    qaoa_obj = qaoa_result.objective_value
    
    # Gap relativo
    if abs(classical_obj) > 1e-10:
        gap = (qaoa_obj - classical_obj) / abs(classical_obj)
    else:
        gap = qaoa_obj - classical_obj
    
    # Ratio de tiempo
    time_ratio = qaoa_result.execution_time / max(classical_result.execution_time, 1e-6)
    
    # Misma solución?
    same_solution = qaoa_result.selection == classical_result.selection
    
    comparison = ComparisonResult(
        qaoa_result=qaoa_result.to_dict(),
        classical_result=classical_result.to_dict(),
        objective_gap=gap,
        time_ratio=time_ratio,
        same_solution=same_solution,
    )
    
    if verbose:
        _display_comparison(comparison)
    
    return comparison


def _display_comparison(result: ComparisonResult) -> None:
    """Mostrar comparación de forma visual."""
    from rich.table import Table
    
    table = Table(title="📊 Comparación de Solvers")
    
    table.add_column("Métrica", style="cyan")
    table.add_column("QAOA", justify="right")
    table.add_column("Clásico", justify="right")
    table.add_column("Diferencia", justify="right")
    
    qaoa = result.qaoa_result
    classical = result.classical_result
    
    # Objetivo
    obj_diff = qaoa["objective_value"] - classical["objective_value"]
    obj_style = "green" if obj_diff <= 0 else "red"
    table.add_row(
        "Valor Objetivo",
        f"{qaoa['objective_value']:.6f}",
        f"{classical['objective_value']:.6f}",
        f"[{obj_style}]{obj_diff:+.6f}[/{obj_style}]",
    )
    
    # Tiempo
    time_style = "red" if result.time_ratio > 1 else "green"
    table.add_row(
        "Tiempo (s)",
        f"{qaoa['execution_time']:.4f}",
        f"{classical['execution_time']:.4f}",
        f"[{time_style}]{result.time_ratio:.2f}x[/{time_style}]",
    )
    
    # Solución
    solution_match = "✓ Igual" if result.same_solution else "✗ Diferente"
    match_style = "green" if result.same_solution else "yellow"
    table.add_row(
        "Solución",
        str(qaoa["selection"]),
        str(classical["selection"]),
        f"[{match_style}]{solution_match}[/{match_style}]",
    )
    
    console.print(table)
    
    # Veredicto
    gap_pct = result.objective_gap * 100
    if result.objective_gap <= 0.01:  # Dentro del 1%
        verdict = "[green]✓ QAOA encuentra solución competitiva[/green]"
    elif result.objective_gap <= 0.05:
        verdict = "[yellow]~ QAOA dentro del 5% del óptimo[/yellow]"
    else:
        verdict = f"[red]✗ QAOA {gap_pct:.1f}% peor que clásico[/red]"
    
    console.print(f"\n[bold]Veredicto:[/bold] {verdict}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    from src.optimization.qubo_engine import QUBOEngine, Constraint, ConstraintType
    
    # Datos de ejemplo
    tickers = ["SAN.MC", "ITX.MC", "IBE.MC", "TEF.MC", "AAPL", "MSFT"]
    
    np.random.seed(42)
    n = len(tickers)
    
    # Generar datos sintéticos realistas
    mu = pd.Series(np.random.uniform(0.05, 0.15, n), index=tickers)
    
    # Matriz de covarianza (asegurar PSD)
    A = np.random.randn(n, n) * 0.1
    sigma_arr = A @ A.T + np.eye(n) * 0.05
    sigma = pd.DataFrame(sigma_arr, index=tickers, columns=tickers)
    
    # Crear QUBO
    qubo_engine = QUBOEngine()
    constraints = [Constraint(ConstraintType.MIN_ASSETS, 2)]
    qubo_result = qubo_engine.to_qubo(mu, sigma, risk_aversion=0.5, constraints=constraints)
    
    # Comparar métodos clásicos
    console.rule("[bold cyan]🖥️ Comparación de Métodos Clásicos[/bold cyan]")
    
    for method in [ClassicalMethod.BRUTE_FORCE, ClassicalMethod.GREEDY, ClassicalMethod.SIMULATED_ANNEALING]:
        solver = ClassicalSolver(method=method)
        result = solver.solve(qubo_result, verbose=True)
        console.print()
    
    # Comparar con QAOA
    console.print()
    comparison = compare_solvers(qubo_result, qaoa_reps=2)
