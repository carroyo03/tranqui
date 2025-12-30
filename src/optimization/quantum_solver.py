"""
Solver cuántico basado en QAOA.

Implementa el Quantum Approximate Optimization Algorithm (QAOA)
usando Qiskit para resolver problemas QUBO de optimización de carteras.

QAOA alternates between:
1. Cost Hamiltonian: H_C = sum_ij Q_ij Z_i Z_j (codifica el problema)
2. Mixer Hamiltonian: H_M = sum_i X_i (explora el espacio de soluciones)

El estado final |ψ(γ,β)⟩ = U_M(β_p) U_C(γ_p) ... U_M(β_1) U_C(γ_1) |+⟩^n
se mide para obtener soluciones candidatas.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
from qiskit.primitives import StatevectorSampler


from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from src.optimization.qubo_engine import QUBOResult

logger = logging.getLogger(__name__)
console = Console()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QAOAConfig:
    """Configuración del solver QAOA."""
    
    reps: int = 1                      # Número de capas (p)
    max_iter: int = 100                # Iteraciones del optimizador clásico
    shots: int = 1024                  # Shots del sampler
    optimizer: str = "COBYLA"          # Optimizador clásico
    initial_point: list[float] | None = None  # Punto inicial (γ, β)
    
    def __post_init__(self):
        if self.reps < 1:
            raise ValueError("reps debe ser >= 1")
        if self.optimizer not in ["COBYLA", "SPSA"]:
            raise ValueError(f"Optimizador '{self.optimizer}' no soportado")


@dataclass
class SolverResult:
    """Resultado del solver cuántico."""
    
    selection: list[int]               # Vector binario de selección
    objective_value: float             # Valor de la función objetivo
    selected_assets: list[str]         # Tickers seleccionados
    rejected_assets: list[str]         # Tickers rechazados
    
    # Metadata de ejecución
    solver_type: str = "QAOA"
    execution_time: float = 0.0
    n_iterations: int = 0
    optimal_params: list[float] = field(default_factory=list)
    
    # Información adicional
    all_samples: dict[str, float] = field(default_factory=dict)
    convergence_history: list[float] = field(default_factory=list)
    
    @property
    def n_selected(self) -> int:
        """Número de activos seleccionados."""
        return sum(self.selection)
    
    def to_dict(self) -> dict:
        """Serializar a diccionario."""
        return {
            "selection": self.selection,
            "objective_value": self.objective_value,
            "selected_assets": self.selected_assets,
            "rejected_assets": self.rejected_assets,
            "solver_type": self.solver_type,
            "execution_time": self.execution_time,
            "n_iterations": self.n_iterations,
            "n_selected": self.n_selected,
        }


# =============================================================================
# Quantum Solver
# =============================================================================

class QuantumSolver:
    """
    Solver cuántico usando QAOA para problemas de optimización de carteras.
    
    Attributes:
        config: Configuración del solver
        convergence_callback: Función llamada en cada iteración
    """
    
    def __init__(
        self,
        config: QAOAConfig | None = None,
        convergence_callback: Callable[[int, float], None] | None = None,
    ):
        """
        Inicializar solver cuántico.
        
        Args:
            config: Configuración QAOA
            convergence_callback: Callback para tracking de convergencia
        """
        self.config = config or QAOAConfig()
        self.convergence_callback = convergence_callback
        self._convergence_history: list[float] = []
        
        logger.info(
            f"QuantumSolver inicializado: reps={self.config.reps}, "
            f"optimizer={self.config.optimizer}"
        )
    
    def solve(
        self,
        qubo_result: QUBOResult,
        verbose: bool = True,
    ) -> SolverResult:
        """
        Resolver problema QUBO usando QAOA.
        
        Args:
            qubo_result: Resultado de la transformación QUBO
            verbose: Si mostrar progreso
        
        Returns:
            SolverResult con la solución óptima encontrada
        """
        Q = qubo_result.Q
        tickers = qubo_result.tickers
        n = len(Q)
        
        if verbose:
            console.print(
                f"[bold yellow]⚛️ Iniciando QAOA con {n} qubits, "
                f"p={self.config.reps}...[/bold yellow]"
            )
        
        # 1. Construir QuadraticProgram de Qiskit
        qp = self._build_quadratic_program(Q, tickers)
        
        # 2. Configurar QAOA
        qaoa = self._configure_qaoa()
        
        # 3. Ejecutar optimización
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not verbose,
        ) as progress:
            task = progress.add_task("Optimizando...", total=None)
            
            algorithm = MinimumEigenOptimizer(qaoa)
            result = algorithm.solve(qp)
            
            progress.update(task, description="[green]✓ Completado[/green]")
        
        execution_time = time.time() - start_time
        
        # 4. Procesar resultado
        selection = [int(x) for x in result.x]
        selected = [t for t, s in zip(tickers, selection) if s == 1]
        rejected = [t for t, s in zip(tickers, selection) if s == 0]
        
        solver_result = SolverResult(
            selection=selection,
            objective_value=float(result.fval),
            selected_assets=selected,
            rejected_assets=rejected,
            solver_type="QAOA",
            execution_time=execution_time,
            convergence_history=self._convergence_history.copy(),
        )
        
        if verbose:
            self._display_result(solver_result)
        
        return solver_result
    
    def _build_quadratic_program(
        self,
        Q: np.ndarray,
        tickers: list[str],
    ) -> QuadraticProgram:
        """
        Construir QuadraticProgram de Qiskit desde matriz QUBO.
        
        Traduce la matriz Q a la representación interna de Qiskit:
        min x^T Q x = sum_i Q_ii x_i + sum_{i<j} (Q_ij + Q_ji) x_i x_j
        """
        n = len(Q)
        qp = QuadraticProgram(name="Portfolio_QUBO")
        
        # Variables binarias
        for i, ticker in enumerate(tickers):
            # Sanitizar nombre para Qiskit (no permite caracteres especiales)
            var_name = f"x_{i}_{ticker.replace('.', '_').replace('-', '_')}"
            qp.binary_var(name=var_name)
        
        # Extraer términos lineales y cuadráticos
        linear_dict = {}
        quadratic_dict = {}
        
        var_names = [v.name for v in qp.variables]
        
        for i in range(n):
            # Términos lineales (diagonal)
            linear_dict[var_names[i]] = Q[i, i]
            
            # Términos cuadráticos (fuera de diagonal)
            for j in range(i + 1, n):
                coef = Q[i, j] + Q[j, i]  # Simetrizar
                if abs(coef) > 1e-10:  # Evitar términos despreciables
                    quadratic_dict[(var_names[i], var_names[j])] = coef
        
        qp.minimize(linear=linear_dict, quadratic=quadratic_dict)
        
        return qp
    
    def _configure_qaoa(self) -> QAOA:
        """Configurar algoritmo QAOA."""
        # Seleccionar optimizador clásico
        if self.config.optimizer == "COBYLA":
            optimizer = COBYLA(maxiter=self.config.max_iter)
        elif self.config.optimizer == "SPSA":
            optimizer = SPSA(maxiter=self.config.max_iter)
        else:
            optimizer = COBYLA(maxiter=self.config.max_iter)
        
        # Sampler (simulador)
        sampler = StatevectorSampler()

        
        # Callback para tracking
        self._convergence_history = []
        
        def callback(eval_count, params, value, metadata):
            self._convergence_history.append(value)
            if self.convergence_callback:
                self.convergence_callback(eval_count, value)
        
        # Configurar QAOA
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=self.config.reps,
            initial_point=self.config.initial_point,
            callback=callback,
        )
        
        return qaoa
    
    def _display_result(self, result: SolverResult) -> None:
        """Mostrar resultado de forma visual."""
        # Panel de resultado
        result_text = (
            f"[bold green]✓ Optimización completada[/bold green]\n\n"
            f"[bold]Valor objetivo:[/bold] {result.objective_value:.6f}\n"
            f"[bold]Tiempo:[/bold] {result.execution_time:.2f}s\n"
            f"[bold]Activos seleccionados:[/bold] {result.n_selected}/{len(result.selection)}\n\n"
        )
        
        # Activos seleccionados
        result_text += "[bold cyan]COMPRAR:[/bold cyan]\n"
        for asset in result.selected_assets:
            result_text += f"  [green]✓[/green] {asset}\n"
        
        result_text += "\n[bold red]EVITAR:[/bold red]\n"
        for asset in result.rejected_assets:
            result_text += f"  [red]✗[/red] {asset}\n"
        
        console.print(Panel(
            result_text,
            title="🏆 Resultado QAOA",
            border_style="green",
        ))


# =============================================================================
# Batch Solver (para benchmarking)
# =============================================================================

class BatchQuantumSolver:
    """
    Ejecutar múltiples configuraciones QAOA para benchmarking.
    """
    
    def __init__(self):
        self.results: list[dict] = []
    
    def run_sweep(
        self,
        qubo_result: QUBOResult,
        reps_range: list[int] = [1, 2, 3],
        n_runs: int = 3,
    ) -> list[dict]:
        """
        Ejecutar sweep de configuraciones.
        
        Args:
            qubo_result: Problema QUBO a resolver
            reps_range: Valores de p (capas) a probar
            n_runs: Número de ejecuciones por configuración
        
        Returns:
            Lista de resultados con estadísticas
        """
        results = []
        
        for reps in reps_range:
            run_results = []
            
            for run in range(n_runs):
                config = QAOAConfig(reps=reps)
                solver = QuantumSolver(config)
                result = solver.solve(qubo_result, verbose=False)
                
                run_results.append({
                    "objective": result.objective_value,
                    "time": result.execution_time,
                    "selection": result.selection,
                })
            
            # Estadísticas agregadas
            objectives = [r["objective"] for r in run_results]
            times = [r["time"] for r in run_results]
            
            results.append({
                "reps": reps,
                "n_runs": n_runs,
                "objective_mean": np.mean(objectives),
                "objective_std": np.std(objectives),
                "objective_best": np.min(objectives),
                "time_mean": np.mean(times),
                "time_std": np.std(times),
                "best_selection": run_results[np.argmin(objectives)]["selection"],
            })
        
        self.results = results
        return results
    
    def display_sweep_results(self) -> None:
        """Mostrar resultados del sweep."""
        from rich.table import Table
        
        table = Table(title="📊 QAOA Hyperparameter Sweep")
        table.add_column("Reps (p)", justify="center", style="cyan")
        table.add_column("Objetivo (mean±std)", justify="right")
        table.add_column("Mejor", justify="right", style="green")
        table.add_column("Tiempo (s)", justify="right")
        
        for r in self.results:
            table.add_row(
                str(r["reps"]),
                f"{r['objective_mean']:.4f} ± {r['objective_std']:.4f}",
                f"{r['objective_best']:.4f}",
                f"{r['time_mean']:.2f} ± {r['time_std']:.2f}",
            )
        
        console.print(table)


# =============================================================================
# Convenience Functions
# =============================================================================

def solve_with_qaoa(
    mu,
    sigma,
    risk_aversion: float = 0.5,
    reps: int = 1,
    min_assets: int | None = None,
    max_assets: int | None = None,
) -> SolverResult:
    """
    Función de conveniencia para resolver directamente desde datos financieros.
    
    Mantiene compatibilidad con código original.
    
    Args:
        mu: Retornos esperados
        sigma: Matriz de covarianza
        risk_aversion: Aversión al riesgo [0,1]
        reps: Capas QAOA
        min_assets: Mínimo de activos
        max_assets: Máximo de activos
    
    Returns:
        SolverResult con la solución
    """
    from src.optimization.qubo_engine import QUBOEngine, Constraint, ConstraintType
    
    # Construir QUBO
    constraints = []
    if min_assets is not None:
        constraints.append(Constraint(ConstraintType.MIN_ASSETS, min_assets))
    if max_assets is not None:
        constraints.append(Constraint(ConstraintType.MAX_ASSETS, max_assets))
    
    qubo_engine = QUBOEngine()
    qubo_result = qubo_engine.to_qubo(mu, sigma, risk_aversion, constraints)
    
    # Resolver
    config = QAOAConfig(reps=reps)
    solver = QuantumSolver(config)
    
    return solver.solve(qubo_result)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    from src.optimization.qubo_engine import QUBOEngine, Constraint, ConstraintType
    
    # Datos de ejemplo
    tickers = ["SAN.MC", "ITX.MC", "IBE.MC", "TSLA"]
    
    np.random.seed(42)
    mu = pd.Series([0.08, 0.12, 0.06, 0.20], index=tickers)
    sigma = pd.DataFrame(
        [
            [0.04, 0.01, 0.02, 0.01],
            [0.01, 0.09, 0.01, 0.03],
            [0.02, 0.01, 0.03, 0.00],
            [0.01, 0.03, 0.00, 0.35],
        ],
        index=tickers,
        columns=tickers,
    )
    
    # Construir QUBO
    console.rule("[bold cyan]⚛️ Quantum Portfolio Optimization[/bold cyan]")
    
    qubo_engine = QUBOEngine()
    constraints = [
        Constraint(ConstraintType.MIN_ASSETS, 2),
    ]
    qubo_result = qubo_engine.to_qubo(mu, sigma, risk_aversion=0.5, constraints=constraints)
    
    # Resolver con diferentes configuraciones
    console.rule("[bold magenta]🔄 QAOA Sweep[/bold magenta]")
    
    batch_solver = BatchQuantumSolver()
    batch_solver.run_sweep(qubo_result, reps_range=[1, 2], n_runs=3)
    batch_solver.display_sweep_results()
    
    # Mejor resultado
    console.rule("[bold green]🏆 Mejor Configuración[/bold green]")
    
    config = QAOAConfig(reps=2, max_iter=150)
    solver = QuantumSolver(config)
    result = solver.solve(qubo_result)
