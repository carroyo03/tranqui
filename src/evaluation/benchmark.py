"""
Motor de benchmarking para evaluación de solvers.

Proporciona herramientas para:
1. Comparación sistemática QAOA vs Clásico
2. Análisis de escalabilidad
3. Evaluación estadística de resultados
4. Generación de reportes para la tesis

Este módulo es CRÍTICO para demostrar rigor académico.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.optimization.qubo_engine import QUBOEngine, QUBOResult, Constraint, ConstraintType
from src.optimization.quantum_solver import QuantumSolver, QAOAConfig, SolverResult
from src.optimization.classical_solver import ClassicalSolver, ClassicalMethod, ClassicalResult

logger = logging.getLogger(__name__)
console = Console()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuración del benchmark."""
    
    # Tamaños de problema a evaluar
    problem_sizes: list[int] = field(default_factory=lambda: [4, 6, 8, 10, 12])
    
    # Configuraciones QAOA
    qaoa_reps_range: list[int] = field(default_factory=lambda: [1, 2])
    qaoa_max_iter: int = 100
    
    # Métodos clásicos a comparar
    classical_methods: list[ClassicalMethod] = field(
        default_factory=lambda: [ClassicalMethod.BRUTE_FORCE, ClassicalMethod.GREEDY]
    )
    
    # Repeticiones para estadísticas
    n_runs: int = 5
    
    # Parámetros financieros
    risk_aversion_range: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    
    # Random seed para reproducibilidad
    random_seed: int = 42
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("results"))


@dataclass
class SingleRunResult:
    """Resultado de una ejecución individual."""
    
    problem_size: int
    risk_aversion: float
    solver_type: str
    solver_config: dict
    
    objective_value: float
    selection: list[int]
    execution_time: float
    
    is_optimal: bool = False
    n_evaluations: int = 0
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkResult:
    """Resultado agregado del benchmark."""
    
    config: BenchmarkConfig
    runs: list[SingleRunResult] = field(default_factory=list)
    
    # Estadísticas agregadas
    summary_df: pd.DataFrame = field(default=None)
    
    def add_run(self, run: SingleRunResult) -> None:
        """Añadir resultado de ejecución."""
        self.runs.append(run)
    
    def compute_summary(self) -> pd.DataFrame:
        """Calcular estadísticas agregadas."""
        if not self.runs:
            return pd.DataFrame()
        
        # Convertir a DataFrame
        data = [asdict(r) for r in self.runs]
        df = pd.DataFrame(data)
        
        # Agrupar y calcular estadísticas
        grouped = df.groupby(['problem_size', 'solver_type', 'risk_aversion']).agg({
            'objective_value': ['mean', 'std', 'min'],
            'execution_time': ['mean', 'std'],
            'is_optimal': 'mean',  # Proporción de óptimos
        }).round(6)
        
        # Aplanar columnas
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        self.summary_df = grouped.reset_index()
        
        return self.summary_df
    
    def get_gap_analysis(self) -> pd.DataFrame:
        """Calcular gap entre QAOA y óptimo clásico."""
        if not self.runs:
            return pd.DataFrame()
        
        data = [asdict(r) for r in self.runs]
        df = pd.DataFrame(data)
        
        # Encontrar mejor solución clásica por configuración
        classical = df[df['solver_type'].str.contains('classical')]
        qaoa = df[df['solver_type'].str.contains('qaoa')]
        
        gaps = []
        for (size, risk), group in classical.groupby(['problem_size', 'risk_aversion']):
            best_classical = group['objective_value'].min()
            
            qaoa_group = qaoa[
                (qaoa['problem_size'] == size) & 
                (qaoa['risk_aversion'] == risk)
            ]
            
            for _, row in qaoa_group.iterrows():
                gap = (row['objective_value'] - best_classical) / abs(best_classical) if best_classical != 0 else 0
                gaps.append({
                    'problem_size': size,
                    'risk_aversion': risk,
                    'qaoa_config': row['solver_config'],
                    'gap_pct': gap * 100,
                    'qaoa_time': row['execution_time'],
                    'classical_best': best_classical,
                })
        
        return pd.DataFrame(gaps)
    
    def save(self, filepath: Path) -> None:
        """Guardar resultados a JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            'config': asdict(self.config),
            'runs': [asdict(r) for r in self.runs],
            'summary': self.summary_df.to_dict() if self.summary_df is not None else None,
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"Resultados guardados en {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'BenchmarkResult':
        """Cargar resultados desde JSON."""
        with open(filepath) as f:
            data = json.load(f)
        
        config = BenchmarkConfig(**data['config'])
        runs = [SingleRunResult(**r) for r in data['runs']]
        
        result = cls(config=config, runs=runs)
        if data.get('summary'):
            result.summary_df = pd.DataFrame(data['summary'])
        
        return result


# =============================================================================
# Benchmark Engine
# =============================================================================

class BenchmarkEngine:
    """
    Motor de benchmarking para comparación sistemática de solvers.
    """
    
    def __init__(self, config: BenchmarkConfig | None = None):
        """
        Inicializar motor de benchmark.
        
        Args:
            config: Configuración del benchmark
        """
        self.config = config or BenchmarkConfig()
        self.result = BenchmarkResult(config=self.config)
        
        # Configurar reproducibilidad
        np.random.seed(self.config.random_seed)
        
        logger.info(f"BenchmarkEngine inicializado con {len(self.config.problem_sizes)} tamaños de problema")
    
    def run_full_benchmark(self, verbose: bool = True) -> BenchmarkResult:
        """
        Ejecutar benchmark completo.
        
        Args:
            verbose: Mostrar progreso
        
        Returns:
            BenchmarkResult con todos los resultados
        """
        total_runs = (
            len(self.config.problem_sizes) *
            len(self.config.risk_aversion_range) *
            (len(self.config.classical_methods) + len(self.config.qaoa_reps_range)) *
            self.config.n_runs
        )
        
        if verbose:
            console.rule("[bold magenta]🔬 Iniciando Benchmark Completo[/bold magenta]")
            console.print(f"Total de ejecuciones: {total_runs}")
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            disable=not verbose,
        ) as progress:
            task = progress.add_task("Benchmark...", total=total_runs)
            
            # Generar todas las tareas
            tasks = []
            
            for n_assets in self.config.problem_sizes:
                for risk_aversion in self.config.risk_aversion_range:
                    # Generar problema una vez para cada configuración (n, risk)
                    qubo_result = self._generate_problem(n_assets, risk_aversion)
                    
                    # Tareas clásicas
                    for method in self.config.classical_methods:
                        if method == ClassicalMethod.BRUTE_FORCE and n_assets > 16:
                            continue
                        
                        for _ in range(self.config.n_runs):
                            tasks.append({
                                "type": "classical",
                                "method": method,
                                "qubo_result": qubo_result,
                                "n_assets": n_assets,
                                "risk_aversion": risk_aversion,
                            })
                    
                    # Tareas QAOA
                    for reps in self.config.qaoa_reps_range:
                        for _ in range(self.config.n_runs):
                            tasks.append({
                                "type": "qaoa",
                                "reps": reps,
                                "qubo_result": qubo_result,
                                "n_assets": n_assets,
                                "risk_aversion": risk_aversion,
                            })
            
            # Ejecutar en paralelo
            from joblib import Parallel, delayed
            
            n_jobs = -1  # Usar todos los núcleos disponibles
            
            if verbose:
                progress.update(task, total=len(tasks), description="[bold cyan]Ejecutando tareas en paralelo...[/bold cyan]")
            
            def execute_task(t):
                if t["type"] == "classical":
                    return self._run_classical(
                        t["qubo_result"], t["method"], t["n_assets"], t["risk_aversion"]
                    )
                else:
                    return self._run_qaoa(
                        t["qubo_result"], t["reps"], t["n_assets"], t["risk_aversion"]
                    )
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(execute_task)(t) for t in tasks
            )
            
            # Recolectar resultados
            for r in results:
                self.result.add_run(r)
                if verbose:
                    progress.update(task, advance=1)

        
        # Calcular resumen
        self.result.compute_summary()
        
        if verbose:
            self._display_summary()
        
        return self.result
    
    def run_scalability_analysis(
        self,
        max_size: int = 16,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Análisis específico de escalabilidad.
        
        Evalúa cómo escala el tiempo de ejecución con el tamaño del problema.
        """
        sizes = list(range(4, max_size + 1, 2))
        results = []
        
        if verbose:
            console.rule("[bold cyan]📈 Análisis de Escalabilidad[/bold cyan]")
        
        for n in sizes:
            qubo = self._generate_problem(n, risk_aversion=0.5)
            
            # QAOA
            qaoa_times = []
            for _ in range(3):
                config = QAOAConfig(reps=1, max_iter=50)
                solver = QuantumSolver(config)
                start = time.time()
                solver.solve(qubo, verbose=False)
                qaoa_times.append(time.time() - start)
            
            # Clásico (greedy para todos los tamaños)
            classical_times = []
            for _ in range(3):
                solver = ClassicalSolver(ClassicalMethod.GREEDY)
                start = time.time()
                solver.solve(qubo, verbose=False)
                classical_times.append(time.time() - start)
            
            results.append({
                'n_assets': n,
                'n_qubits': n,
                'qaoa_time_mean': np.mean(qaoa_times),
                'qaoa_time_std': np.std(qaoa_times),
                'classical_time_mean': np.mean(classical_times),
                'classical_time_std': np.std(classical_times),
                'speedup': np.mean(classical_times) / np.mean(qaoa_times),
            })
            
            if verbose:
                console.print(f"n={n}: QAOA={np.mean(qaoa_times):.3f}s, Classical={np.mean(classical_times):.3f}s")
        
        return pd.DataFrame(results)
    
    def _generate_problem(
        self,
        n_assets: int,
        risk_aversion: float,
    ) -> QUBOResult:
        """Generar problema QUBO sintético."""
        # Tickers sintéticos
        tickers = [f"Asset_{i}" for i in range(n_assets)]
        
        # Retornos esperados (distribución realista)
        mu = pd.Series(
            np.random.uniform(0.02, 0.15, n_assets),
            index=tickers
        )
        
        # Matriz de covarianza (asegurar PSD)
        A = np.random.randn(n_assets, n_assets) * 0.1
        sigma_arr = A @ A.T + np.eye(n_assets) * 0.02
        sigma = pd.DataFrame(sigma_arr, index=tickers, columns=tickers)
        
        # Construir QUBO
        qubo_engine = QUBOEngine()
        constraints = [Constraint(ConstraintType.MIN_ASSETS, max(2, n_assets // 3))]
        
        return qubo_engine.to_qubo(mu, sigma, risk_aversion, constraints)
    
    def _run_classical(
        self,
        qubo: QUBOResult,
        method: ClassicalMethod,
        n_assets: int,
        risk_aversion: float,
    ) -> SingleRunResult:
        """Ejecutar solver clásico."""
        solver = ClassicalSolver(method)
        result = solver.solve(qubo, verbose=False)
        
        return SingleRunResult(
            problem_size=n_assets,
            risk_aversion=risk_aversion,
            solver_type=f"classical_{method.value}",
            solver_config={"method": method.value},
            objective_value=result.objective_value,
            selection=result.selection,
            execution_time=result.execution_time,
            is_optimal=result.is_optimal,
            n_evaluations=result.n_evaluations,
        )
    
    def _run_qaoa(
        self,
        qubo: QUBOResult,
        reps: int,
        n_assets: int,
        risk_aversion: float,
    ) -> SingleRunResult:
        """Ejecutar solver QAOA."""
        config = QAOAConfig(reps=reps, max_iter=self.config.qaoa_max_iter)
        solver = QuantumSolver(config)
        result = solver.solve(qubo, verbose=False)
        
        return SingleRunResult(
            problem_size=n_assets,
            risk_aversion=risk_aversion,
            solver_type=f"qaoa_reps{reps}",
            solver_config={"reps": reps, "max_iter": self.config.qaoa_max_iter},
            objective_value=result.objective_value,
            selection=result.selection,
            execution_time=result.execution_time,
            is_optimal=False,
        )
    
    def _display_summary(self) -> None:
        """Mostrar resumen de resultados."""
        console.rule("[bold green]📊 Resumen del Benchmark[/bold green]")
        
        df = self.result.summary_df
        if df is None or df.empty:
            console.print("[yellow]No hay resultados para mostrar[/yellow]")
            return
        
        # Tabla por tamaño de problema
        table = Table(title="Resultados por Configuración")
        table.add_column("Tamaño", style="cyan", justify="center")
        table.add_column("Solver", style="yellow")
        table.add_column("λ", justify="center")
        table.add_column("Objetivo (mean±std)", justify="right")
        table.add_column("Tiempo (s)", justify="right")
        
        for _, row in df.iterrows():
            table.add_row(
                str(int(row['problem_size'])),
                row['solver_type'],
                f"{row['risk_aversion']:.1f}",
                f"{row['objective_value_mean']:.4f} ± {row['objective_value_std']:.4f}",
                f"{row['execution_time_mean']:.3f}",
            )
        
        console.print(table)
        
        # Gap analysis
        gap_df = self.result.get_gap_analysis()
        if not gap_df.empty:
            console.print("\n[bold]Gap QAOA vs Óptimo Clásico:[/bold]")
            for _, row in gap_df.groupby('problem_size').agg({'gap_pct': 'mean'}).iterrows():
                gap = row['gap_pct']
                style = "green" if gap < 1 else "yellow" if gap < 5 else "red"
                console.print(f"  n={_}: [{style}]{gap:.2f}%[/{style}]")


# =============================================================================
# Statistical Tests
# =============================================================================

def statistical_comparison(
    result: BenchmarkResult,
    significance_level: float = 0.05,
) -> dict:
    """
    Realizar tests estadísticos de comparación.
    
    Args:
        result: Resultados del benchmark
        significance_level: Nivel de significancia (α)
    
    Returns:
        Diccionario con resultados de tests estadísticos
    """
    from scipy import stats
    
    data = [asdict(r) for r in result.runs]
    df = pd.DataFrame(data)
    
    results = {}
    
    # Para cada tamaño de problema, comparar QAOA vs mejor clásico
    for size in df['problem_size'].unique():
        size_df = df[df['problem_size'] == size]
        
        # Obtener valores objetivo
        classical = size_df[size_df['solver_type'].str.contains('brute_force')]['objective_value']
        qaoa = size_df[size_df['solver_type'].str.contains('qaoa')]['objective_value']
        
        if len(classical) > 1 and len(qaoa) > 1:
            # Test t de dos muestras
            t_stat, p_value = stats.ttest_ind(classical, qaoa)
            
            # Test de Mann-Whitney (no paramétrico)
            u_stat, p_value_mw = stats.mannwhitneyu(classical, qaoa, alternative='two-sided')
            
            results[f"size_{size}"] = {
                't_statistic': t_stat,
                't_pvalue': p_value,
                'mannwhitney_u': u_stat,
                'mannwhitney_pvalue': p_value_mw,
                'significant': p_value < significance_level,
                'classical_mean': classical.mean(),
                'qaoa_mean': qaoa.mean(),
            }
    
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Configuración reducida para demo
    config = BenchmarkConfig(
        problem_sizes=[4, 6, 8],
        qaoa_reps_range=[1, 2],
        classical_methods=[ClassicalMethod.BRUTE_FORCE, ClassicalMethod.GREEDY],
        n_runs=3,
        risk_aversion_range=[0.5],
    )
    
    # Ejecutar benchmark
    engine = BenchmarkEngine(config)
    result = engine.run_full_benchmark()
    
    # Guardar resultados
    output_path = Path("results/benchmark_results.json")
    result.save(output_path)
    console.print(f"\n[green]Resultados guardados en {output_path}[/green]")
    
    # Análisis estadístico
    console.rule("[bold cyan]📈 Análisis Estadístico[/bold cyan]")
    stats_results = statistical_comparison(result)
    for key, value in stats_results.items():
        sig = "✓ Significativo" if value['significant'] else "✗ No significativo"
        console.print(f"{key}: p={value['t_pvalue']:.4f} ({sig})")
    
    # Análisis de escalabilidad
    scalability_df = engine.run_scalability_analysis(max_size=10)
    console.print("\n[bold]Escalabilidad:[/bold]")
    console.print(scalability_df.to_string())
