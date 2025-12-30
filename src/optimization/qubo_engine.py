"""
Motor QUBO para optimización de carteras.

Transforma el problema de optimización de carteras Markowitz
en formato QUBO (Quadratic Unconstrained Binary Optimization)
compatible con algoritmos cuánticos como QAOA.

Formulación matemática:
    min_x  λ * x^T Σ x - (1-λ) * μ^T x + penalties
    
    donde:
    - x ∈ {0,1}^n: vector de selección binaria
    - Σ: matriz de covarianza (riesgo)
    - μ: retornos esperados
    - λ: aversión al riesgo [0,1]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from src.data.data_engine import MarketData

logger = logging.getLogger(__name__)
console = Console()


# =============================================================================
# Enums y Tipos
# =============================================================================

class ConstraintType(Enum):
    """Tipos de restricciones soportadas."""
    MIN_ASSETS = "min_assets"          # Mínimo de activos a seleccionar
    MAX_ASSETS = "max_assets"          # Máximo de activos a seleccionar
    EXACT_ASSETS = "exact_assets"      # Número exacto de activos
    SECTOR_LIMIT = "sector_limit"      # Límite por sector
    BUDGET = "budget"                  # Restricción de presupuesto


@dataclass
class Constraint:
    """Definición de una restricción."""
    
    type: ConstraintType
    value: int | float
    penalty: float = 1000.0
    assets: list[int] | None = None  # Índices de activos afectados (para sector)
    
    def __post_init__(self):
        if self.penalty <= 0:
            raise ValueError("La penalización debe ser positiva")


@dataclass
class QUBOResult:
    """Resultado de la transformación QUBO."""
    
    Q: np.ndarray                      # Matriz QUBO
    tickers: list[str]                 # Símbolos de activos
    risk_aversion: float               # Parámetro λ usado
    constraints: list[Constraint]      # Restricciones aplicadas
    metadata: dict = field(default_factory=dict)
    
    @property
    def n_qubits(self) -> int:
        """Número de qubits necesarios."""
        return len(self.Q)
    
    @property
    def sparsity(self) -> float:
        """Porcentaje de elementos no-cero."""
        non_zero = np.count_nonzero(self.Q)
        total = self.Q.size
        return non_zero / total
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluar la función objetivo para una solución.
        
        Args:
            x: Vector binario de selección
        
        Returns:
            Valor de la función objetivo
        """
        x = np.array(x).flatten()
        return float(x @ self.Q @ x)
    
    def get_selected_assets(self, x: np.ndarray) -> list[str]:
        """Obtener tickers seleccionados dado un vector solución."""
        return [t for t, selected in zip(self.tickers, x) if selected == 1]


# =============================================================================
# QUBO Engine
# =============================================================================

class QUBOEngine:
    """
    Motor de transformación Markowitz → QUBO.
    
    Soporta múltiples tipos de restricciones mediante penalizaciones
    cuadráticas añadidas a la función objetivo.
    """
    
    def __init__(self, default_penalty: float = 1000.0):
        """
        Inicializar motor QUBO.
        
        Args:
            default_penalty: Penalización por defecto para restricciones
        """
        self.default_penalty = default_penalty
        logger.info(f"QUBOEngine inicializado (penalty={default_penalty})")
    
    def to_qubo(
        self,
        mu: pd.Series | np.ndarray,
        sigma: pd.DataFrame | np.ndarray,
        risk_aversion: float = 0.5,
        constraints: list[Constraint] | None = None,
        normalize: bool = True,
    ) -> QUBOResult:
        """
        Transformar problema de optimización a formato QUBO.
        
        Args:
            mu: Retornos esperados
            sigma: Matriz de covarianza
            risk_aversion: Parámetro λ ∈ [0,1] (0=max retorno, 1=min riesgo)
            constraints: Lista de restricciones
            normalize: Si normalizar los datos antes de transformar
        
        Returns:
            QUBOResult con la matriz Q y metadata
        """
        # Validar parámetros
        self._validate_inputs(mu, sigma, risk_aversion)
        
        # Extraer datos
        if isinstance(mu, pd.Series):
            tickers = list(mu.index)
            mu_arr = mu.values.astype(float)
        else:
            tickers = [f"Asset_{i}" for i in range(len(mu))]
            mu_arr = np.array(mu, dtype=float)
        
        if isinstance(sigma, pd.DataFrame):
            sigma_arr = sigma.values.astype(float)
        else:
            sigma_arr = np.array(sigma, dtype=float)
        
        n = len(mu_arr)
        
        # Normalizar si es necesario (mejora convergencia de QAOA)
        if normalize:
            mu_arr, sigma_arr, scale_factors = self._normalize(mu_arr, sigma_arr)
        else:
            scale_factors = {"mu_scale": 1.0, "sigma_scale": 1.0}
        
        # 1. Construir QUBO base (Markowitz)
        Q = self._build_base_qubo(mu_arr, sigma_arr, risk_aversion)
        
        # 2. Añadir penalizaciones por restricciones
        constraints = constraints or []
        for constraint in constraints:
            Q = self._add_constraint_penalty(Q, constraint, n)
        
        # Metadata
        metadata = {
            "normalized": normalize,
            "scale_factors": scale_factors,
            "condition_number": np.linalg.cond(Q),
            "eigenvalue_range": (np.min(np.linalg.eigvalsh(Q)), np.max(np.linalg.eigvalsh(Q))),
        }
        
        return QUBOResult(
            Q=Q,
            tickers=tickers,
            risk_aversion=risk_aversion,
            constraints=constraints,
            metadata=metadata,
        )
    
    def _validate_inputs(
        self,
        mu: pd.Series | np.ndarray,
        sigma: pd.DataFrame | np.ndarray,
        risk_aversion: float,
    ) -> None:
        """Validar entradas."""
        n_mu = len(mu)
        
        if isinstance(sigma, pd.DataFrame):
            n_sigma = sigma.shape[0]
        else:
            n_sigma = sigma.shape[0]
        
        if n_mu != n_sigma:
            raise ValueError(f"Dimensiones inconsistentes: mu={n_mu}, sigma={n_sigma}")
        
        if not 0 <= risk_aversion <= 1:
            raise ValueError(f"risk_aversion debe estar en [0,1], recibido: {risk_aversion}")
        
        # Verificar que sigma es semidefinida positiva
        if isinstance(sigma, pd.DataFrame):
            sigma_arr = sigma.values
        else:
            sigma_arr = sigma
        
        eigenvalues = np.linalg.eigvalsh(sigma_arr)
        if np.any(eigenvalues < -1e-10):
            logger.warning(
                f"Matriz sigma no es semidefinida positiva. "
                f"Eigenvalor mínimo: {eigenvalues.min():.6f}"
            )
    
    def _normalize(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Normalizar datos para mejorar convergencia.
        
        Escala mu y sigma para que tengan magnitudes comparables,
        lo cual mejora el rendimiento de optimizadores como COBYLA.
        """
        mu_scale = np.abs(mu).max() if np.abs(mu).max() > 0 else 1.0
        sigma_scale = np.abs(sigma).max() if np.abs(sigma).max() > 0 else 1.0
        
        # Escalar para que ambos tengan magnitud ~1
        mu_normalized = mu / mu_scale
        sigma_normalized = sigma / sigma_scale
        
        return mu_normalized, sigma_normalized, {
            "mu_scale": mu_scale,
            "sigma_scale": sigma_scale,
        }
    
    def _build_base_qubo(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        risk_aversion: float,
    ) -> np.ndarray:
        """
        Construir matriz QUBO base sin restricciones.
        
        QUBO para Markowitz:
            Q_ij = λ * Σ_ij          (fuera de diagonal)
            Q_ii = λ * Σ_ii - (1-λ) * μ_i  (diagonal)
        
        Esto viene de expandir:
            λ * x^T Σ x - (1-λ) * μ^T x
        
        donde para variables binarias x_i^2 = x_i
        """
        n = len(mu)
        
        # Término cuadrático (riesgo): λ * Σ
        Q = risk_aversion * sigma.copy()
        
        # Término lineal (retorno): -(1-λ) * μ
        # Como x_i^2 = x_i para binarios, los términos lineales van a la diagonal
        linear_terms = -(1 - risk_aversion) * mu
        np.fill_diagonal(Q, np.diag(Q) + linear_terms)
        
        return Q
    
    def _add_constraint_penalty(
        self,
        Q: np.ndarray,
        constraint: Constraint,
        n: int,
    ) -> np.ndarray:
        """
        Añadir penalización por restricción a la matriz QUBO.
        
        Las restricciones se convierten en penalizaciones cuadráticas:
            penalty * (sum(x) - target)^2
        
        Expandiendo:
            = penalty * (sum_i x_i - target)^2
            = penalty * (sum_i sum_j x_i x_j - 2*target*sum_i x_i + target^2)
            = penalty * (sum_i sum_j x_i x_j - 2*target*sum_i x_i) + constante
        """
        Q = Q.copy()
        p = constraint.penalty
        
        if constraint.type == ConstraintType.MIN_ASSETS:
            # Penalizar si sum(x) < min_assets
            # Usamos: penalty * max(0, min_assets - sum(x))^2
            # Aproximación: añadimos incentivo para seleccionar más activos
            target = constraint.value
            
            # Término que favorece seleccionar activos (reduce objetivo)
            # Esto no es una penalización exacta, pero incentiva diversificación
            for i in range(n):
                Q[i, i] -= p * (2 * target / n)
                for j in range(i + 1, n):
                    Q[i, j] += p / n
                    Q[j, i] += p / n
        
        elif constraint.type == ConstraintType.MAX_ASSETS:
            # Penalizar si sum(x) > max_assets
            target = constraint.value
            
            # Añadir penalización: penalty * (sum(x) - target)^2 si sum > target
            # Aproximación: penalizar selección excesiva
            for i in range(n):
                Q[i, i] += p * (1 - 2 * target / n)
                for j in range(i + 1, n):
                    Q[i, j] += p / n
                    Q[j, i] += p / n
        
        elif constraint.type == ConstraintType.EXACT_ASSETS:
            # Penalizar desviación: penalty * (sum(x) - k)^2
            k = constraint.value
            
            # Expandiendo (sum_i x_i - k)^2:
            # = sum_i sum_j x_i x_j - 2k * sum_i x_i + k^2
            # 
            # Q_ii += penalty * (1 - 2k)  [de x_i^2 = x_i y -2k*x_i]
            # Q_ij += penalty * 2         [de 2*x_i*x_j]
            for i in range(n):
                Q[i, i] += p * (1 - 2 * k)
                for j in range(i + 1, n):
                    Q[i, j] += p
                    Q[j, i] += p
        
        elif constraint.type == ConstraintType.SECTOR_LIMIT:
            # Limitar activos de un sector específico
            if constraint.assets is None:
                logger.warning("SECTOR_LIMIT requiere lista de assets")
                return Q
            
            sector_idx = constraint.assets
            limit = constraint.value
            
            # Penalizar si se seleccionan más de 'limit' activos del sector
            for i in sector_idx:
                Q[i, i] += p * (1 - 2 * limit)
                for j in sector_idx:
                    if i < j:
                        Q[i, j] += p
                        Q[j, i] += p
        
        return Q
    
    def analyze_qubo(self, result: QUBOResult) -> dict:
        """
        Analizar propiedades de la matriz QUBO.
        
        Útil para entender la dificultad del problema y ajustar parámetros.
        """
        Q = result.Q
        n = len(Q)
        
        # Análisis espectral
        eigenvalues = np.linalg.eigvalsh(Q)
        
        # Gap espectral (importante para QAOA)
        sorted_eig = np.sort(eigenvalues)
        spectral_gap = sorted_eig[1] - sorted_eig[0] if len(sorted_eig) > 1 else 0
        
        # Análisis de la diagonal (términos lineales efectivos)
        diagonal = np.diag(Q)
        
        # Análisis de conectividad (términos cuadráticos)
        off_diagonal = Q - np.diag(diagonal)
        connectivity = np.count_nonzero(off_diagonal) / (n * (n - 1))
        
        return {
            "n_qubits": n,
            "eigenvalue_min": eigenvalues.min(),
            "eigenvalue_max": eigenvalues.max(),
            "spectral_gap": spectral_gap,
            "condition_number": result.metadata.get("condition_number", np.linalg.cond(Q)),
            "diagonal_mean": diagonal.mean(),
            "diagonal_std": diagonal.std(),
            "connectivity": connectivity,
            "sparsity": result.sparsity,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def to_qubo(
    mu: pd.Series | np.ndarray,
    sigma: pd.DataFrame | np.ndarray,
    risk_aversion: float = 0.5,
    min_assets: int | None = None,
    max_assets: int | None = None,
) -> np.ndarray:
    """
    Función de conveniencia para compatibilidad con código original.
    
    Args:
        mu: Retornos esperados
        sigma: Matriz de covarianza
        risk_aversion: Parámetro de aversión al riesgo
        min_assets: Mínimo de activos (opcional)
        max_assets: Máximo de activos (opcional)
    
    Returns:
        Matriz QUBO como numpy array
    """
    constraints = []
    if min_assets is not None:
        constraints.append(Constraint(ConstraintType.MIN_ASSETS, min_assets))
    if max_assets is not None:
        constraints.append(Constraint(ConstraintType.MAX_ASSETS, max_assets))
    
    engine = QUBOEngine()
    result = engine.to_qubo(mu, sigma, risk_aversion, constraints)
    return result.Q


# =============================================================================
# Display Functions
# =============================================================================

def display_qubo(result: QUBOResult) -> None:
    """Mostrar matriz QUBO de forma visual."""
    console.rule("[bold cyan]⚛️ Matriz QUBO[/bold cyan]")
    
    # Info básica
    info_panel = Panel(
        f"[bold]Qubits:[/bold] {result.n_qubits}\n"
        f"[bold]Aversión al riesgo:[/bold] {result.risk_aversion}\n"
        f"[bold]Restricciones:[/bold] {len(result.constraints)}\n"
        f"[bold]Sparsity:[/bold] {result.sparsity:.2%}",
        title="Información QUBO",
        border_style="cyan",
    )
    console.print(info_panel)
    
    # Matriz (si es pequeña)
    if result.n_qubits <= 8:
        table = Table(title="Matriz Q", show_header=True, header_style="bold yellow")
        table.add_column("", style="yellow")
        for ticker in result.tickers:
            table.add_column(ticker[:6], justify="center")
        
        for i, ticker in enumerate(result.tickers):
            row = [ticker[:6]]
            for j in range(result.n_qubits):
                val = result.Q[i, j]
                style = "green" if val < 0 else "red" if val > 0 else "dim"
                row.append(f"[{style}]{val:+.3f}[/{style}]")
            table.add_row(*row)
        
        console.print(table)
    
    # Análisis
    engine = QUBOEngine()
    analysis = engine.analyze_qubo(result)
    
    analysis_table = Table(title="Análisis Espectral", show_header=True)
    analysis_table.add_column("Métrica", style="cyan")
    analysis_table.add_column("Valor", justify="right")
    
    for key, value in analysis.items():
        if isinstance(value, float):
            analysis_table.add_row(key, f"{value:.6f}")
        else:
            analysis_table.add_row(key, str(value))
    
    console.print(analysis_table)
    
    # Interpretación de la diagonal
    console.print("\n[bold]📊 Interpretación de la Diagonal:[/bold]")
    diagonal = np.diag(result.Q)
    for i, (ticker, d) in enumerate(zip(result.tickers, diagonal)):
        if d < 0:
            console.print(f"  [green]✓[/green] {ticker}: {d:+.4f} (favorecido)")
        else:
            console.print(f"  [red]✗[/red] {ticker}: {d:+.4f} (penalizado)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Demo con datos de ejemplo
    tickers = ["SAN.MC", "ITX.MC", "IBE.MC", "BTC-EUR"]
    
    # Simular datos
    np.random.seed(42)
    mu = pd.Series([0.08, 0.12, 0.06, 0.25], index=tickers)
    sigma = pd.DataFrame(
        [
            [0.04, 0.01, 0.02, 0.01],
            [0.01, 0.09, 0.01, 0.02],
            [0.02, 0.01, 0.03, 0.00],
            [0.01, 0.02, 0.00, 0.50],
        ],
        index=tickers,
        columns=tickers,
    )
    
    # Crear QUBO con restricciones
    engine = QUBOEngine()
    
    constraints = [
        Constraint(ConstraintType.MIN_ASSETS, 2, penalty=500),
        Constraint(ConstraintType.MAX_ASSETS, 3, penalty=500),
    ]
    
    result = engine.to_qubo(
        mu,
        sigma,
        risk_aversion=0.5,
        constraints=constraints,
        normalize=True,
    )
    
    display_qubo(result)
    
    # Evaluar soluciones candidatas
    console.rule("[bold magenta]🧪 Evaluación de Soluciones[/bold magenta]")
    
    test_solutions = [
        [1, 0, 1, 0],  # SAN + IBE (conservador)
        [1, 1, 0, 0],  # SAN + ITX
        [0, 1, 1, 0],  # ITX + IBE
        [1, 1, 1, 0],  # Sin crypto
        [1, 1, 1, 1],  # Todos
    ]
    
    for sol in test_solutions:
        value = result.evaluate(sol)
        selected = result.get_selected_assets(sol)
        console.print(f"{sol} → {value:+.4f} | {selected}")
