"""
Métricas financieras para evaluación de carteras.

Proporciona métricas estándar de la industria para evaluar
el rendimiento y riesgo de las carteras optimizadas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from src.data.data_engine import MarketData
    from src.optimization.quantum_solver import SolverResult

console = Console()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PortfolioMetrics:
    """Métricas de una cartera."""
    
    # Retorno y Riesgo
    expected_return: float          # Retorno esperado anualizado
    volatility: float               # Volatilidad anualizada (std)
    variance: float                 # Varianza
    
    # Métricas ajustadas por riesgo
    sharpe_ratio: float             # (return - rf) / volatility
    sortino_ratio: float            # (return - rf) / downside_volatility
    
    # Riesgo de cola
    var_95: float                   # Value at Risk 95%
    cvar_95: float                  # Conditional VaR (Expected Shortfall)
    max_drawdown: float             # Máxima caída desde pico
    
    # Diversificación
    n_assets: int                   # Número de activos
    concentration: float            # Herfindahl-Hirschman Index
    
    # Metadatos
    risk_free_rate: float = 0.035  # Letras del Tesoro España
    
    def to_dict(self) -> dict:
        """Convertir a diccionario."""
        return {
            'expected_return': self.expected_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'max_drawdown': self.max_drawdown,
            'n_assets': self.n_assets,
            'concentration': self.concentration,
        }
    
    def is_acceptable(self) -> bool:
        """Verificar si la cartera cumple criterios mínimos."""
        return (
            self.sharpe_ratio > 0 and
            self.n_assets >= 2 and
            self.max_drawdown > -0.5  # No más de 50% drawdown
        )


# =============================================================================
# Metrics Calculator
# =============================================================================

class MetricsCalculator:
    """
    Calculadora de métricas financieras.
    
    Implementa las métricas estándar usadas en la industria
    para evaluar carteras de inversión.
    """
    
    def __init__(self, risk_free_rate: float = 0.035):
        """
        Inicializar calculadora.
        
        Args:
            risk_free_rate: Tasa libre de riesgo (default: 3.5% Letras del Tesoro)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_portfolio_metrics(
        self,
        selection: list[int],
        market_data: MarketData,
        weights: list[float] | None = None,
    ) -> PortfolioMetrics:
        """
        Calcular métricas completas de una cartera.
        
        Args:
            selection: Vector binario de selección
            market_data: Datos de mercado
            weights: Pesos de cada activo (None = equiponderado)
        
        Returns:
            PortfolioMetrics con todas las métricas
        """
        # Activos seleccionados
        selected_idx = [i for i, s in enumerate(selection) if s == 1]
        n_selected = len(selected_idx)
        
        if n_selected == 0:
            return self._empty_portfolio_metrics()
        
        # Pesos (equiponderado por defecto)
        if weights is None:
            weights = np.array([1.0 / n_selected if s == 1 else 0 for s in selection])
        else:
            weights = np.array(weights)
        
        # Extraer datos para activos seleccionados
        mu = market_data.mu.values
        sigma = market_data.sigma.values
        returns = market_data.returns.values
        
        # Retorno esperado de la cartera
        portfolio_return = np.dot(weights, mu)
        
        # Volatilidad de la cartera
        portfolio_variance = weights @ sigma @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe Ratio
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Sortino Ratio (solo volatilidad negativa)
        portfolio_returns = returns @ weights
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else portfolio_volatility
        sortino = (portfolio_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # VaR y CVaR (95%)
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252) if len(portfolio_returns) > 0 else var_95
        
        # Max Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0
        
        # Concentración (HHI)
        active_weights = weights[weights > 0]
        hhi = np.sum(active_weights ** 2)
        
        return PortfolioMetrics(
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            variance=portfolio_variance,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            n_assets=n_selected,
            concentration=hhi,
            risk_free_rate=self.risk_free_rate,
        )
    
    def _empty_portfolio_metrics(self) -> PortfolioMetrics:
        """Métricas para cartera vacía."""
        return PortfolioMetrics(
            expected_return=0,
            volatility=0,
            variance=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            var_95=0,
            cvar_95=0,
            max_drawdown=0,
            n_assets=0,
            concentration=0,
        )
    
    def compare_portfolios(
        self,
        portfolios: dict[str, list[int]],
        market_data: MarketData,
    ) -> pd.DataFrame:
        """
        Comparar múltiples carteras.
        
        Args:
            portfolios: Diccionario {nombre: selección}
            market_data: Datos de mercado
        
        Returns:
            DataFrame con métricas comparativas
        """
        results = []
        
        for name, selection in portfolios.items():
            metrics = self.calculate_portfolio_metrics(selection, market_data)
            row = {'portfolio': name, **metrics.to_dict()}
            results.append(row)
        
        return pd.DataFrame(results)
    
    def calculate_efficient_frontier(
        self,
        market_data: MarketData,
        n_points: int = 50,
    ) -> pd.DataFrame:
        """
        Calcular frontera eficiente para referencia.
        
        Args:
            market_data: Datos de mercado
            n_points: Número de puntos en la frontera
        
        Returns:
            DataFrame con puntos de la frontera eficiente
        """
        from scipy.optimize import minimize
        
        n = market_data.n_assets
        mu = market_data.mu.values
        sigma = market_data.sigma.values
        
        # Rango de retornos objetivo
        min_ret = mu.min()
        max_ret = mu.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier = []
        
        for target in target_returns:
            # Minimizar varianza sujeto a retorno objetivo
            def objective(w):
                return w @ sigma @ w
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Pesos suman 1
                {'type': 'eq', 'fun': lambda w: w @ mu - target},  # Retorno objetivo
            ]
            
            bounds = [(0, 1)] * n
            x0 = np.ones(n) / n
            
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                vol = np.sqrt(result.fun)
                frontier.append({
                    'return': target,
                    'volatility': vol,
                    'sharpe': (target - self.risk_free_rate) / vol if vol > 0 else 0,
                })
        
        return pd.DataFrame(frontier)


# =============================================================================
# Display Functions
# =============================================================================

def display_portfolio_metrics(
    metrics: PortfolioMetrics,
    portfolio_name: str = "Cartera",
) -> None:
    """Mostrar métricas de forma visual."""
    table = Table(title=f"📊 Métricas: {portfolio_name}")
    
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", justify="right")
    table.add_column("Interpretación", style="dim")
    
    # Retorno
    ret_style = "green" if metrics.expected_return > 0 else "red"
    table.add_row(
        "Retorno Esperado",
        f"[{ret_style}]{metrics.expected_return:.2%}[/{ret_style}]",
        "Anualizado",
    )
    
    # Volatilidad
    vol_style = "green" if metrics.volatility < 0.20 else "yellow" if metrics.volatility < 0.35 else "red"
    table.add_row(
        "Volatilidad",
        f"[{vol_style}]{metrics.volatility:.2%}[/{vol_style}]",
        "<20% bajo, >35% alto",
    )
    
    # Sharpe
    sharpe_style = "green" if metrics.sharpe_ratio > 1 else "yellow" if metrics.sharpe_ratio > 0.5 else "red"
    table.add_row(
        "Sharpe Ratio",
        f"[{sharpe_style}]{metrics.sharpe_ratio:.2f}[/{sharpe_style}]",
        ">1 excelente, >0.5 bueno",
    )
    
    # Sortino
    table.add_row(
        "Sortino Ratio",
        f"{metrics.sortino_ratio:.2f}",
        "Ajustado por riesgo bajista",
    )
    
    # VaR
    table.add_row(
        "VaR 95%",
        f"[red]{metrics.var_95:.2%}[/red]",
        "Pérdida máxima esperada (95%)",
    )
    
    # CVaR
    table.add_row(
        "CVaR 95%",
        f"[red]{metrics.cvar_95:.2%}[/red]",
        "Pérdida esperada en cola",
    )
    
    # Max Drawdown
    dd_style = "green" if metrics.max_drawdown > -0.15 else "yellow" if metrics.max_drawdown > -0.30 else "red"
    table.add_row(
        "Max Drawdown",
        f"[{dd_style}]{metrics.max_drawdown:.2%}[/{dd_style}]",
        "Máxima caída histórica",
    )
    
    # Diversificación
    table.add_row(
        "Activos",
        str(metrics.n_assets),
        "",
    )
    
    div_style = "green" if metrics.concentration < 0.25 else "yellow" if metrics.concentration < 0.5 else "red"
    table.add_row(
        "Concentración (HHI)",
        f"[{div_style}]{metrics.concentration:.2%}[/{div_style}]",
        "<25% bien diversificado",
    )
    
    console.print(table)


def display_portfolio_comparison(comparison_df: pd.DataFrame) -> None:
    """Mostrar comparación de carteras."""
    table = Table(title="📊 Comparación de Carteras")
    
    table.add_column("Cartera", style="cyan")
    table.add_column("Retorno", justify="right")
    table.add_column("Volatilidad", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Activos", justify="center")
    
    for _, row in comparison_df.iterrows():
        sharpe_style = "green" if row['sharpe_ratio'] > 1 else "yellow" if row['sharpe_ratio'] > 0.5 else "red"
        
        table.add_row(
            row['portfolio'],
            f"{row['expected_return']:.2%}",
            f"{row['volatility']:.2%}",
            f"[{sharpe_style}]{row['sharpe_ratio']:.2f}[/{sharpe_style}]",
            f"{row['max_drawdown']:.2%}",
            str(row['n_assets']),
        )
    
    console.print(table)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Demo con datos sintéticos
    from src.data.data_engine import DataEngine, MarketData
    import numpy as np
    
    # Generar datos de ejemplo
    tickers = ["SAN.MC", "ITX.MC", "IBE.MC", "TEF.MC"]
    np.random.seed(42)
    
    n = len(tickers)
    mu = pd.Series(np.random.uniform(0.05, 0.12, n), index=tickers)
    
    A = np.random.randn(n, n) * 0.1
    sigma_arr = A @ A.T + np.eye(n) * 0.03
    sigma = pd.DataFrame(sigma_arr, index=tickers, columns=tickers)
    
    # Crear retornos históricos sintéticos
    returns = pd.DataFrame(
        np.random.multivariate_normal(mu.values / 252, sigma.values / 252, size=500),
        columns=tickers
    )
    
    market_data = MarketData(
        tickers=tickers,
        prices=pd.DataFrame(),  # No necesario para demo
        returns=returns,
        mu=mu,
        sigma=sigma,
        metadata={}
    )
    
    # Calcular métricas para diferentes carteras
    calculator = MetricsCalculator(risk_free_rate=0.035)
    
    portfolios = {
        "Conservadora": [1, 0, 1, 1],  # Sin ITX
        "Agresiva": [0, 1, 0, 0],       # Solo ITX
        "Equilibrada": [1, 1, 1, 0],    # Sin TEF
        "Completa": [1, 1, 1, 1],       # Todos
    }
    
    # Comparar
    console.rule("[bold cyan]📊 Comparación de Carteras[/bold cyan]")
    comparison = calculator.compare_portfolios(portfolios, market_data)
    display_portfolio_comparison(comparison)
    
    # Mostrar detalle de la mejor
    best_portfolio = comparison.loc[comparison['sharpe_ratio'].idxmax(), 'portfolio']
    best_selection = portfolios[best_portfolio]
    best_metrics = calculator.calculate_portfolio_metrics(best_selection, market_data)
    
    console.print(f"\n[bold green]Mejor cartera: {best_portfolio}[/bold green]")
    display_portfolio_metrics(best_metrics, best_portfolio)
