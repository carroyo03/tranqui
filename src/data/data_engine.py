"""
Motor de datos para el pipeline de optimización de carteras.

Gestiona la adquisición, validación, almacenamiento y transformación
de datos financieros utilizando DuckDB como capa analítica.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from config.settings import DataSettings

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)
console = Console()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MarketData:
    """
    Contenedor para datos de mercado procesados.
    
    Attributes:
        tickers: Lista de símbolos de activos
        prices: Serie temporal de precios ajustados
        returns: Retornos logarítmicos
        mu: Retornos esperados anualizados
        sigma: Matriz de covarianza anualizada
        metadata: Información adicional sobre los datos
    """
    
    tickers: list[str]
    prices: pd.DataFrame
    returns: pd.DataFrame
    mu: pd.Series
    sigma: pd.DataFrame
    metadata: dict
    
    def __post_init__(self):
        """Validar consistencia de datos."""
        assert len(self.tickers) == len(self.mu), "Mismatch entre tickers y mu"
        assert self.sigma.shape[0] == self.sigma.shape[1] == len(self.tickers), "Sigma debe ser cuadrada"
    
    @property
    def n_assets(self) -> int:
        """Número de activos."""
        return len(self.tickers)
    
    @property
    def n_observations(self) -> int:
        """Número de observaciones temporales."""
        return len(self.prices)
    
    @property
    def correlation_matrix(self) -> pd.DataFrame:
        """Matriz de correlación."""
        std = np.sqrt(np.diag(self.sigma))
        corr = self.sigma / np.outer(std, std)
        return pd.DataFrame(corr, index=self.tickers, columns=self.tickers)
    
    def get_sharpe_ratios(self, risk_free_rate: float = 0.035) -> pd.Series:
        """
        Calcular Sharpe Ratio para cada activo.
        
        Args:
            risk_free_rate: Tasa libre de riesgo (default: 3.5% Letras del Tesoro España)
        
        Returns:
            Sharpe Ratio por activo
        """
        volatility = np.sqrt(np.diag(self.sigma))
        sharpe = (self.mu - risk_free_rate) / volatility
        return pd.Series(sharpe, index=self.tickers, name="Sharpe Ratio")
    
    def to_dict(self) -> dict:
        """Serializar a diccionario."""
        return {
            "tickers": self.tickers,
            "mu": self.mu.to_dict(),
            "sigma": self.sigma.to_dict(),
            "metadata": self.metadata,
        }


# =============================================================================
# Data Engine
# =============================================================================

class DataEngine:
    """
    Motor de datos con DuckDB como capa analítica.
    
    Utiliza DuckDB para:
    - Validación y limpieza de datos
    - Cálculo eficiente de estadísticas rolling
    - Detección de outliers
    - Análisis de calidad de datos
    """
    
    def __init__(self, settings: DataSettings | None = None):
        """
        Inicializar motor de datos.
        
        Args:
            settings: Configuración de datos (usa defaults si no se proporciona)
        """
        if settings is None:
            from config.settings import DataSettings
            settings = DataSettings()
        
        self.settings = settings
        self.con = duckdb.connect(":memory:")
        self._setup_database()
        
        logger.info("DataEngine inicializado con DuckDB")
    
    def _setup_database(self) -> None:
        """Configurar esquema de base de datos."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                date DATE,
                ticker VARCHAR,
                adj_close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (date, ticker)
            )
        """)
        
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS data_quality (
                ticker VARCHAR PRIMARY KEY,
                total_rows INTEGER,
                missing_count INTEGER,
                missing_pct DOUBLE,
                min_date DATE,
                max_date DATE,
                outlier_count INTEGER
            )
        """)
    
    def fetch_data(
        self,
        tickers: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        validate: bool = True,
    ) -> MarketData:
        """
        Obtener y procesar datos de mercado.
        
        Args:
            tickers: Lista de símbolos
            start_date: Fecha inicio (YYYY-MM-DD)
            end_date: Fecha fin (None = hoy)
            validate: Si ejecutar validaciones de calidad
        
        Returns:
            MarketData con datos procesados
        """
        start = start_date or self.settings.start_date
        end = end_date or datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Descargando datos para {tickers} desde {start} hasta {end}")
        
        # 1. Descargar de Yahoo Finance
        raw_data = self._download_from_yahoo(tickers, start, end)
        
        # 2. Ingestar en DuckDB
        self._ingest_to_duckdb(raw_data, tickers)
        
        # 3. Validar calidad (si está habilitado)
        if validate:
            quality_report = self._validate_data_quality(tickers)
            self._display_quality_report(quality_report)
        
        # 4. Calcular métricas financieras
        prices = self._get_clean_prices(tickers)
        
        # Asegurar que solo usamos tickers que tienen datos
        valid_tickers = prices.columns.tolist()
        if len(valid_tickers) < len(tickers):
            missing = set(tickers) - set(valid_tickers)
            console.print(f"[yellow]⚠ Descartando activos sin datos: {missing}[/yellow]")
        
        returns, mu, sigma = self._calculate_financial_metrics(prices)
        
        # 5. Construir metadata
        metadata = {
            "start_date": start,
            "end_date": end,
            "fetch_timestamp": datetime.now().isoformat(),
            "trading_days": self.settings.trading_days_per_year,
            "source": "Yahoo Finance",
            "quality_validated": validate,
        }
        
        return MarketData(
            tickers=valid_tickers,
            prices=prices,
            returns=returns,
            mu=mu,
            sigma=sigma,
            metadata=metadata,
        )
    
    def _download_from_yahoo(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Descargar datos de Yahoo Finance."""
        console.print(f"[cyan]📡 Descargando datos de Yahoo Finance...[/cyan]")
        
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,  # Usar precios ajustados
            )
            
            # Manejar estructura de datos (single vs multiple tickers)
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
            
            prices = prices.dropna(how="all")
            
            console.print(f"[green]✅ Descargados {len(prices)} días de datos[/green]")
            return prices
            
        except Exception as e:
            logger.error(f"Error descargando datos: {e}")
            raise RuntimeError(f"No se pudieron descargar datos para {tickers}: {e}")
    
    def _ingest_to_duckdb(self, prices: pd.DataFrame, tickers: list[str]) -> None:
        """Ingestar datos en DuckDB para análisis."""
        # Limpiar tabla existente
        self.con.execute("DELETE FROM price_data")
        
        # Transformar a formato largo para DuckDB
        df_long = prices.reset_index().melt(
            id_vars=["Date"],
            var_name="ticker",
            value_name="adj_close",
        )
        df_long = df_long.rename(columns={"Date": "date"})
        df_long = df_long.dropna(subset=["adj_close"])
        
        # Insertar en DuckDB
        self.con.execute("INSERT INTO price_data SELECT date, ticker, adj_close, NULL FROM df_long")
        
        # Verificar ingesta
        count = self.con.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
        logger.info(f"Ingestados {count} registros en DuckDB")
    
    def _validate_data_quality(self, tickers: list[str]) -> pd.DataFrame:
        """
        Ejecutar validaciones de calidad de datos usando SQL.
        
        Esta es la justificación real de DuckDB en el proyecto:
        análisis de calidad de datos con SQL expresivo.
        """
        # Limpiar tabla de calidad
        self.con.execute("DELETE FROM data_quality")
        
        # Calcular métricas de calidad por ticker
        quality_query = """
            INSERT INTO data_quality
            SELECT 
                ticker,
                COUNT(*) as total_rows,
                COUNT(*) - COUNT(adj_close) as missing_count,
                (COUNT(*) - COUNT(adj_close))::DOUBLE / COUNT(*) * 100 as missing_pct,
                MIN(date) as min_date,
                MAX(date) as max_date,
                -- Detectar outliers usando IQR
                SUM(CASE 
                    WHEN adj_close < q1 - 1.5 * iqr OR adj_close > q3 + 1.5 * iqr 
                    THEN 1 ELSE 0 
                END) as outlier_count
            FROM price_data
            LEFT JOIN (
                SELECT 
                    ticker as t,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY adj_close) as q1,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY adj_close) as q3,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY adj_close) - 
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY adj_close) as iqr
                FROM price_data
                GROUP BY ticker
            ) quartiles ON price_data.ticker = quartiles.t
            GROUP BY ticker
        """
        self.con.execute(quality_query)
        
        return self.con.execute("SELECT * FROM data_quality").fetchdf()
    
    def _display_quality_report(self, quality_df: pd.DataFrame) -> None:
        """Mostrar reporte de calidad de datos."""
        table = Table(
            title="📊 Reporte de Calidad de Datos",
            show_header=True,
            header_style="bold cyan",
        )
        
        table.add_column("Ticker", style="yellow")
        table.add_column("Registros", justify="right")
        table.add_column("Missing %", justify="right")
        table.add_column("Outliers", justify="right")
        table.add_column("Rango", justify="center")
        
        for _, row in quality_df.iterrows():
            missing_style = "red" if row["missing_pct"] > 5 else "green"
            outlier_style = "yellow" if row["outlier_count"] > 10 else "green"
            
            table.add_row(
                row["ticker"],
                str(row["total_rows"]),
                f"[{missing_style}]{row['missing_pct']:.2f}%[/{missing_style}]",
                f"[{outlier_style}]{row['outlier_count']}[/{outlier_style}]",
                f"{row['min_date']} → {row['max_date']}",
            )
        
        console.print(table)
    
    def _get_clean_prices(self, tickers: list[str]) -> pd.DataFrame:
        """Obtener precios limpios pivotados desde DuckDB."""
        query = """
            PIVOT price_data
            ON ticker
            USING FIRST(adj_close)
            GROUP BY date
            ORDER BY date
        """
        df = self.con.execute(query).fetchdf()
        df = df.set_index("date")
        
        # Asegurar orden de columnas según tickers originales
        available_cols = [t for t in tickers if t in df.columns]
        return df[available_cols].dropna()
    
    def _calculate_financial_metrics(
        self,
        prices: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Calcular retornos, media y covarianza.
        
        Returns:
            Tuple de (returns, mu, sigma)
        """
        # Retornos logarítmicos (más estables que simples)
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        # Anualizar
        trading_days = self.settings.trading_days_per_year
        mu = log_returns.mean() * trading_days
        sigma = log_returns.cov() * trading_days
        
        # Aplicar shrinkage a la covarianza para mejor estimación
        sigma = self._ledoit_wolf_shrinkage(sigma.values, log_returns.values)
        sigma = pd.DataFrame(sigma, index=prices.columns, columns=prices.columns)
        
        return log_returns, mu, sigma
    
    def _ledoit_wolf_shrinkage(
        self,
        sigma: np.ndarray,
        returns: np.ndarray,
    ) -> np.ndarray:
        """
        Aplicar shrinkage de Ledoit-Wolf a la matriz de covarianza.
        
        Mejora la estimación para matrices mal condicionadas
        (típico cuando n_observations < n_assets * 10).
        """
        n, p = returns.shape
        
        if n < p * 2:
            logger.warning(
                f"Pocas observaciones ({n}) para {p} activos. "
                "Aplicando shrinkage de Ledoit-Wolf."
            )
        
        # Target: diagonal (varianzas individuales)
        mu_target = np.trace(sigma) / p
        delta = sigma.copy()
        np.fill_diagonal(delta, delta.diagonal() - mu_target)
        
        # Intensidad de shrinkage
        delta_sq = (delta ** 2).sum()
        
        # Shrinkage hacia matriz diagonal
        shrinkage_intensity = min(1.0, max(0.0, delta_sq / (n * delta_sq + 1e-10)))
        
        # Aplicar shrinkage
        target = np.eye(p) * mu_target
        sigma_shrunk = shrinkage_intensity * target + (1 - shrinkage_intensity) * sigma
        
        return sigma_shrunk
    
    def calculate_rolling_metrics(
        self,
        tickers: list[str],
        window: int = 60,
    ) -> pd.DataFrame:
        """
        Calcular métricas rolling usando DuckDB window functions.
        
        Demuestra uso avanzado de SQL para análisis de series temporales.
        """
        query = f"""
            SELECT 
                date,
                ticker,
                adj_close,
                AVG(adj_close) OVER w as rolling_mean,
                STDDEV(adj_close) OVER w as rolling_std,
                (adj_close - AVG(adj_close) OVER w) / NULLIF(STDDEV(adj_close) OVER w, 0) as z_score
            FROM price_data
            WHERE ticker IN ({','.join([f"'{t}'" for t in tickers])})
            WINDOW w AS (
                PARTITION BY ticker 
                ORDER BY date 
                ROWS BETWEEN {window} PRECEDING AND CURRENT ROW
            )
            ORDER BY date, ticker
        """
        return self.con.execute(query).fetchdf()
    
    def get_correlation_analysis(self) -> pd.DataFrame:
        """
        Análisis de correlación entre activos usando DuckDB.
        """
        # Calcular retornos diarios
        query = """
            WITH daily_returns AS (
                SELECT 
                    date,
                    ticker,
                    (adj_close - LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date)) 
                    / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) as daily_return
                FROM price_data
            )
            SELECT 
                a.ticker as ticker_a,
                b.ticker as ticker_b,
                CORR(a.daily_return, b.daily_return) as correlation
            FROM daily_returns a
            JOIN daily_returns b ON a.date = b.date AND a.ticker < b.ticker
            WHERE a.daily_return IS NOT NULL AND b.daily_return IS NOT NULL
            GROUP BY a.ticker, b.ticker
            ORDER BY ABS(correlation) DESC
        """
        return self.con.execute(query).fetchdf()
    
    def close(self) -> None:
        """Cerrar conexión a DuckDB."""
        self.con.close()
        logger.info("Conexión DuckDB cerrada")


# =============================================================================
# Convenience Functions
# =============================================================================

def get_data(
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Función de conveniencia para obtener datos.
    
    Mantiene compatibilidad con el código original.
    
    Args:
        tickers: Lista de símbolos
        start_date: Fecha inicio
        end_date: Fecha fin
    
    Returns:
        Tuple de (mu, sigma)
    """
    engine = DataEngine()
    try:
        data = engine.fetch_data(tickers, start_date, end_date)
        return data.mu, data.sigma
    finally:
        engine.close()


# =============================================================================
# Display Functions
# =============================================================================

def display_market_data(data: MarketData) -> None:
    """Mostrar datos de mercado de forma visual."""
    console.rule("[bold cyan]📈 Datos de Mercado[/bold cyan]")
    
    # Tabla de retornos esperados
    mu_table = Table(
        title="Retornos Esperados Anualizados",
        show_header=True,
        header_style="bold green",
    )
    mu_table.add_column("Ticker", style="yellow")
    mu_table.add_column("Retorno", justify="right")
    mu_table.add_column("Volatilidad", justify="right")
    mu_table.add_column("Sharpe", justify="right")
    
    sharpe = data.get_sharpe_ratios()
    vol = np.sqrt(np.diag(data.sigma))
    
    for i, ticker in enumerate(data.tickers):
        ret_style = "green" if data.mu[ticker] > 0 else "red"
        sharpe_style = "green" if sharpe[ticker] > 0.5 else "yellow" if sharpe[ticker] > 0 else "red"
        
        mu_table.add_row(
            ticker,
            f"[{ret_style}]{data.mu[ticker]:+.2%}[/{ret_style}]",
            f"{vol[i]:.2%}",
            f"[{sharpe_style}]{sharpe[ticker]:.2f}[/{sharpe_style}]",
        )
    
    console.print(mu_table)
    
    # Tabla de covarianza
    sigma_table = Table(
        title="Matriz de Covarianza Anualizada",
        show_header=True,
        header_style="bold cyan",
    )
    sigma_table.add_column("", style="yellow")
    for ticker in data.tickers:
        sigma_table.add_column(ticker, justify="center")
    
    for ticker in data.tickers:
        row = [ticker]
        for other in data.tickers:
            val = data.sigma.loc[ticker, other]
            row.append(f"{val:.4f}")
        sigma_table.add_row(*row)
    
    console.print(sigma_table)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Demo con activos españoles
    tickers = ["SAN.MC", "ITX.MC", "IBE.MC", "TEF.MC"]
    
    engine = DataEngine()
    data = engine.fetch_data(tickers)
    
    display_market_data(data)
    
    # Mostrar análisis de correlación
    console.rule("[bold magenta]🔗 Análisis de Correlación[/bold magenta]")
    corr_df = engine.get_correlation_analysis()
    console.print(corr_df.to_string())
    
    engine.close()
