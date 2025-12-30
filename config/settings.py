"""
Configuración centralizada del proyecto.

Utiliza Pydantic Settings para gestión de configuración type-safe
con soporte para variables de entorno y archivos .env.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Rutas del Proyecto
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / ".cache"


# =============================================================================
# Configuración de Datos
# =============================================================================

class DataSettings(BaseSettings):
    """Configuración para el pipeline de datos."""
    
    model_config = SettingsConfigDict(
        env_prefix="DATA_",
        env_file=".env",
        extra="ignore",
    )
    
    # Período de datos históricos
    start_date: str = Field(default="2015-01-01", description="Fecha inicio datos históricos")
    end_date: str | None = Field(default=None, description="Fecha fin (None = hoy)")
    
    # Frecuencia
    frequency: Literal["1d", "1wk", "1mo"] = Field(default="1d", description="Frecuencia de datos")
    
    # Trading days por año (España/EU)
    trading_days_per_year: int = Field(default=252, ge=200, le=260)
    
    # Cache
    cache_enabled: bool = Field(default=True)
    cache_ttl_hours: int = Field(default=24, ge=1, le=168)


# =============================================================================
# Configuración de Optimización
# =============================================================================

class OptimizationSettings(BaseSettings):
    """Configuración para los solvers de optimización."""
    
    model_config = SettingsConfigDict(
        env_prefix="OPT_",
        env_file=".env",
        extra="ignore",
    )
    
    # QAOA Parameters
    qaoa_reps: int = Field(default=1, ge=1, le=5, description="Capas QAOA (p)")
    qaoa_max_iter: int = Field(default=100, ge=10, le=1000, description="Iteraciones COBYLA")
    qaoa_shots: int = Field(default=1024, ge=100, le=10000, description="Shots del sampler")
    
    # Classical solver
    classical_solver: Literal["GLPK_MI", "CBC", "SCIP", "SCIPY"] = Field(
        default="SCIPY",
        description="Solver clásico para baseline"
    )
    classical_timeout: float = Field(default=60.0, ge=1.0, le=3600.0)
    
    # QUBO constraints
    min_assets: int = Field(default=2, ge=1, description="Mínimo de activos en cartera")
    max_assets: int | None = Field(default=None, description="Máximo de activos (None = sin límite)")
    penalty_strength: float = Field(default=1000.0, ge=1.0, description="Fuerza de penalización")
    
    # Risk profiles
    risk_aversion_conservative: float = Field(default=0.8, ge=0.5, le=1.0)
    risk_aversion_balanced: float = Field(default=0.5, ge=0.3, le=0.7)
    risk_aversion_aggressive: float = Field(default=0.2, ge=0.0, le=0.4)


# =============================================================================
# Configuración de LLM
# =============================================================================

class LLMSettings(BaseSettings):
    """Configuración para el motor de explicaciones LLM."""
    
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        extra="ignore",
    )
    
    # API Configuration
    api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    api_base: str = Field(default="https://openrouter.ai/api/v1")
    
    # Model selection
    model: str = Field(
        default="anthropic/claude-3-haiku",
        description="Modelo LLM a utilizar"
    )
    fallback_model: str = Field(
        default="nvidia/nemotron-3-nano-30b-a3b:free",
        description="Modelo fallback gratuito"
    )
    
    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, ge=100, le=4000)
    
    # Language
    language: Literal["es", "en"] = Field(default="es", description="Idioma de las explicaciones")
    
    @field_validator("api_key", mode="before")
    @classmethod
    def check_api_key(cls, v):
        """Validar que la API key existe si se necesita."""
        if v is None or v == "":
            return None
        return v


# =============================================================================
# Configuración de Evaluación
# =============================================================================

class EvaluationSettings(BaseSettings):
    """Configuración para benchmarking y evaluación."""
    
    model_config = SettingsConfigDict(
        env_prefix="EVAL_",
        env_file=".env",
        extra="ignore",
    )
    
    # Benchmark parameters
    benchmark_sizes: list[int] = Field(default=[4, 6, 8, 10, 12])
    benchmark_runs: int = Field(default=5, ge=1, le=100, description="Runs por configuración")
    
    # Backtest
    backtest_start: str = Field(default="2020-01-01")
    backtest_end: str = Field(default="2024-01-01")
    rebalance_frequency: Literal["monthly", "quarterly", "yearly"] = Field(default="quarterly")
    
    # Risk-free rate (Letras del Tesoro España ~3.5% en 2024)
    risk_free_rate: float = Field(default=0.035, ge=0.0, le=0.15)
    
    # Statistical significance
    significance_level: float = Field(default=0.05, ge=0.01, le=0.1)


# =============================================================================
# Configuración Global
# =============================================================================

class Settings(BaseSettings):
    """Configuración global del proyecto."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )
    
    # Sub-configuraciones
    data: DataSettings = Field(default_factory=DataSettings)
    optimization: OptimizationSettings = Field(default_factory=OptimizationSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    
    # Random seed para reproducibilidad
    random_seed: int = Field(default=42)
    
    # Project info
    project_name: str = Field(default="QuantumCoach")
    version: str = Field(default="1.0.0")


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Obtener instancia singleton de configuración.
    
    Returns:
        Settings: Configuración global del proyecto
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# =============================================================================
# Exportaciones
# =============================================================================

__all__ = [
    "Settings",
    "DataSettings",
    "OptimizationSettings",
    "LLMSettings",
    "EvaluationSettings",
    "get_settings",
    "PROJECT_ROOT",
    "DATA_DIR",
    "RESULTS_DIR",
]
