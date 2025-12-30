"""
Universo de activos para inversores españoles.

Define los activos disponibles categorizados por tipo, riesgo,
y accesibilidad para inversores retail en España.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar


class AssetClass(Enum):
    """Clasificación de activos."""
    EQUITY_SPAIN = "Renta Variable España"
    EQUITY_EU = "Renta Variable Europa"
    EQUITY_US = "Renta Variable EEUU"
    EQUITY_EMERGING = "Renta Variable Emergentes"
    FIXED_INCOME = "Renta Fija"
    CRYPTO = "Criptomonedas"
    COMMODITY = "Materias Primas"
    ETF_GLOBAL = "ETF Global"
    REIT = "Inmobiliario (SOCIMI/REIT)"


class RiskLevel(Enum):
    """Nivel de riesgo del activo."""
    LOW = 1
    MEDIUM_LOW = 2
    MEDIUM = 3
    MEDIUM_HIGH = 4
    HIGH = 5
    VERY_HIGH = 6


class Liquidity(Enum):
    """Liquidez del activo."""
    HIGH = "Alta"
    MEDIUM = "Media"
    LOW = "Baja"


@dataclass(frozen=True)
class Asset:
    """Definición de un activo financiero."""
    
    ticker: str
    name: str
    asset_class: AssetClass
    risk_level: RiskLevel
    liquidity: Liquidity
    currency: str = "EUR"
    min_investment: float = 1.0  # Mínimo en EUR
    description: str = ""
    isin: str | None = None
    
    # Metadatos para el contexto español
    available_platforms: tuple[str, ...] = field(default_factory=tuple)
    tax_efficient: bool = False  # Si es fiscalmente eficiente en España
    
    def __hash__(self):
        return hash(self.ticker)


# =============================================================================
# IBEX 35 - Blue Chips Españoles
# =============================================================================

IBEX35_ASSETS = {
    Asset(
        ticker="SAN.MC",
        name="Banco Santander",
        asset_class=AssetClass.EQUITY_SPAIN,
        risk_level=RiskLevel.MEDIUM_HIGH,
        liquidity=Liquidity.HIGH,
        description="Mayor banco de la Eurozona por capitalización",
        isin="ES0113900J37",
        available_platforms=("MyInvestor", "DEGIRO", "Interactive Brokers"),
    ),
    Asset(
        ticker="BBVA.MC",
        name="BBVA",
        asset_class=AssetClass.EQUITY_SPAIN,
        risk_level=RiskLevel.MEDIUM_HIGH,
        liquidity=Liquidity.HIGH,
        description="Segundo banco español, fuerte presencia en México",
        isin="ES0113211835",
        available_platforms=("MyInvestor", "DEGIRO", "Interactive Brokers"),
    ),
    Asset(
        ticker="ITX.MC",
        name="Inditex",
        asset_class=AssetClass.EQUITY_SPAIN,
        risk_level=RiskLevel.MEDIUM,
        liquidity=Liquidity.HIGH,
        description="Líder mundial en moda rápida (Zara, Massimo Dutti)",
        isin="ES0148396007",
        available_platforms=("MyInvestor", "DEGIRO", "Interactive Brokers"),
    ),
    Asset(
        ticker="IBE.MC",
        name="Iberdrola",
        asset_class=AssetClass.EQUITY_SPAIN,
        risk_level=RiskLevel.MEDIUM_LOW,
        liquidity=Liquidity.HIGH,
        description="Líder mundial en energías renovables",
        isin="ES0144580Y14",
        available_platforms=("MyInvestor", "DEGIRO", "Interactive Brokers"),
    ),
    Asset(
        ticker="TEF.MC",
        name="Telefónica",
        asset_class=AssetClass.EQUITY_SPAIN,
        risk_level=RiskLevel.MEDIUM,
        liquidity=Liquidity.HIGH,
        description="Telecomunicaciones, dividendo estable",
        isin="ES0178430E18",
        available_platforms=("MyInvestor", "DEGIRO", "Interactive Brokers"),
    ),
    Asset(
        ticker="REP.MC",
        name="Repsol",
        asset_class=AssetClass.EQUITY_SPAIN,
        risk_level=RiskLevel.MEDIUM_HIGH,
        liquidity=Liquidity.HIGH,
        description="Energía integrada, transición energética",
        isin="ES0173516115",
        available_platforms=("MyInvestor", "DEGIRO", "Interactive Brokers"),
    ),
    Asset(
        ticker="AMS.MC",
        name="Amadeus IT",
        asset_class=AssetClass.EQUITY_SPAIN,
        risk_level=RiskLevel.MEDIUM,
        liquidity=Liquidity.HIGH,
        description="Tecnología para sector turístico",
        isin="ES0109067019",
        available_platforms=("MyInvestor", "DEGIRO", "Interactive Brokers"),
    ),
    Asset(
        ticker="FER.MC",
        name="Ferrovial",
        asset_class=AssetClass.EQUITY_SPAIN,
        risk_level=RiskLevel.MEDIUM,
        liquidity=Liquidity.HIGH,
        description="Infraestructuras y construcción",
        isin="ES0118900010",
        available_platforms=("MyInvestor", "DEGIRO", "Interactive Brokers"),
    ),
}


# =============================================================================
# ETFs Accesibles en España
# =============================================================================

ETF_ASSETS = {
    Asset(
        ticker="VWCE.DE",
        name="Vanguard FTSE All-World",
        asset_class=AssetClass.ETF_GLOBAL,
        risk_level=RiskLevel.MEDIUM,
        liquidity=Liquidity.HIGH,
        description="Exposición global diversificada, bajo coste (TER 0.22%)",
        isin="IE00BK5BQT80",
        available_platforms=("MyInvestor", "DEGIRO", "Interactive Brokers"),
        tax_efficient=True,  # Acumulación, no distribuye dividendos
    ),
    Asset(
        ticker="CSPX.L",
        name="iShares Core S&P 500",
        asset_class=AssetClass.ETF_GLOBAL,
        risk_level=RiskLevel.MEDIUM,
        liquidity=Liquidity.HIGH,
        currency="USD",
        description="S&P 500 acumulación, TER 0.07%",
        isin="IE00B5BMR087",
        available_platforms=("MyInvestor", "DEGIRO", "Interactive Brokers"),
        tax_efficient=True,
    ),
    Asset(
        ticker="EUNL.DE",
        name="iShares Core MSCI World",
        asset_class=AssetClass.ETF_GLOBAL,
        risk_level=RiskLevel.MEDIUM,
        liquidity=Liquidity.HIGH,
        description="MSCI World acumulación, TER 0.20%",
        isin="IE00B4L5Y983",
        available_platforms=("MyInvestor", "DEGIRO", "Interactive Brokers"),
        tax_efficient=True,
    ),
    Asset(
        ticker="IEMA.L",
        name="iShares MSCI EM IMI",
        asset_class=AssetClass.EQUITY_EMERGING,
        risk_level=RiskLevel.HIGH,
        liquidity=Liquidity.HIGH,
        description="Mercados emergentes, TER 0.18%",
        isin="IE00BKM4GZ66",
        available_platforms=("MyInvestor", "DEGIRO", "Interactive Brokers"),
        tax_efficient=True,
    ),
    Asset(
        ticker="IBTS.L",
        name="iShares EUR Govt Bond 1-3yr",
        asset_class=AssetClass.FIXED_INCOME,
        risk_level=RiskLevel.LOW,
        liquidity=Liquidity.HIGH,
        description="Bonos gobierno EUR corto plazo, bajo riesgo",
        isin="IE00B14X4Q57",
        available_platforms=("MyInvestor", "DEGIRO", "Interactive Brokers"),
        tax_efficient=True,
    ),
}


# =============================================================================
# Criptomonedas (para perfiles agresivos)
# =============================================================================

CRYPTO_ASSETS = {
    Asset(
        ticker="BTC-EUR",
        name="Bitcoin",
        asset_class=AssetClass.CRYPTO,
        risk_level=RiskLevel.VERY_HIGH,
        liquidity=Liquidity.HIGH,
        description="Criptomoneda líder, alta volatilidad",
        available_platforms=("Binance", "Kraken", "Coinbase"),
        min_investment=10.0,
    ),
    Asset(
        ticker="ETH-EUR",
        name="Ethereum",
        asset_class=AssetClass.CRYPTO,
        risk_level=RiskLevel.VERY_HIGH,
        liquidity=Liquidity.HIGH,
        description="Plataforma smart contracts, alta volatilidad",
        available_platforms=("Binance", "Kraken", "Coinbase"),
        min_investment=10.0,
    ),
}


# =============================================================================
# US Tech (muy populares entre Gen Z)
# =============================================================================

US_TECH_ASSETS = {
    Asset(
        ticker="AAPL",
        name="Apple Inc.",
        asset_class=AssetClass.EQUITY_US,
        risk_level=RiskLevel.MEDIUM,
        liquidity=Liquidity.HIGH,
        currency="USD",
        description="Tecnología de consumo, ecosistema sólido",
        available_platforms=("DEGIRO", "Interactive Brokers", "eToro"),
    ),
    Asset(
        ticker="MSFT",
        name="Microsoft Corporation",
        asset_class=AssetClass.EQUITY_US,
        risk_level=RiskLevel.MEDIUM,
        liquidity=Liquidity.HIGH,
        currency="USD",
        description="Software empresarial, cloud Azure",
        available_platforms=("DEGIRO", "Interactive Brokers", "eToro"),
    ),
    Asset(
        ticker="GOOGL",
        name="Alphabet Inc.",
        asset_class=AssetClass.EQUITY_US,
        risk_level=RiskLevel.MEDIUM,
        liquidity=Liquidity.HIGH,
        currency="USD",
        description="Búsqueda, publicidad, cloud",
        available_platforms=("DEGIRO", "Interactive Brokers", "eToro"),
    ),
    Asset(
        ticker="NVDA",
        name="NVIDIA Corporation",
        asset_class=AssetClass.EQUITY_US,
        risk_level=RiskLevel.HIGH,
        liquidity=Liquidity.HIGH,
        currency="USD",
        description="GPUs, líder en IA/ML hardware",
        available_platforms=("DEGIRO", "Interactive Brokers", "eToro"),
    ),
    Asset(
        ticker="TSLA",
        name="Tesla Inc.",
        asset_class=AssetClass.EQUITY_US,
        risk_level=RiskLevel.VERY_HIGH,
        liquidity=Liquidity.HIGH,
        currency="USD",
        description="Vehículos eléctricos, muy volátil",
        available_platforms=("DEGIRO", "Interactive Brokers", "eToro"),
    ),
}


# =============================================================================
# Carteras Predefinidas por Perfil
# =============================================================================

@dataclass
class PortfolioProfile:
    """Perfil de cartera predefinido."""
    
    name: str
    description: str
    risk_aversion: float
    tickers: list[str]
    target_allocation: dict[str, float] | None = None


# Carteras modelo para diferentes perfiles
PORTFOLIO_PROFILES: dict[str, PortfolioProfile] = {
    "conservador_espanol": PortfolioProfile(
        name="Conservador España",
        description="Para quien prioriza preservar capital. Ideal para primeros €1,000-5,000",
        risk_aversion=0.8,
        tickers=["IBE.MC", "TEF.MC", "IBTS.L", "EUNL.DE"],
    ),
    "equilibrado_global": PortfolioProfile(
        name="Equilibrado Global",
        description="Balance riesgo-retorno. Para inversión a 5-10 años",
        risk_aversion=0.5,
        tickers=["VWCE.DE", "EUNL.DE", "SAN.MC", "ITX.MC", "CSPX.L"],
    ),
    "crecimiento_tech": PortfolioProfile(
        name="Crecimiento Tech",
        description="Mayor riesgo, mayor potencial. Solo con capital que puedes perder",
        risk_aversion=0.3,
        tickers=["AAPL", "MSFT", "NVDA", "GOOGL", "ITX.MC", "VWCE.DE"],
    ),
    "agresivo_crypto": PortfolioProfile(
        name="Agresivo con Crypto",
        description="Alta volatilidad. Máximo 5-10% de tu patrimonio total",
        risk_aversion=0.15,
        tickers=["BTC-EUR", "ETH-EUR", "NVDA", "TSLA", "AAPL", "VWCE.DE"],
    ),
}


# =============================================================================
# Funciones de Utilidad
# =============================================================================

def get_all_assets() -> dict[str, Asset]:
    """Obtener todos los activos disponibles indexados por ticker."""
    all_assets = {}
    for asset_set in [IBEX35_ASSETS, ETF_ASSETS, CRYPTO_ASSETS, US_TECH_ASSETS]:
        for asset in asset_set:
            all_assets[asset.ticker] = asset
    return all_assets


def get_assets_by_risk(max_risk: RiskLevel) -> list[Asset]:
    """Filtrar activos por nivel de riesgo máximo."""
    all_assets = get_all_assets()
    return [a for a in all_assets.values() if a.risk_level.value <= max_risk.value]


def get_assets_by_class(asset_class: AssetClass) -> list[Asset]:
    """Filtrar activos por clase."""
    all_assets = get_all_assets()
    return [a for a in all_assets.values() if a.asset_class == asset_class]


def get_spanish_only() -> list[Asset]:
    """Obtener solo activos españoles (sin riesgo divisa)."""
    all_assets = get_all_assets()
    return [a for a in all_assets.values() if a.currency == "EUR"]


def get_tax_efficient() -> list[Asset]:
    """Obtener activos fiscalmente eficientes para residentes españoles."""
    all_assets = get_all_assets()
    return [a for a in all_assets.values() if a.tax_efficient]


def get_profile_tickers(profile_name: str) -> list[str]:
    """Obtener tickers para un perfil predefinido."""
    if profile_name not in PORTFOLIO_PROFILES:
        raise ValueError(f"Perfil '{profile_name}' no encontrado. Disponibles: {list(PORTFOLIO_PROFILES.keys())}")
    return PORTFOLIO_PROFILES[profile_name].tickers


# =============================================================================
# Exportaciones
# =============================================================================

__all__ = [
    "Asset",
    "AssetClass",
    "RiskLevel",
    "Liquidity",
    "PortfolioProfile",
    "IBEX35_ASSETS",
    "ETF_ASSETS",
    "CRYPTO_ASSETS",
    "US_TECH_ASSETS",
    "PORTFOLIO_PROFILES",
    "get_all_assets",
    "get_assets_by_risk",
    "get_assets_by_class",
    "get_spanish_only",
    "get_tax_efficient",
    "get_profile_tickers",
]
