"""
Motor de explicaciones con LLM.

Genera explicaciones en lenguaje natural de las decisiones
de optimización de carteras, adaptadas al contexto español
y al público Gen Z.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    from src.data.data_engine import MarketData
    from src.optimization.quantum_solver import SolverResult
    from src.evaluation.metrics import PortfolioMetrics

logger = logging.getLogger(__name__)
console = Console()


# =============================================================================
# Enums
# =============================================================================

class RiskProfile(Enum):
    """Perfiles de riesgo del usuario."""
    CONSERVATIVE = "conservador"
    BALANCED = "equilibrado"
    AGGRESSIVE = "agresivo"
    
    @classmethod
    def from_risk_aversion(cls, risk_aversion: float) -> 'RiskProfile':
        """Determinar perfil desde aversión al riesgo."""
        if risk_aversion > 0.7:
            return cls.CONSERVATIVE
        elif risk_aversion < 0.3:
            return cls.AGGRESSIVE
        return cls.BALANCED
    
    @property
    def description_es(self) -> str:
        """Descripción en español."""
        descriptions = {
            RiskProfile.CONSERVATIVE: "Priorizas la seguridad sobre el rendimiento",
            RiskProfile.BALANCED: "Buscas equilibrio entre riesgo y rendimiento",
            RiskProfile.AGGRESSIVE: "Aceptas más riesgo por mayor potencial de ganancias",
        }
        return descriptions[self]


class Language(Enum):
    """Idiomas soportados."""
    ES = "es"
    EN = "en"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CoachConfig:
    """Configuración del coach LLM."""
    
    api_key: str | None = None
    api_base: str = "https://openrouter.ai/api/v1"
    model: str = "nvidia/nemotron-3-nano-30b-a3b:free"
    fallback_model: str = "nvidia/nemotron-3-nano-30b-a3b:free"
    temperature: float = 0.7
    max_tokens: int = 600
    language: Language = Language.ES
    
    def __post_init__(self):
        # Intentar cargar de entorno si no se proporciona
        if self.api_key is None:
            self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Cargar configuración opcional del entorno
        if os.getenv("LLM_MODEL"):
            self.model = os.getenv("LLM_MODEL")
        if os.getenv("LLM_TEMPERATURE"):
            self.temperature = float(os.getenv("LLM_TEMPERATURE"))
        if os.getenv("LLM_MAX_TOKENS"):
            self.max_tokens = int(os.getenv("LLM_MAX_TOKENS"))


# =============================================================================
# Prompts
# =============================================================================

SPANISH_COACH_PROMPT = """
Eres un "Quantum Financial Coach" especializado en ayudar a jóvenes españoles 
de la Generación Z a entender y comenzar a invertir con confianza.

## CONTEXTO DEL USUARIO
- Perfil de riesgo: {risk_profile} ({risk_profile_description})
- Aversión al riesgo: {risk_aversion:.0%}
- Contexto: Joven español, probablemente con salario limitado y preocupado 
  por el futuro económico (vivienda, pensiones, inflación)

## DECISIÓN DEL ALGORITMO CUÁNTICO
El algoritmo QAOA ha analizado {n_assets} activos y ha optimizado la cartera 
minimizando el riesgo según tu perfil.

**Activos seleccionados (COMPRAR):**
{selected_assets}

**Activos descartados (EVITAR por ahora):**
{rejected_assets}

## DATOS DEL ANÁLISIS
{market_data_summary}

## MÉTRICAS DE LA CARTERA RECOMENDADA
- Retorno esperado: {expected_return}
- Volatilidad: {volatility}
- Ratio Sharpe: {sharpe_ratio}

## TU MISIÓN
Escribe una explicación breve y accesible que:

1. **Resuma la decisión** en 2-3 frases claras
2. **Explique el "por qué"** de forma que un novato entienda:
   - Si se descartó crypto o tech volátil, explica cómo esto protege sus ahorros
   - Si se seleccionó renta fija o blue chips, destaca la importancia de la base sólida
3. **Use una analogía cotidiana** que conecte (ej: comparar con el alquiler vs compra, 
   con ahorrar para unas vacaciones, etc.)
4. **Cierre con un paso accionable** específico para España (mencionar plataformas 
   como MyInvestor, Indexa si es relevante)

## FORMATO
- Máximo 250 palabras
- Tono cercano pero informativo (tutea al usuario)
- Evita jerga financiera sin explicar
- NO uses emojis en exceso
- NO seas condescendiente

## IMPORTANTE
- Si el usuario es conservador y el algoritmo descartó cripto, valida esa decisión
- Si el usuario es agresivo pero el algoritmo moderó, explica por qué es prudente
- Menciona el contexto español (tipos de interés BCE, Letras del Tesoro, fiscalidad) 
  solo si es relevante
"""

ENGLISH_COACH_PROMPT = """
You are a Quantum Financial Coach helping young investors understand 
portfolio optimization decisions.

## USER CONTEXT
- Risk profile: {risk_profile}
- Risk aversion: {risk_aversion:.0%}

## ALGORITHM DECISION
The QAOA algorithm analyzed {n_assets} assets and optimized for your risk profile.

**Selected (BUY):** {selected_assets}
**Rejected (AVOID):** {rejected_assets}

## PORTFOLIO METRICS
- Expected return: {expected_return}
- Volatility: {volatility}
- Sharpe ratio: {sharpe_ratio}

## YOUR TASK
Write a brief, accessible explanation (max 200 words) that:
1. Summarizes the decision
2. Explains why certain assets were selected/rejected
3. Uses a simple analogy
4. Gives one actionable next step

Be friendly but informative. Avoid jargon without explanation.
"""


# =============================================================================
# Coach Engine
# =============================================================================

class QuantumCoach:
    """
    Genera explicaciones en lenguaje natural de las decisiones de cartera.
    """
    
    def __init__(self, config: CoachConfig | None = None):
        """
        Inicializar coach.
        
        Args:
            config: Configuración del LLM
        """
        self.config = config or CoachConfig()
        self.llm = self._init_llm()
        self.prompt = self._get_prompt_template()
        
        if self.llm:
            logger.info(f"QuantumCoach inicializado con modelo: {self.config.model}")
        else:
            logger.warning("QuantumCoach en modo simulación (sin API key)")
    
    def _init_llm(self) -> ChatOpenAI | None:
        """Inicializar cliente LLM."""
        if not self.config.api_key:
            return None
        
        try:
            return ChatOpenAI(
                model=self.config.model,
                openai_api_key=self.config.api_key,
                openai_api_base=self.config.api_base,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception as e:
            logger.error(f"Error inicializando LLM: {e}")
            # Intentar con modelo fallback
            try:
                return ChatOpenAI(
                    model=self.config.fallback_model,
                    openai_api_key=self.config.api_key,
                    openai_api_base=self.config.api_base,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            except Exception:
                return None
    
    def _get_prompt_template(self) -> PromptTemplate:
        """Obtener template según idioma."""
        template = SPANISH_COACH_PROMPT if self.config.language == Language.ES else ENGLISH_COACH_PROMPT
        
        return PromptTemplate(
            input_variables=[
                "risk_profile",
                "risk_profile_description",
                "risk_aversion",
                "n_assets",
                "selected_assets",
                "rejected_assets",
                "market_data_summary",
                "expected_return",
                "volatility",
                "sharpe_ratio",
            ],
            template=template,
        )
    
    def generate_explanation(
        self,
        solver_result: SolverResult,
        market_data: MarketData,
        metrics: PortfolioMetrics | None = None,
        risk_aversion: float = 0.5,
    ) -> str:
        """
        Generar explicación de la decisión de cartera.
        
        Args:
            solver_result: Resultado del solver
            market_data: Datos de mercado
            metrics: Métricas de la cartera (opcional)
            risk_aversion: Aversión al riesgo del usuario
        
        Returns:
            Explicación en lenguaje natural
        """
        # Determinar perfil
        profile = RiskProfile.from_risk_aversion(risk_aversion)
        
        # Formatear datos de mercado
        market_summary = self._format_market_data(market_data, solver_result.selection)
        
        # Preparar inputs
        inputs = {
            "risk_profile": profile.value,
            "risk_profile_description": profile.description_es,
            "risk_aversion": risk_aversion,
            "n_assets": len(solver_result.selection),
            "selected_assets": ", ".join(solver_result.selected_assets) or "Ninguno",
            "rejected_assets": ", ".join(solver_result.rejected_assets) or "Ninguno",
            "market_data_summary": market_summary,
            "expected_return": f"{metrics.expected_return:.1%}" if metrics else "N/A",
            "volatility": f"{metrics.volatility:.1%}" if metrics else "N/A",
            "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}" if metrics else "N/A",
        }
        
        # Generar con LLM o fallback
        if self.llm:
            try:
                chain = self.prompt | self.llm
                response = chain.invoke(inputs)
                return response.content
            except Exception as e:
                logger.error(f"Error generando explicación: {e}")
                return self._generate_fallback(inputs, profile)
        else:
            return self._generate_fallback(inputs, profile)
    
    def _format_market_data(
        self,
        market_data: MarketData,
        selection: list[int],
    ) -> str:
        """Formatear datos de mercado para el prompt."""
        lines = ["| Activo | Retorno | Volatilidad | Seleccionado |"]
        lines.append("|--------|---------|-------------|--------------|")
        
        for i, ticker in enumerate(market_data.tickers):
            ret = market_data.mu.get(ticker, 0)
            vol = market_data.sigma.loc[ticker, ticker] ** 0.5 if ticker in market_data.sigma.index else 0
            selected = "✓" if selection[i] == 1 else "✗"
            
            lines.append(f"| {ticker} | {ret:+.1%} | {vol:.1%} | {selected} |")
        
        return "\n".join(lines)
    
    def _generate_fallback(
        self,
        inputs: dict,
        profile: RiskProfile,
    ) -> str:
        """Generar explicación sin LLM (fallback)."""
        selected = inputs["selected_assets"]
        rejected = inputs["rejected_assets"]
        
        # Templates según perfil
        if profile == RiskProfile.CONSERVATIVE:
            intro = f"""El algoritmo ha optimizado tu cartera priorizando la estabilidad, 
            acorde con tu perfil conservador (aversión al riesgo: {inputs['risk_aversion']:.0%})."""
            
            if "BTC" in rejected or "crypto" in rejected.lower():
                reasoning = """Se han descartado activos de alta volatilidad como las criptomonedas. 
                Aunque pueden ofrecer altos retornos, su volatilidad puede superar el 50% anual, 
                lo que significa que podrías ver tu inversión reducida a la mitad en meses malos."""
            else:
                reasoning = """Los activos seleccionados tienen un historial de menor volatilidad, 
                lo que significa fluctuaciones más predecibles en tu cartera."""
            
            analogy = """Piensa en esto como elegir un piso en un barrio consolidado vs uno en una zona 
            "de moda" pero incierta. Menos emoción, pero duermes más tranquilo."""
            
            action = """Siguiente paso: Con €100-200/mes podrías empezar en MyInvestor o Indexa Capital 
            con una cartera diversificada automáticamente."""
        
        elif profile == RiskProfile.AGGRESSIVE:
            intro = f"""El algoritmo ha buscado maximizar el potencial de retorno, 
            aunque ha moderado algunas posiciones extremas para tu protección."""
            
            reasoning = f"""Se han seleccionado: {selected}. Estos activos ofrecen mayor potencial 
            de crecimiento, pero recuerda que la volatilidad también es mayor."""
            
            analogy = """Es como elegir una startup donde trabajar: más riesgo, pero si funciona, 
            el upside es mucho mayor que en una empresa tradicional."""
            
            action = """Siguiente paso: Asegúrate de que esta inversión sea dinero que no necesites 
            en los próximos 5-10 años. La volatilidad es tu amiga solo si tienes tiempo."""
        
        else:  # BALANCED
            intro = f"""El algoritmo ha encontrado un equilibrio entre retorno y riesgo, 
            ideal para tu perfil (aversión al riesgo: {inputs['risk_aversion']:.0%})."""
            
            reasoning = f"""Activos seleccionados: {selected}. Esta combinación ofrece diversificación 
            entre diferentes sectores y niveles de riesgo."""
            
            analogy = """Es como una dieta equilibrada: ni solo ensaladas ni solo postres. 
            Una mezcla que puedes mantener a largo plazo."""
            
            action = """Siguiente paso: Considera automatizar aportaciones mensuales para aprovechar 
            el "dollar-cost averaging" y reducir el impacto de la volatilidad."""
        
        return f"""**Resumen de la Optimización**

{intro}

**Por qué estas decisiones:**
{reasoning}

**Analogía:**
{analogy}

**Acción recomendada:**
{action}

---
*Nota: Esta explicación se ha generado en modo offline. Para explicaciones personalizadas 
con IA, configura tu API key de OpenRouter.*
"""
    
    def explain_single_asset(
        self,
        ticker: str,
        selected: bool,
        market_data: MarketData,
    ) -> str:
        """
        Explicar por qué un activo específico fue seleccionado o rechazado.
        
        Args:
            ticker: Símbolo del activo
            selected: Si fue seleccionado
            market_data: Datos de mercado
        
        Returns:
            Explicación breve
        """
        if ticker not in market_data.tickers:
            return f"No hay datos disponibles para {ticker}"
        
        ret = market_data.mu.get(ticker, 0)
        vol = market_data.sigma.loc[ticker, ticker] ** 0.5 if ticker in market_data.sigma.index else 0
        sharpe = (ret - 0.035) / vol if vol > 0 else 0
        
        if selected:
            if sharpe > 1:
                reason = f"excelente ratio riesgo-retorno (Sharpe: {sharpe:.2f})"
            elif sharpe > 0.5:
                reason = f"buen equilibrio entre retorno ({ret:.1%}) y riesgo ({vol:.1%})"
            else:
                reason = f"contribuye a la diversificación de la cartera"
            return f"✓ {ticker} seleccionado: {reason}"
        else:
            if vol > 0.30:
                reason = f"volatilidad muy alta ({vol:.1%})"
            elif sharpe < 0:
                reason = f"retorno inferior a la tasa libre de riesgo"
            else:
                reason = f"no mejora el perfil riesgo-retorno de la cartera"
            return f"✗ {ticker} descartado: {reason}"


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_report(
    tickers: list[str],
    selection: list[int],
    risk_aversion: float,
    mu,
    sigma,
) -> str:
    """
    Función de conveniencia para compatibilidad con código original.
    """
    from src.data.data_engine import MarketData
    from src.optimization.quantum_solver import SolverResult
    import pandas as pd
    
    # Construir objetos necesarios
    market_data = MarketData(
        tickers=tickers,
        prices=pd.DataFrame(),
        returns=pd.DataFrame(),
        mu=mu if isinstance(mu, pd.Series) else pd.Series(mu, index=tickers),
        sigma=sigma if isinstance(sigma, pd.DataFrame) else pd.DataFrame(sigma, index=tickers, columns=tickers),
        metadata={},
    )
    
    solver_result = SolverResult(
        selection=selection,
        objective_value=0,
        selected_assets=[t for t, s in zip(tickers, selection) if s == 1],
        rejected_assets=[t for t, s in zip(tickers, selection) if s == 0],
    )
    
    coach = QuantumCoach()
    return coach.generate_explanation(solver_result, market_data, risk_aversion=risk_aversion)


# =============================================================================
# Display
# =============================================================================

def display_explanation(explanation: str, title: str = "Quantum Coach") -> None:
    """Mostrar explicación de forma visual."""
    console.print(Panel(
        explanation,
        title=f"🎓 {title}",
        border_style="magenta",
        padding=(1, 2),
    ))


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from src.data.data_engine import MarketData
    from src.optimization.quantum_solver import SolverResult
    from src.evaluation.metrics import MetricsCalculator
    
    # Datos de ejemplo
    tickers = ["SAN.MC", "ITX.MC", "IBE.MC", "BTC-EUR"]
    
    np.random.seed(42)
    mu = pd.Series([0.08, 0.12, 0.06, 0.25], index=tickers)
    sigma = pd.DataFrame(
        [[0.04, 0.01, 0.02, 0.01],
         [0.01, 0.09, 0.01, 0.02],
         [0.02, 0.01, 0.03, 0.00],
         [0.01, 0.02, 0.00, 0.50]],
        index=tickers, columns=tickers
    )
    
    returns = pd.DataFrame(
        np.random.multivariate_normal(mu.values / 252, sigma.values / 252, size=500),
        columns=tickers
    )
    
    market_data = MarketData(
        tickers=tickers,
        prices=pd.DataFrame(),
        returns=returns,
        mu=mu,
        sigma=sigma,
        metadata={}
    )
    
    # Simular resultado de solver
    selection = [1, 1, 1, 0]  # Descartó crypto
    solver_result = SolverResult(
        selection=selection,
        objective_value=-0.05,
        selected_assets=["SAN.MC", "ITX.MC", "IBE.MC"],
        rejected_assets=["BTC-EUR"],
    )
    
    # Calcular métricas
    calculator = MetricsCalculator()
    metrics = calculator.calculate_portfolio_metrics(selection, market_data)
    
    # Generar explicación
    console.rule("[bold magenta]🎓 Quantum Financial Coach[/bold magenta]")
    
    coach = QuantumCoach()
    
    for risk_level in [0.2, 0.5, 0.8]:
        console.print(f"\n[bold]Perfil: Aversión al riesgo = {risk_level:.0%}[/bold]")
        explanation = coach.generate_explanation(
            solver_result, market_data, metrics, risk_aversion=risk_level
        )
        display_explanation(explanation, f"Perfil λ={risk_level}")
