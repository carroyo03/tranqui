from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class OptimizationRequest(BaseModel):
    tickers: List[str] = Field(..., min_items=2, description="List of ticker symbols (e.g., ['SAN.MC', 'ITX.MC'])")
    risk_aversion: float = Field(0.5, ge=0.0, le=1.0, description="Risk aversion parameter (0.0 to 1.0)")
    qaoa_reps: int = Field(1, ge=1, le=5, description="Number of QAOA layers")

class PortfolioMetric(BaseModel):
    expected_return: float
    volatility: float
    sharpe_ratio: float

class AssetResult(BaseModel):
    ticker: str
    selected: bool
    weight: float
    return_annual: float
    volatility_annual: float

class SolverResultSchema(BaseModel):
    selection: List[int]
    objective_value: float
    assets: List[AssetResult]
    metrics: PortfolioMetric

class OptimizationResponse(BaseModel):
    qaoa: SolverResultSchema
    classical: SolverResultSchema
    gap: float
    market_data: Dict[str, Any]  # Simplified market data for charts

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
