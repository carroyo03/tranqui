from fastapi import APIRouter, HTTPException
from api.models import OptimizationRequest, OptimizationResponse, ChatRequest, ChatResponse, AssetResult, SolverResultSchema, PortfolioMetric
import sys
from pathlib import Path
import numpy as np

# Add src to path if not already there
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.data_engine import DataEngine
from src.optimization.qubo_engine import QUBOEngine, Constraint, ConstraintType
from src.optimization.classical_solver import compare_solvers, ClassicalMethod
from src.evaluation.metrics import MetricsCalculator
from src.explanation.coach_engine import QuantumCoach
from src.optimization.quantum_solver import SolverResult

router = APIRouter()

@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_portfolio(request: OptimizationRequest):
    try:
        # 1. Fetch Data
        engine = DataEngine()
        market_data = engine.fetch_data(request.tickers)
        engine.close()

        if market_data.returns.empty:
            raise HTTPException(status_code=400, detail="Could not fetch market data for provided tickers")

        # 2. Build QUBO
        qubo_engine = QUBOEngine()
        # Default constraint: Min 2 assets or 50% of portfolio if small
        # Use market_data.tickers (valid ones) instead of request.tickers
        min_assets = max(2, int(len(market_data.tickers) * 0.3))
        constraints = [Constraint(ConstraintType.MIN_ASSETS, min_assets)]
        
        qubo_result = qubo_engine.to_qubo(
            market_data.mu,
            market_data.sigma,
            risk_aversion=request.risk_aversion,
            constraints=constraints
        )

        # 3. Solve (QAOA vs Classical)
        # For web response time, we might want to limit reps or use a mock if real QAOA is too slow
        # But for now, let's run it real.
        comparison = compare_solvers(
            qubo_result,
            qaoa_reps=request.qaoa_reps,
            classical_method=ClassicalMethod.GREEDY # Faster than Brute Force
        )

        # 4. Format Results
        calculator = MetricsCalculator()
        
        def format_result(selection, method_name):
            metrics = calculator.calculate_portfolio_metrics(selection, market_data)
            assets = []
            # Use market_data.tickers to ensure we iterate only over valid assets that match selection index
            for i, ticker in enumerate(market_data.tickers):
                ret = market_data.mu.get(ticker, 0)
                vol = np.sqrt(market_data.sigma.loc[ticker, ticker]) if ticker in market_data.sigma.index else 0
                assets.append(AssetResult(
                    ticker=ticker,
                    selected=bool(selection[i]),
                    weight=selection[i] / sum(selection) if sum(selection) > 0 else 0,
                    return_annual=float(ret),
                    volatility_annual=float(vol)
                ))
            
            return SolverResultSchema(
                selection=selection,
                objective_value=getattr(comparison, f"{method_name}_result")["objective_value"],
                assets=assets,
                metrics=PortfolioMetric(
                    expected_return=metrics.expected_return,
                    volatility=metrics.volatility,
                    sharpe_ratio=metrics.sharpe_ratio
                )
            )

        qaoa_schema = format_result(comparison.qaoa_result["selection"], "qaoa")
        classical_schema = format_result(comparison.classical_result["selection"], "classical")
        
        # Prepare market data for frontend charts (e.g., cumulative returns history)
        # Converting index to string for JSON serialization
        history = market_data.returns.cumsum().to_dict() # This might be too heavy?
        # Let's send just mu/sigma for now to keep payload light
        simple_market_data = {
           "tickers": request.tickers,
           "mu": market_data.mu.to_dict(),
           "sigma": market_data.sigma.to_dict() # Nested dict
        }

        return OptimizationResponse(
            qaoa=qaoa_schema,
            classical=classical_schema,
            gap=comparison.objective_gap,
            market_data=simple_market_data # We can expand this later
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat_coach(request: ChatRequest):
    try:
        # Verify context is sufficient
        if not request.context:
             # Basic answer if no context
             return ChatResponse(response="¡Hola! Soy tu Quantum Coach. Para darte consejos específicos, primero necesito que optimices una cartera.")
        
        # Reconstruct necessary objects for the Coach
        # This is a bit tricky because Coach expects internal objects, but we can pass mock/simplified versions
        # Or, easier: We use the context provided in request to generate the prompt text manually or reuse part of Coach logic
        
        coach = QuantumCoach() 
        # Check if we have API Key
        if not coach.llm:
             return ChatResponse(response="[Modo Demo] Para activar el chat interactivo, configura tu OPENROUTER_API_KEY. Por ahora, te explicaré que el algoritmo QAOA busca minimizar la energía de tu Hamiltoniano de inversión...")

        # If we have LLM, let's construct a prompt with the context
        # We can implement a generic 'chat' method in Coach or use the generation logic
        
        # For now, let's act as if we are generating the explanation again but with the custom message?
        # Actually the user asked for "Interactive chat".
        
        # Implementation:
        messages = [
            ("system", "Eres un experto en finanzas cuánticas y asesor para la Gen Z. Responde de forma breve y didáctica."),
            ("user", f"Contexto de mi cartera: {request.context}. Pregunta: {request.message}")
        ]
        
        response = coach.llm.invoke(messages)
        return ChatResponse(response=response.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
