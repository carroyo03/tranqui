#!/usr/bin/env python3
"""
QuantumCoach: Optimización Cuántica de Carteras

Entry point principal del sistema híbrido cuántico-clásico
para optimización de carteras de inversión.

Uso:
    python main.py --help
    python main.py --tickers SAN.MC ITX.MC IBE.MC --risk-aversion 0.5
    python main.py --benchmark --sizes 4 6 8
    python main.py --profile equilibrado_global
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

console = Console()


def setup_paths():
    """Añadir rutas del proyecto al path."""
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    """Parsear argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="QuantumCoach: Optimización de Carteras con QAOA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py --tickers SAN.MC ITX.MC IBE.MC
  python main.py --profile conservador_espanol
  python main.py --benchmark --sizes 4 6 8 10
  python main.py --tickers AAPL MSFT GOOGL --risk-aversion 0.3 --explain
        """
    )
    
    # Modo de operación
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--tickers", "-t",
        nargs="+",
        help="Lista de tickers a optimizar"
    )
    mode_group.add_argument(
        "--profile", "-p",
        choices=["conservador_espanol", "equilibrado_global", "crecimiento_tech", "agresivo_crypto"],
        help="Usar perfil de cartera predefinido"
    )
    mode_group.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Ejecutar benchmark QAOA vs Clásico"
    )
    
    # Parámetros de optimización
    parser.add_argument(
        "--risk-aversion", "-r",
        type=float,
        default=0.5,
        help="Aversión al riesgo [0-1] (default: 0.5)"
    )
    parser.add_argument(
        "--min-assets",
        type=int,
        default=2,
        help="Mínimo de activos a seleccionar (default: 2)"
    )
    parser.add_argument(
        "--qaoa-reps",
        type=int,
        default=1,
        help="Capas QAOA (default: 1)"
    )
    
    # Benchmark
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[4, 6, 8],
        help="Tamaños de problema para benchmark"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Número de ejecuciones por configuración"
    )
    
    # Output
    parser.add_argument(
        "--explain", "-e",
        action="store_true",
        help="Generar explicación con LLM"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results"),
        help="Directorio de salida"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Modo verbose"
    )
    
    return parser.parse_args()


def run_optimization(args: argparse.Namespace) -> None:
    """Ejecutar optimización de cartera."""
    from config.assets import get_profile_tickers, PORTFOLIO_PROFILES
    from src.data.data_engine import DataEngine, display_market_data
    from src.optimization.qubo_engine import QUBOEngine, Constraint, ConstraintType, display_qubo
    from src.optimization.quantum_solver import QuantumSolver, QAOAConfig
    from src.optimization.classical_solver import ClassicalSolver, ClassicalMethod, compare_solvers
    from src.evaluation.metrics import MetricsCalculator, display_portfolio_metrics
    from src.explanation.coach_engine import QuantumCoach, display_explanation
    
    # Determinar tickers
    if args.profile:
        tickers = get_profile_tickers(args.profile)
        profile_info = PORTFOLIO_PROFILES[args.profile]
        console.print(Panel(
            f"[bold]{profile_info.name}[/bold]\n{profile_info.description}",
            title="📊 Perfil Seleccionado",
            border_style="cyan"
        ))
    else:
        tickers = args.tickers
    
    if not tickers:
        console.print("[red]Error: Especifica --tickers o --profile[/red]")
        return
    
    console.rule(f"[bold cyan]⚛️ QuantumCoach - Optimización de Cartera[/bold cyan]")
    console.print(f"Activos: {', '.join(tickers)}")
    console.print(f"Aversión al riesgo: {args.risk_aversion}")
    console.print()
    
    # 1. Obtener datos de mercado
    console.rule("[bold blue]📡 Obteniendo Datos de Mercado[/bold blue]")
    
    engine = DataEngine()
    try:
        market_data = engine.fetch_data(tickers)
        display_market_data(market_data)
    except Exception as e:
        console.print(f"[red]Error obteniendo datos: {e}[/red]")
        return
    finally:
        engine.close()
    
    # 2. Construir QUBO
    console.rule("[bold yellow]⚛️ Construyendo Problema QUBO[/bold yellow]")
    
    qubo_engine = QUBOEngine()
    constraints = [
        Constraint(ConstraintType.MIN_ASSETS, args.min_assets),
    ]
    
    qubo_result = qubo_engine.to_qubo(
        market_data.mu,
        market_data.sigma,
        risk_aversion=args.risk_aversion,
        constraints=constraints,
    )
    
    display_qubo(qubo_result)
    
    # 3. Resolver con QAOA y comparar con clásico
    console.rule("[bold magenta]🔬 Comparación Quantum vs Clásico[/bold magenta]")
    
    comparison = compare_solvers(
        qubo_result,
        qaoa_reps=args.qaoa_reps,
        classical_method=ClassicalMethod.BRUTE_FORCE if len(tickers) <= 12 else ClassicalMethod.GREEDY,
    )
    
    # 4. Calcular métricas financieras
    console.rule("[bold green]📊 Métricas de la Cartera[/bold green]")
    
    # Usar resultado de QAOA
    selection = comparison.qaoa_result["selection"]
    
    calculator = MetricsCalculator()
    metrics = calculator.calculate_portfolio_metrics(selection, market_data)
    display_portfolio_metrics(metrics, "Cartera QAOA")
    
    # 5. Generar explicación (si se solicita)
    if args.explain:
        console.rule("[bold magenta]🎓 Explicación del Coach[/bold magenta]")
        
        from src.optimization.quantum_solver import SolverResult
        
        solver_result = SolverResult(
            selection=selection,
            objective_value=comparison.qaoa_result["objective_value"],
            selected_assets=comparison.qaoa_result["selected_assets"],
            rejected_assets=comparison.qaoa_result["rejected_assets"],
        )
        
        coach = QuantumCoach()
        explanation = coach.generate_explanation(
            solver_result,
            market_data,
            metrics,
            risk_aversion=args.risk_aversion,
        )
        
        display_explanation(explanation)
    
    # 6. Resumen final
    console.rule("[bold cyan]🏆 Resumen Final[/bold cyan]")
    
    summary_table = Table(title="Cartera Optimizada")
    summary_table.add_column("Activo", style="cyan")
    summary_table.add_column("Decisión", justify="center")
    summary_table.add_column("Retorno Esp.", justify="right")
    summary_table.add_column("Volatilidad", justify="right")
    
    for i, ticker in enumerate(tickers):
        decision = "[green]✓ COMPRAR[/green]" if selection[i] == 1 else "[red]✗ EVITAR[/red]"
        ret = market_data.mu.get(ticker, 0)
        vol = np.sqrt(market_data.sigma.loc[ticker, ticker]) if ticker in market_data.sigma.index else 0
        
        summary_table.add_row(ticker, decision, f"{ret:.2%}", f"{vol:.2%}")
    
    console.print(summary_table)
    
    console.print(f"\n[bold green]✓ Optimización completada[/bold green]")
    console.print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    console.print(f"  Gap vs óptimo clásico: {comparison.objective_gap*100:.2f}%")


def run_benchmark(args: argparse.Namespace) -> None:
    """Ejecutar benchmark completo."""
    from src.evaluation.benchmark import BenchmarkEngine, BenchmarkConfig, statistical_comparison
    from src.optimization.classical_solver import ClassicalMethod
    
    console.rule("[bold magenta]🔬 Benchmark QAOA vs Clásico[/bold magenta]")
    
    config = BenchmarkConfig(
        problem_sizes=args.sizes,
        qaoa_reps_range=[1, 2],
        classical_methods=[ClassicalMethod.BRUTE_FORCE, ClassicalMethod.GREEDY],
        n_runs=args.runs,
        risk_aversion_range=[0.3, 0.5, 0.7],
    )
    
    engine = BenchmarkEngine(config)
    result = engine.run_full_benchmark()
    
    # Guardar resultados
    output_path = args.output / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    
    console.print(f"\n[green]Resultados guardados en {output_path}[/green]")
    
    # Análisis estadístico
    console.rule("[bold cyan]📈 Análisis Estadístico[/bold cyan]")
    
    stats = statistical_comparison(result)
    for key, value in stats.items():
        sig = "[green]✓ Significativo[/green]" if value['significant'] else "[yellow]✗ No significativo[/yellow]"
        console.print(f"{key}: p={value['t_pvalue']:.4f} {sig}")
    
    # Análisis de escalabilidad
    console.print()
    scalability = engine.run_scalability_analysis(max_size=max(args.sizes))
    
    console.print("\n[bold]Escalabilidad (tiempo medio en segundos):[/bold]")
    for _, row in scalability.iterrows():
        console.print(
            f"  n={int(row['n_assets']):2d}: "
            f"QAOA={row['qaoa_time_mean']:.3f}s, "
            f"Clásico={row['classical_time_mean']:.3f}s"
        )


def main() -> int:
    """Entry point principal."""
    setup_paths()
    args = parse_args()
    
    # Banner
    console.print(Panel.fit(
        "[bold magenta]QuantumCoach[/bold magenta]\n"
        "[dim]Optimización de Carteras con Computación Cuántica[/dim]",
        border_style="magenta",
    ))
    
    try:
        if args.benchmark:
            run_benchmark(args)
        elif args.tickers or args.profile:
            run_optimization(args)
        else:
            console.print("[yellow]Usa --help para ver opciones disponibles[/yellow]")
            console.print("\nEjemplos rápidos:")
            console.print("  python main.py --profile equilibrado_global --explain")
            console.print("  python main.py --tickers SAN.MC ITX.MC IBE.MC -r 0.5")
            console.print("  python main.py --benchmark --sizes 4 6 8")
            return 1
        
        return 0
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelado por el usuario[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if args.verbose:
            console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())
