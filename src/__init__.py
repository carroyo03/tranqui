"""
QuantumCoach - Optimización de Carteras con Computación Cuántica.

Sistema híbrido cuántico-clásico para optimización de carteras de inversión,
diseñado para inversores retail de la Generación Z en España.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@university.es"

from src.data.data_engine import DataEngine, MarketData, get_data
from src.optimization.qubo_engine import QUBOEngine, QUBOResult, to_qubo
from src.optimization.quantum_solver import QuantumSolver, QAOAConfig, solve_with_qaoa
from src.optimization.classical_solver import ClassicalSolver, ClassicalMethod, compare_solvers
from src.explanation.coach_engine import QuantumCoach, generate_report

__all__ = [
    # Data
    "DataEngine",
    "MarketData",
    "get_data",
    # Optimization
    "QUBOEngine",
    "QUBOResult",
    "to_qubo",
    "QuantumSolver",
    "QAOAConfig",
    "solve_with_qaoa",
    "ClassicalSolver",
    "ClassicalMethod",
    "compare_solvers",
    # Explanation
    "QuantumCoach",
    "generate_report",
]
