# QuantumCoach: Optimización Cuántica de Carteras para Inversores Retail

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Descripción

Sistema híbrido cuántico-clásico para optimización de carteras de inversión, diseñado específicamente para inversores retail de la Generación Z en España. Combina el algoritmo QAOA (Quantum Approximate Optimization Algorithm) con explicaciones en lenguaje natural generadas por LLM.

### Motivación

- **Contexto español**: Baja educación financiera (OCDE), salarios estancados, crisis de vivienda
- **Target**: Generación Z con capital limitado (€50-500/mes) y alta aversión a pérdidas
- **Objetivo**: Democratizar herramientas de optimización sofisticadas con explicaciones accesibles

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │Yahoo Finance│  │ IBEX 35     │  │ ECB Rates   │  │ INE (CPI)   │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         └────────────────┴────────────────┴────────────────┘            │
│                                   │                                      │
│                          ┌────────▼────────┐                            │
│                          │   DataEngine    │                            │
│                          │   (DuckDB)      │                            │
│                          └────────┬────────┘                            │
└───────────────────────────────────┼─────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼─────────────────────────────────────┐
│                        OPTIMIZATION LAYER                                │
│                                   │                                      │
│         ┌─────────────────────────┼─────────────────────────────┐       │
│         │                         ▼                             │       │
│         │               ┌─────────────────┐                     │       │
│         │               │   QUBOEngine    │                     │       │
│         │               │   (Markowitz)   │                     │       │
│         │               └────────┬────────┘                     │       │
│         │                        │                              │       │
│         │         ┌──────────────┴──────────────┐               │       │
│         │         ▼                             ▼               │       │
│         │  ┌─────────────┐              ┌─────────────┐         │       │
│         │  │   QAOA      │              │  Classical  │         │       │
│         │  │  (Qiskit)   │              │   (CVXPY)   │         │       │
│         │  └──────┬──────┘              └──────┬──────┘         │       │
│         │         └──────────────┬─────────────┘               │       │
│         │                        ▼                              │       │
│         │               ┌─────────────────┐                     │       │
│         │               │   Comparator    │                     │       │
│         │               │   (Benchmark)   │                     │       │
│         │               └────────┬────────┘                     │       │
│         └────────────────────────┼──────────────────────────────┘       │
└───────────────────────────────────┼─────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼─────────────────────────────────────┐
│                        EXPLANATION LAYER                                 │
│                                   ▼                                      │
│                          ┌─────────────────┐                            │
│                          │  QuantumCoach   │                            │
│                          │    (LLM)        │                            │
│                          └────────┬────────┘                            │
└───────────────────────────────────┼─────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼─────────────────────────────────────┐
│                        EVALUATION LAYER                                  │
│                                   ▼                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Backtest   │  │   Metrics   │  │   Survey    │  │  Visualizer │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📁 Estructura del Proyecto

```
quantum_portfolio_optimizer/
├── README.md
├── pyproject.toml
├── requirements.txt
├── config/
│   ├── __init__.py
│   ├── settings.py          # Configuración centralizada
│   └── assets.py             # Universo de activos españoles
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_engine.py    # Pipeline de datos
│   │   └── spanish_assets.py # Activos específicos España
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── qubo_engine.py    # Formulación QUBO
│   │   ├── quantum_solver.py # QAOA (Qiskit)
│   │   └── classical_solver.py # Baseline clásico
│   ├── explanation/
│   │   ├── __init__.py
│   │   └── coach_engine.py   # LLM explanations
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark.py      # Comparación QAOA vs Clásico
│   │   ├── metrics.py        # Métricas financieras
│   │   └── backtest.py       # Backtesting engine
│   └── visualization/
│       ├── __init__.py
│       └── plots.py          # Visualizaciones
├── tests/
│   ├── __init__.py
│   ├── test_qubo.py
│   ├── test_solvers.py
│   └── test_metrics.py
├── notebooks/
│   └── thesis_experiments.ipynb
└── main.py                   # Entry point
```

## 🚀 Instalación

```bash
# Clonar repositorio
git clone https://github.com/username/quantum-portfolio-optimizer.git
cd quantum-portfolio-optimizer

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tu OPENROUTER_API_KEY
```

## 💻 Uso

### Ejecución Básica

```bash
python main.py --tickers SAN.MC ITX.MC IBE.MC --risk-aversion 0.5
```

### Benchmark Completo

```bash
python main.py --benchmark --sizes 4 8 12 16 --output results/
```

### Con Explicación LLM

```bash
python main.py --tickers SAN.MC ITX.MC IBE.MC --explain --language es
```

## 📊 Resultados Experimentales

Ver `notebooks/thesis_experiments.ipynb` para reproducir todos los experimentos.

### Comparación QAOA vs Clásico (n=8 activos)

| Métrica | QAOA (reps=1) | QAOA (reps=2) | Clásico (MILP) |
|---------|---------------|---------------|----------------|
| Tiempo (s) | 2.34 | 8.91 | 0.12 |
| Gap (%) | 3.2% | 0.8% | 0.0% |
| Sharpe Ratio | 1.23 | 1.28 | 1.31 |

## 📚 Referencias

- Hodson, M., et al. (2019). Portfolio rebalancing experiments using the Quantum Alternating Operator Ansatz.
- Brandhofer, S., et al. (2022). Benchmarking the performance of portfolio optimization with QAOA.
- Markowitz, H. (1952). Portfolio Selection. The Journal of Finance.

## 📄 Licencia

MIT License - ver [LICENSE](LICENSE)

## 👤 Autor

[Tu Nombre] - MSc Big Data Science & AI - [Universidad]
