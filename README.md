# 🧘 Tranqui (QuantumCoach)
> **Financial Peace through Quantum Intelligence.**
>
> *A hybrid quantum-classical system democratizing portfolio optimization for Gen Z.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Framework: FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Frontend: Vite+React](https://img.shields.io/badge/Web-Vite%2BReact-646CFF.svg)](https://vitejs.dev/)

---

## 🌗 The Dual Vision

This project (Master's Thesis) addresses the retail investment problem from two complementary perspectives:

### 1. The Scientific Perspective (Thesis Core)
**"Benchmarking Quantum Algorithms in Finance"**
A rigorous comparison engine between:
*   **Classical Solvers**: Brute Force (exact for $N < 20$) and Greedy (fast approximation).
*   **Quantum Solvers**: QAOA (Quantum Approximate Optimization Algorithm) executed on Qiskit/Aer simulators.
*   **Objective**: Demonstrate the viability of "Hybrid Logic" -> Use classical today, be ready for quantum advantage tomorrow.

### 2. The Product Perspective (UX/Gen Z)
**"Tranqui: Your Zen Financial Coach"**
Quantum technology is intimidating; so is finance.
*   **The Solution**: A "Dark/Neon Premium" interface that hides mathematical complexity under a layer of financial well-being.
*   **The Coach**: An LLM-based system (LangChain) that translates "Sharpe Ratios" and "Efficient Frontiers" into empathetic natural language. *"Don't worry about volatility; your portfolio is mathematically shielded."*

---

## 🏗️ System Architecture

The system follows a strict modular architecture to separate research logic (Core) from the user application (Web/API).

```mermaid
graph TD
    subgraph "Frontend Layer (Vite/React)"
        UI[Web App UI] --> |HTTP/JSON| API
        UI --> |Auth| Firebase[Firebase Auth]
    end

    subgraph "Service Layer (FastAPI)"
        API[API Gateway] --> Core
        API --> DB[(DuckDB / Cache)]
    end

    subgraph "Core Library (src/)"
        Core --> Data[Data Engine]
        Core --> Opt[Optimization Engine]
        Core --> Coach[AI Coach Engine]
    end

    subgraph "Optimization Engine"
        Opt --> CLASSICAL[Classical Solvers\n(Brute Force / Greedy)]
        Opt --> QUANTUM[Quantum Solver\n(Qiskit QAOA)]
        QUANTUM -.-> |Benchmarking| CLASSICAL
    end

    subgraph "External Providers"
        Data --> YFin[Yahoo Finance]
        Coach --> LLM[OpenAI / LLM API]
    end
```

### Key Components

| Module | Technology | Function |
|--------|------------|---------|
| **`src.optimization`** | Qiskit, Numpy | The mathematical heart. Transforms Markowitz problems into Hamiltonians (QUBO) for QAOA. |
| **`src.explanation`** | LangChain | Converts numerical decision vectors into personalized financial narratives. |
| **`api/`** | FastAPI | Exposes the Core's power through REST endpoints (`/optimize`, `/chat`, `/benchmark`). |
| **`web/`** | React + Vite | Interactive dashboard with "Glassmorphism" aesthetics and real-time data visualization. |

---

## 🚀 Quick Start (For Developers)

### Prerequisites
- Python 3.10+
- Node.js 18+ (for the frontend)
- LLM API Key (OpenRouter/OpenAI)

### 1. Core & Backend (Python)

```bash
# 1. Clone and set up environment
git clone https://github.com/username/tranqui-quantum.git
cd tranqui-quantum
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configuration
cp .env.example .env
# [!] Edit .env and add your API keys

# 4. Test the Core (CLI)
# Run a quick optimization of 3 Spanish assets
python main.py --tickers SAN.MC ITX.MC IBE.MC --risk-aversion 0.5 --explain

# 5. Start the API Server
./start_app.sh
# Or manually: uvicorn api.main:app --reload
```

### 2. Frontend (React)

```bash
cd web

# 1. Install dependencies
npm install

# 2. Start development server
npm run dev

# Access http://localhost:5173
```

---

## 🔬 Scientific Validation (Thesis Experiments)

The project includes a benchmarking module (`src.evaluation.benchmark`) to validate QAOA performance against classical methods.

**Preliminary Results (Simulation):**
*   **Small Scale ($N=4-8$)**: QAOA reaches the global optimum (GAP < 1%) consistently with > p=2 layers.
*   **Medium Scale ($N=12-16$)**: Simulation noise and circuit depth require more robust hybrid classical optimizers (COBYLA/SPSA).
*   **Speed**: Classical is currently orders of magnitude faster (milliseconds vs seconds/minutes), justifying Tranqui's hybrid approach: *Use Classical for real-time response, Quantum for deep offline analysis.*

To reproduce the thesis experiments:
```bash
python main.py --benchmark --sizes 4 8 12 --runs 5 --output thesis_results/
```

---

## 📁 Repository Structure

```text
tranqui/
├── api/                 # Service Layer (FastAPI)
│   ├── routes.py        # Endpoints (Bridge between Web and Core)
│   └── models.py        # Pydantic Schemas
├── src/                 # Core Library (Pure Business Logic)
│   ├── data/            # Yahoo Finance Ingestion + DuckDB
│   ├── optimization/    # QUBO, Qiskit QAOA, Classical Solvers
│   ├── explanation/     # Text generator with LangChain
│   └── evaluation/      # Financial Metrics (Sharpe, Volatility)
├── web/                 # Frontend Application
│   ├── src/             # React Components (Atomic Design)
│   └── services/        # Backend Connection
├── main.py              # CLI Entrypoint for experimentation
├── requirements.txt     # Python Dependencies
└── README.md            # This file
```

---

## 👤 Author & Credits

**Carlos Gustavo Arroyo Lorenzo**
*Master in Big Data Science & AI - University of Navarra*

This project combines:
*   Modern Portfolio Theory (Markowitz, 1952)
*   Variational Quantum Computing (Farhi et al., 2014)
*   Modern Software Engineering (Clean Architecture)
