# 🧘 Tranqui (QuantumCoach)
> **Financial Peace through Quantum Intelligence.**
>
> *Un sistema híbrido cuántico-clásico que democratiza la optimización de carteras para la Gen Z.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Framework: FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Frontend: Vite+React](https://img.shields.io/badge/Web-Vite%2BReact-646CFF.svg)](https://vitejs.dev/)

---

## 🌗 The Dual Vision (El Doble Enfoque)

Este proyecto (TFM) aborda el problema de la inversión retail desde dos perspectivas complementarias:

### 1. La Perspectiva Científica (Thesis Core)
**"Benchmarking de Algoritmos Cuánticos en Finanzas"**
Un motor de comparación riguroso entre:
*   **Classical Solvers**: Brute Force (exacto para $N < 20$) y Greedy (aproximado rápido).
*   **Quantum Solvers**: QAOA (Quantum Approximate Optimization Algorithm) ejecutado en simuladores Qiskit/Aer.
*   **Objetivo**: Demostrar la viabilidad del "Hybrid Logic" -> Usar clásico hoy, estar preparado para la ventaja cuántica mañana.

### 2. La Perspectiva de Producto (UX/Gen Z)
**"Tranqui: Tu Coach Financiero Zen"**
La tecnología cuántica es intimidante; las finanzas también.
*   **La Solución**: Una interfaz "Dark/Neon Premium" que oculta la complejidad matemática bajo una capa de bienestar financiero.
*   **El Coach**: Un sistema LLM (LangChain) que traduce "Sharpe Ratios" y "Fronteras Eficientes" a lenguaje natural empático. *"No te preocupes por la volatilidad, tu cartera está blindada matemáticamente."*

---

## 🏗️ Arquitectura del Sistema

El sistema sigue una arquitectura modular estricta para separar la lógica de investigación (Core) de la aplicación de usuario (Web/API).

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

### Componentes Clave

| Módulo | Tecnología | Función |
|--------|------------|---------|
| **`src.optimization`** | Qiskit, Numpy | El corazón matemático. Transforma problemas de Markowitz en Hamiltonianos (QUBO) para QAOA. |
| **`src.explanation`** | LangChain | Convierte vectores numéricos de decisión en narrativa financiera personalizada. |
| **`api/`** | FastAPI | Expone la potencia del Core a través de endpoints REST (`/optimize`, `/chat`, `/benchmark`). |
| **`web/`** | React + Vite | Dashboard interactivo con estética "Glassmorphism" y visualización de datos en tiempo real. |

---

## 🚀 Quick Start (Para Desarrolladores)

### Prerrequisitos
- Python 3.10+
- Node.js 18+ (para el frontend)
- Clave de API para LLM (OpenRouter/OpenAI)

### 1. Core & Backend (Python)

```bash
# 1. Clonar y preparar entorno
git clone https://github.com/username/tranqui-quantum.git
cd tranqui-quantum
python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate en Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configuración
cp .env.example .env
# [!] Edita .env y añade tus claves API

# 4. Probar el Core (CLI)
# Ejecuta una optimización rápida de 3 activos españoles
python main.py --tickers SAN.MC ITX.MC IBE.MC --risk-aversion 0.5 --explain

# 5. Levantar el Servidor API
./start_app.sh
# O manualmente: uvicorn api.main:app --reload
```

### 2. Frontend (React)

```bash
cd web

# 1. Instalar dependencias
npm install

# 2. Iniciar servidor de desarrollo
npm run dev

# Accede a http://localhost:5173
```

---

## 🔬 Scientific Validation (Thesis Experiments)

El proyecto incluye un módulo de benchmarking (`src.evaluation.benchmark`) para validar el rendimiento de QAOA frente a métodos clásicos.

**Resultados Preliminares (Simulación):**
*   **Pequeña Escala ($N=4-8$)**: QAOA alcanza el óptimo global (GAP < 1%) consistente con > p=2 capas.
*   **Media Escala ($N=12-16$)**: El ruido de simulación y la profundidad del circuito requieren optimizadores clásicos híbridos (COBYLA/SPSA) más robustos.
*   **Velocidad**: Clásico es órdenes de magnitud más rápido actualmente (milisegundos vs segundos/minutos), justificando el enfoque híbrido de "Tranqui": *Usar Clásico para respuesta real-time, Cuántico para análisis profundo offline.*

Para reproducir los experimentos de la tesis:
```bash
python main.py --benchmark --sizes 4 8 12 --runs 5 --output thesis_results/
```

---

## 📁 Estructura del Repositorio

```text
tranqui/
├── api/                 # Capa de Servicio (FastAPI)
│   ├── routes.py        # Endpoints (Bridge entre Web y Core)
│   └── models.py        # Pydantic Schemas
├── src/                 # Core Library (Lógica de Negocio Pura)
│   ├── data/            # Ingesta de Yahoo Finance + DuckDB
│   ├── optimization/    # QUBO, Qiskit QAOA, Solvers Clásicos
│   ├── explanation/     # Generador de texto con LangChain
│   └── evaluation/      # Métricas Financieras (Sharpe, Volatilidad)
├── web/                 # Frontend Application
│   ├── src/             # React Components (Atomic Design)
│   └── services/        # Conexión con Backend
├── main.py              # CLI Entrypoint para experimentación
├── requirements.txt     # Dependencias Python
└── README.md            # Este archivo
```

---

## 👤 Autor & Créditos

**[Tu Nombre]**
*Master in Big Data Science & AI - Universidad de Navarra*

Este proyecto combina:
*   Teoría Moderna de Carteras (Markowitz, 1952)
*   Computación Cuántica Variacional (Farhi et al., 2014)
*   Ingeniería de Software Moderna (Clean Architecture)
