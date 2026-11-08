# TrialGuard 🛡️

Evidence-based detection and graduated mitigation of cloud / API free-credit abuse.

## Overview
TrialGuard combines independent signals (identity links, behavioral anomalies, consumption rates, and ML model confidence) to escalate responses dynamically (`ALLOW`, `THROTTLE`, `FLAG`, `BLOCK`), reducing false positives while protecting cloud resources from automated bots, multi-account farming, and credential sharing.

## Tech Stack
- **Language**: Python 3.10+
- **Data & Features**: NumPy, pandas
- **Graph Analysis**: NetworkX (Connected components and node degrees)
- **Models**: scikit-learn (Isolation Forest), XGBoost (Supervised classification)
- **Interfaces**: Streamlit (Cloud Mirror Portal & Security HQ Dashboard)
- **Backend & API**: FastAPI, Uvicorn, Pydantic
- **Testing**: pytest, httpx

## Project Structure
```text
trialguard/
├── data/               # Generated train/test CSV datasets
├── src/                # Pipeline logic (generator, features, graph, models, actions)
├── app/                # Streamlit portals (mock_portal.py) & FastAPI service (api.py)
├── tests/              # Pytest unit tests
├── plots/              # Evaluation charts
├── requirements.txt    # Project dependencies
└── README.md           # Documentation