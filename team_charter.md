# Team Charter – Crypto Volatility Real-Time Service

## Team Name
TBD (e.g., "Volatility Vanguards")

## Members & Roles
- **Matt Ross** – ML Lead  
  - Owns model selection, feature engineering, evaluation, and MLflow setup.
- **Teammate 2** – Data & Ingestion Lead  
  - Owns Coinbase streaming client, Kafka producers, schemas, and replay scripts.
- **Teammate 3** – API & Serving Lead  
  - Owns FastAPI model-server, `/predict` contract, retries, and load tests.
- **Teammate 4** – MLOps & Monitoring Lead  
  - Owns Prometheus, Grafana dashboards, SLOs, and drift-monitor wiring.
- **Teammate 5** – Reliability & CI/CD Lead  
  - Owns GitHub Actions, linting, basic tests, and release tagging.

## Working Agreements
- Communicate via WhatsApp + GitHub Issues.
- Main branch is always deployable; feature branches for new work.
- PRs require at least one reviewer.
- Any breaking changes must be accompanied by:
  - Updated README (≤10-line setup)
  - Passing CI

## Decision Process
- Technical decisions: ML Lead + relevant owner.
- Ties: resolved in a 15-minute synchronous discussion.
- Anything affecting deployment or grading: discussed and documented in an Issue.

## Deliverable Ownership
- **Model & selection rationale** – Matt  
- **Kafka + replay pipeline** – Data & Ingestion Lead  
- **FastAPI model server** – API & Serving Lead  
- **Prometheus/Grafana/Evidently** – MLOps Lead  
- **CI, load tests, runbook** – Reliability Lead
