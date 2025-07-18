# Real-Time Data Drift Monitoring

Exploring what a real-time data drift monitoring solution could look like within MLOps.

## How It Works

1. A baseline dataset (reference) is loaded at startup.
2. Incoming feature data is buffered in a rolling window.
3. Once the buffer is full:
   - A KS test is run per feature.
   - P-values and drift flags are recorded.
   - Metrics are exposed to Prometheus.
4. Grafana visualizes:
   - Number of features drifting
   - Feature-level p-values & drift flags
   - Last drift timestamp
   - Historical drift trends


## Running the Project

### 1. Install Dependencies

Use [`uv`](https://github.com/astral-sh/uv) to manage the Python environment:

```bash
uv venv
uv sync
```

### 2. Start All Services
Use Docker Compose to spin up  the Model Server, Metric Server, Prometheus and Grafana:

```bash
docker compose up --build
```

### 3. Run the Drift Monitor
To simulate a live data stream:
- Without Drift (Normal Scenario):
```bash
uv run run.py --drift false
```

- With Drift (Simulated Drift Scenario):
```bash
uv run run.py --drift true
```