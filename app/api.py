# api.py

import time
import json
import logging
from pathlib import Path
from collections import defaultdict
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pipeline import run_pipeline

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("api")

app = FastAPI(title="Hawker NLP Pipeline")

# ── Model Info ─────────────────────────────────────────────────────────────────
MODEL_INFO_PATHS = {
    "ner_model": Path("model/ner_model_v1/model_info.json"),
    "rvi_model": Path("model/rvi_model_v1/model_info.json"),
}

def load_model_info():
    info = {}
    for name, path in MODEL_INFO_PATHS.items():
        if path.exists():
            with open(path) as f:
                info[name] = json.load(f)
            logger.info(f"Loaded model info for {name} ({info[name]['version']})")
        else:
            logger.warning(f"model_info.json not found for {name} at {path}")
            info[name] = {"error": "model_info.json not found"}
    return info

model_info = load_model_info()

# ── In-Memory Metrics ──────────────────────────────────────────────────────────
metrics = {
    "total_requests": 0,
    "total_latency": 0.0,
    "intent_counts": defaultdict(int),
    "latency_history": [],   # last 50 requests
}


# ── Input Schema ───────────────────────────────────────────────────────────────
class ReviewInput(BaseModel):
    review: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    logger.info("Health check called")
    return {"status": "running"}


@app.get("/model-info")
def get_model_info():
    logger.info("Model info endpoint called")
    return model_info


@app.post("/analyze")
def analyze(data: ReviewInput):
    logger.info(f"Received review: '{data.review[:80]}{'...' if len(data.review) > 80 else ''}'")

    start = time.time()
    result = run_pipeline(data.review)
    elapsed = round(time.time() - start, 3)

    # Update metrics
    metrics["total_requests"] += 1
    metrics["total_latency"] += elapsed
    metrics["intent_counts"][result["revisit_intent"]] += 1
    metrics["latency_history"].append(elapsed)
    if len(metrics["latency_history"]) > 50:
        metrics["latency_history"].pop(0)

    logger.info(f"Pipeline completed in {elapsed:.3f}s | "
                f"foods={result['food_entities']} | "
                f"intent={result['revisit_intent']} | "
                f"confidence={result['confidence']:.2f}")

    return result


@app.get("/metrics")
def get_metrics():
    total = metrics["total_requests"]
    avg_latency = round(metrics["total_latency"] / total, 3) if total > 0 else 0
    return {
        "total_requests": total,
        "average_latency_seconds": avg_latency,
        "intent_distribution": dict(metrics["intent_counts"]),
        "recent_latencies": metrics["latency_history"][-10:],
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    total = metrics["total_requests"]
    avg_latency = round(metrics["total_latency"] / total, 3) if total > 0 else 0
    intent_counts = dict(metrics["intent_counts"])
    recent_latencies = metrics["latency_history"][-20:]

    intent_labels = list(intent_counts.keys())
    intent_values = list(intent_counts.values())
    latency_labels = list(range(1, len(recent_latencies) + 1))

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hawker NLP Dashboard</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: -apple-system, sans-serif; background: #f0f2f5; color: #333; }}
            header {{ background: #1a1a2e; color: white; padding: 20px 40px; }}
            header h1 {{ font-size: 1.5rem; }}
            header p {{ font-size: 0.85rem; opacity: 0.6; margin-top: 4px; }}
            .container {{ max-width: 1100px; margin: 30px auto; padding: 0 20px; }}
            .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px; }}
            .card {{ background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
            .card .label {{ font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }}
            .card .value {{ font-size: 2.2rem; font-weight: 700; margin-top: 8px; color: #1a1a2e; }}
            .card .unit {{ font-size: 0.9rem; color: #aaa; }}
            .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .chart-card {{ background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
            .chart-card h3 {{ font-size: 0.95rem; color: #555; margin-bottom: 16px; }}
            .refresh {{ margin-top: 24px; text-align: right; }}
            .refresh a {{ font-size: 0.85rem; color: #4a90d9; text-decoration: none; }}
        </style>
    </head>
    <body>
        <header>
            <h1>🍜 Hawker NLP — Monitoring Dashboard</h1>
            <p>Live metrics from the NLP inference pipeline</p>
        </header>
        <div class="container">
            <div class="cards">
                <div class="card">
                    <div class="label">Total Requests</div>
                    <div class="value">{total}</div>
                </div>
                <div class="card">
                    <div class="label">Avg Latency</div>
                    <div class="value">{avg_latency}<span class="unit"> s</span></div>
                </div>
                <div class="card">
                    <div class="label">Intent Types Seen</div>
                    <div class="value">{len(intent_labels)}</div>
                </div>
            </div>
            <div class="charts">
                <div class="chart-card">
                    <h3>Revisit Intent Distribution</h3>
                    <canvas id="intentChart"></canvas>
                </div>
                <div class="chart-card">
                    <h3>Recent Request Latencies (last 20)</h3>
                    <canvas id="latencyChart"></canvas>
                </div>
            </div>
            <div class="refresh">
                <a href="/dashboard">↻ Refresh</a> &nbsp;|&nbsp;
                <a href="/metrics">View raw JSON</a>
            </div>
        </div>
        <script>
            new Chart(document.getElementById('intentChart'), {{
                type: 'doughnut',
                data: {{
                    labels: {intent_labels},
                    datasets: [{{ 
                        data: {intent_values},
                        backgroundColor: ['#4a90d9', '#e85d5d', '#50c878', '#f4a261'],
                    }}]
                }},
                options: {{ plugins: {{ legend: {{ position: 'bottom' }} }} }}
            }});

            new Chart(document.getElementById('latencyChart'), {{
                type: 'line',
                data: {{
                    labels: {latency_labels},
                    datasets: [{{
                        label: 'Latency (s)',
                        data: {recent_latencies},
                        borderColor: '#4a90d9',
                        backgroundColor: 'rgba(74,144,217,0.1)',
                        fill: true,
                        tension: 0.4,
                    }}]
                }},
                options: {{
                    scales: {{
                        y: {{ beginAtZero: true, title: {{ display: true, text: 'seconds' }} }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html