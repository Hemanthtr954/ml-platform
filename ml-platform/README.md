# ML Platform

A production ML lifecycle platform covering the full journey from training to monitoring. Built to the same architectural patterns used at Meta (FBLearner), Google (Vertex AI), Uber (Michelangelo), and Amazon (SageMaker).

---

## What This Covers

| Component | What it does | FAANG equivalent |
|---|---|---|
| **Experiment Tracker** | Track hyperparams, metrics, artifacts per run | MLflow, W&B, Google Vizier |
| **Fine-Tuning Pipeline** | LoRA/QLoRA fine-tuning with data validation | Meta's internal LLM fine-tuning |
| **Feature Store** | Online + offline feature serving with point-in-time joins | Uber Michelangelo, LinkedIn Feathr |
| **Model Registry** | Version control for models with promotion workflow | SageMaker Model Registry |
| **Drift Detector** | PSI + KL divergence drift detection with alerting | Evidently, Arize, internal monitors |
| **Safety Evaluator** | Automated safety test suite for LLMs | Meta Llama Guard, Anthropic evals |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        FastAPI Layer                            │
│  /experiments  /models  /features  /monitoring  /eval          │
└─────────────────────────────┬──────────────────────────────────┘
                               │
    ┌──────────────────────────┼───────────────────────────┐
    │                          │                           │
    ▼                          ▼                           ▼
┌──────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Experiment  │    │  Model Registry  │    │   Feature Store      │
│  Tracker     │    │                  │    │                      │
│              │    │  DEV → STAGING   │    │  Online (Redis-like) │
│  params      │    │  → PRODUCTION    │    │  Offline (Parquet)   │
│  metrics     │    │  → ARCHIVED      │    │  Point-in-time joins │
│  artifacts   │    │                  │    │  Hit rate tracking   │
│  git hash    │    │  A/B splits      │    │                      │
│  run compare │    │  Audit trail     │    │                      │
└──────────────┘    └──────────────────┘    └─────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                    Fine-Tuning Pipeline                        │
│                                                               │
│  Data Validation → LoRA Config → QLoRA Training →            │
│  Checkpoint Saving → Eval → Model Registration               │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────┐    ┌─────────────────────────────────────┐
│  Drift Detector  │    │         Safety Evaluator             │
│                  │    │                                      │
│  PSI             │    │  Toxicity / Bias / Hallucination     │
│  KL Divergence   │    │  Instruction Following               │
│  KS Test         │    │  Refusal Quality / Robustness        │
│  Alerting        │    │  Production certification            │
└──────────────────┘    └─────────────────────────────────────┘
```

---

## Quickstart

```bash
git clone https://github.com/yourusername/ml-platform.git
cd ml-platform
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --reload
```

---

## API Examples

### Track an experiment

```bash
# Create a run
curl -X POST http://localhost:8000/experiments/runs \
  -H "Content-Type: application/json" \
  -d '{"experiment_name": "llm_finetuning", "params": {"lr": 0.0002, "lora_r": 16, "epochs": 3}}'

# Log metrics
curl -X POST http://localhost:8000/experiments/runs/abc12345/metrics \
  -d '{"metrics": {"train_loss": 0.45, "eval_loss": 0.51}, "step": 100}'

# Complete run
curl -X POST http://localhost:8000/experiments/runs/abc12345/complete
```

### Register and promote a model

```bash
# Register
curl -X POST http://localhost:8000/models/register \
  -d '{"model_name": "llama_finetuned", "version": "1.0", "artifact_path": "s3://models/v1", "framework": "huggingface"}'

# Promote to staging
curl -X POST http://localhost:8000/models/llama_finetuned/1.0/promote \
  -d '{"to_stage": "staging", "promoted_by": "ci_bot", "reason": "eval loss improved 12%"}'

# Promote to production
curl -X POST http://localhost:8000/models/llama_finetuned/1.0/promote \
  -d '{"to_stage": "production", "promoted_by": "ml_engineer", "reason": "safety eval passed"}'
```

### Feature store

```bash
# Register feature
curl -X POST http://localhost:8000/features/register \
  -d '{"name": "user_engagement_score", "feature_type": "float", "description": "7-day rolling engagement"}'

# Write features
curl -X POST http://localhost:8000/features/materialize \
  -d '{"entity_id": "user_12345", "features": {"user_engagement_score": 0.87}}'

# Get features for inference
curl http://localhost:8000/features/user_12345?features=user_engagement_score
```

### Drift detection

```bash
# Set training baseline
curl -X POST http://localhost:8000/monitoring/baseline \
  -d '{"feature_name": "user_engagement_score", "values": [0.1, 0.5, 0.8, ...]}'

# Check for drift in production
curl -X POST http://localhost:8000/monitoring/drift \
  -d '{"feature_name": "user_engagement_score", "current_values": [0.9, 0.95, ...], "method": "psi"}'
```

### Safety evaluation

```bash
curl -X POST http://localhost:8000/eval/safety \
  -d '{"model_name": "llama_finetuned", "model_version": "1.0", "system_prompt": "You are a helpful assistant."}'
```

**Response:**
```json
{
  "pass_rate": 0.95,
  "overall_score": 0.93,
  "is_safe_for_production": true,
  "critical_failures": 0,
  "by_category": {
    "toxicity": {"pass_rate": 1.0},
    "refusal_quality": {"pass_rate": 0.92},
    "hallucination": {"pass_rate": 0.95}
  }
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
ml-platform/
├── experiment_tracker/
│   └── tracker.py          # MLflow-style tracking: params, metrics, artifacts
├── fine_tuning/
│   └── pipeline.py         # LoRA/QLoRA pipeline with data validation
├── feature_store/
│   └── store.py            # Online + offline store with point-in-time joins
├── model_registry/
│   └── registry.py         # Versioning, stage promotion, A/B splits, audit trail
├── monitoring/
│   └── drift_detector.py   # PSI, KL divergence, data quality, alerting
├── evaluation/
│   └── safety_eval.py      # LLM safety suite: toxicity, bias, refusal quality
├── api/
│   └── main.py             # Unified FastAPI with 20+ endpoints
├── tests/
│   └── test_ml_platform.py # 35+ tests covering all components
├── runtime.txt
├── Dockerfile
└── requirements.txt
```

---

## Key Design Decisions

**Why PSI for drift detection?**
PSI (Population Stability Index) is the industry standard at banks and tech companies because it's interpretable: PSI < 0.1 is safe, 0.1-0.2 needs investigation, > 0.2 requires action. KL divergence is more sensitive but harder to threshold.

**Why point-in-time joins in the feature store?**
Without point-in-time correctness, training uses future feature values that weren't available at prediction time. This creates training/serving skew — the model appears accurate in training but performs poorly in production. This is the #1 silent failure mode in ML systems.

**Why build LoRA from first principles?**
Understanding LoRA mechanics (rank decomposition, alpha scaling, target modules) is essential for fine-tuning decisions. Blindly using a library without understanding the tradeoffs leads to poor results.

**Why a unified promotion workflow?**
DEV → STAGING → PRODUCTION enforces quality gates. You can't skip to production without passing through staging. The audit trail of every promotion (who promoted, when, why) is essential for compliance and debugging.

---

## License

MIT
