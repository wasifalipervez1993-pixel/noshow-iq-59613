---
title: NoShowIQ 59613
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# 🏥 NoShowIQ — Appointment No-Show Prediction System

![CI/CD](https://github.com/wasifalipervez1993-pixel/noshow-iq-59613/actions/workflows/ci-cd.yml/badge.svg)

NoShowIQ is a **production-ready MLOps system** that predicts whether a patient will miss a clinic appointment and provides actionable recommendations for healthcare providers.

---

## 🌐 Live Deployment

👉 Hugging Face Space:  
https://wasifalipervez123-noshow-iq-59613.hf.space

---

## 📦 Python Package (TestPyPI)

👉 https://test.pypi.org/project/noshow-iq-59613/

Install locally:

```bash
pip install -i https://test.pypi.org/simple/ noshow-iq-59613
```

---

## 🧠 Model Overview

- Model: **HistGradientBoostingClassifier**
- Handles class imbalance using:
  - `class_weight="balanced"`
  - Threshold tuning (F1-optimized)
- Designed to prioritize:
  - **No-show detection (recall)** over accuracy

---

## 📊 Model Performance

| Metric | Value |
|------|------|
| Accuracy | ~63% |
| No-show Recall | ~73% |
| No-show Precision | ~32% |
| No-show F1 | ~0.44 |

⚠️ Accuracy is misleading due to dataset imbalance (~80% show vs 20% no-show)

---

## ⚙️ Feature Engineering

Key features:
- `days_in_advance` ✅ (required)
- `wait_time_bin`
- `is_same_day`
- `is_child`, `is_senior`
- `has_chronic_condition`

---

## 🚀 API Endpoints

### GET /health

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### POST /predict

Input:
```json
{
  "gender": "F",
  "age": 25
}
```

Output:
```json
{
  "risk_level": "high",
  "probability": 0.82,
  "confidence": "high",
  "recommendation": "Call patient + consider controlled overbooking."
}
```

---

### POST /predict-batch

Supports bulk predictions.

---

### GET /history

Returns last 20 predictions.

---

### GET /stats

MongoDB aggregation-based statistics:

```json
{
  "total_predictions": 87,
  "high_risk_count": 31,
  "avg_probability": 0.64,
  "drift_status": "STABLE"
}
```

---

## 🗄️ MongoDB Design

### predictions collection

Stores:
- timestamp
- raw input
- cleaned features
- risk level
- probability
- recommendation

### training_runs collection

Stores:
- training size
- classification metrics (precision, recall, F1)
- imbalance handling technique
- threshold

---

## 🐳 Docker Setup

Build:

```bash
docker build -t noshow-iq-59613 .
```

Run:

```bash
docker compose up
```

Services:
- API
- MongoDB
- Mongo Express (port 8081)

---

## 🔄 CI/CD Pipeline

Runs on push to main:

- flake8 linting
- pytest testing
- Docker build
- Docker push
- Hugging Face rebuild

---

## 🧪 Testing

Run:

```bash
pytest -v
```

---

## 🔍 Smoke Test

```bash
python smoke_test.py https://wasifalipervez123-noshow-iq-59613.hf.space
```

Expected output:

```
PASS
```

---

## 🔐 Security

- No secrets committed
- MONGO_URI handled via environment variables

---

## 📊 Key Features

✔ FastAPI production API  
✔ Imbalanced ML handling  
✔ Threshold optimization  
✔ MongoDB logging  
✔ Drift detection  
✔ CI/CD automation  
✔ Docker deployment  

---

## 📋 Submission Checklist

- ✔ Public repo with README and CI badge  
- ✔ 8+ commits  
- ✔ Pull request created and merged  
- ✔ No secrets in repo  
- ✔ CI/CD pipeline green  
- ✔ TestPyPI package published  
- ✔ Docker image published  
- ✔ docker-compose working  
- ✔ MongoDB collections populated  
- ✔ Hugging Face deployed  
- ✔ Smoke test PASS  

---

## 👤 Author

Wasif Ali Pervez  
MS Data Science — MLOps (Spring 2026)

---