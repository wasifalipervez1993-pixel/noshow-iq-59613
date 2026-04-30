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

NoShowIQ is a production-ready MLOps system that predicts whether a patient will miss a clinic appointment and provides actionable recommendations for healthcare providers.

---

## 🌐 Live Deployment

Hugging Face Space:

https://wasifalipervez123-noshow-iq-59613.hf.space

API documentation:

https://wasifalipervez123-noshow-iq-59613.hf.space/docs

---

## 📦 Python Package — TestPyPI

https://test.pypi.org/project/noshow-iq-59613/

Install:

```bash
pip install -i https://test.pypi.org/simple/ noshow-iq-59613
```

---

## 🐳 Docker Hub Image

https://hub.docker.com/r/wasifalipervez1993/noshow-iq-59613

Pull image:

```bash
docker pull wasifalipervez1993/noshow-iq-59613:latest
```

---

## 🧠 Model Overview

- Model: **HistGradientBoostingClassifier**
- Problem type: **Binary classification**
- Target classes:
  - `show`
  - `no_show`
- Class imbalance handling:
  - `class_weight="balanced"`
  - threshold tuning based on no-show F1-score
- Model artifact saved and loaded using `joblib`

The system focuses on detecting likely no-show patients rather than maximizing accuracy only, because the dataset is imbalanced.

---

## 📊 Model Performance

| Metric | Value |
|---|---:|
| Accuracy | ~63% |
| No-show Precision | ~32% |
| No-show Recall | ~73% |
| No-show F1-score | ~0.44 |

Accuracy alone is misleading because most patients attend appointments. Therefore, no-show recall, no-show precision, and no-show F1-score are more important for this problem.

---

## ⚙️ Feature Engineering

Important engineered features include:

- `days_in_advance` — days between scheduling and appointment date
- `wait_time_bin`
- `appointment_weekday`
- `scheduled_weekday`
- `is_same_day`
- `is_child`
- `is_senior`
- `has_chronic_condition`

---

## 🚀 API Endpoints

### GET `/`

Root endpoint.

Example response:

```json
{
  "message": "Welcome to NoShowIQ API",
  "docs": "/docs",
  "health": "/health"
}
```

---

### GET `/health`

Checks whether the API is running and whether the model is loaded.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### POST `/predict`

Accepts one raw appointment record in the same format as the original dataset and returns a no-show risk prediction.

Example request:

```json
{
  "PatientId": 1,
  "AppointmentID": 2,
  "Gender": "M",
  "ScheduledDay": "2016-04-29T10:00:00Z",
  "AppointmentDay": "2016-05-03T10:00:00Z",
  "Age": 45,
  "Neighbourhood": "CENTRO",
  "Scholarship": 0,
  "Hipertension": 1,
  "Diabetes": 0,
  "Alcoholism": 0,
  "Handcap": 0,
  "SMS_received": 1
}
```

Example response:

```json
{
  "risk_level": "high",
  "probability": 0.6483,
  "recommendation": "Call patient and consider controlled overbooking.",
  "confidence": "medium",
  "action_priority": "urgent"
}
```

Each successful prediction is logged to MongoDB.

---

### POST `/predict-batch`

Accepts multiple appointment records and returns predictions for all records.

---

### GET `/history`

Returns the latest 20 prediction records stored in MongoDB.

---

### GET `/stats`

Returns MongoDB aggregation-based statistics.

Example response:

```json
{
  "total_predictions": 87,
  "high_risk_count": 31,
  "medium_risk_count": 0,
  "low_risk_count": 56,
  "average_probability": 0.64,
  "last_trained": "2026-04-29T09:00:00Z"
}
```

The `/stats` endpoint uses a MongoDB aggregation pipeline instead of Python-side counting.

---

## 🗄️ MongoDB Design

### `predictions` collection

Each `/predict` call stores:

- timestamp
- raw input
- cleaned features
- risk level
- probability
- recommendation

### `training_runs` collection

Each model training stores:

- timestamp
- training size
- test size
- selected model
- imbalance technique
- decision threshold
- precision, recall, and F1-score for both classes

---

## 🐳 Docker Setup

Build:

```bash
docker build -t noshow-iq-59613 .
```

Run with Docker Compose:

```bash
docker compose up --build
```

Services:

- FastAPI application
- MongoDB
- Mongo Express on port `8081`

The Docker image is optimized to stay under the 300 MB requirement.

---

## 🔄 CI/CD Pipeline

The GitHub Actions workflow runs automatically on every push to `main`.

Pipeline order:

1. Lint with flake8
2. Run pytest
3. Build Docker image
4. Push Docker image to Docker Hub
5. Trigger Hugging Face Space rebuild

Required secrets are stored securely in GitHub Actions:

- `DOCKER_USERNAME`
- `DOCKER_TOKEN`
- `HF_TOKEN`
- `HF_SPACE_ID`

---

## 🧪 Testing

Run tests:

```bash
pytest -v
```

The project includes more than 6 pytest tests covering preprocessing, model pipeline construction, API health, and feature generation.

---

## 🔍 Smoke Test

Run:

```bash
python smoke_test.py https://wasifalipervez123-noshow-iq-59613.hf.space
```

Expected output:

```text
PASS
```

---

## 🔐 Security

- No secrets are hardcoded in source code
- `MONGO_URI` is provided through environment variables
- Hugging Face uses `MONGO_URI` as a Space Secret
- GitHub Actions uses repository secrets for deployment credentials

---

## 📋 Submission Checklist

- Public repository with README and CI badge
- 8+ conventional commits
- Pull request created and merged
- No secrets committed
- TestPyPI package published
- Docker image pushed to Docker Hub
- Docker Compose working with app, MongoDB, and mongo-express
- MongoDB Atlas collections populated
- Hugging Face deployment working
- Smoke test passing

---

## 👤 Author

Wasif Ali Pervez  
MS Data Science — MLOps  
Spring 2026
