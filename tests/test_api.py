from fastapi.testclient import TestClient

from noshow_iq.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()


def test_predict_rejects_invalid_age():
    payload = {
        "PatientId": 1,
        "AppointmentID": 2,
        "Gender": "M",
        "ScheduledDay": "2016-04-29T10:00:00Z",
        "AppointmentDay": "2016-05-03T10:00:00Z",
        "Age": -1,
        "Neighbourhood": "CENTRO",
        "Scholarship": 0,
        "Hipertension": 1,
        "Diabetes": 0,
        "Alcoholism": 0,
        "Handcap": 0,
        "SMS_received": 1,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422