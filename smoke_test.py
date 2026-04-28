from __future__ import annotations

import sys

import requests


SAMPLE_RECORD = {
    "PatientId": 29872499824296,
    "AppointmentID": 5642903,
    "Gender": "F",
    "ScheduledDay": "2016-04-29T18:38:08Z",
    "AppointmentDay": "2016-04-29T00:00:00Z",
    "Age": 62,
    "Neighbourhood": "JARDIM DA PENHA",
    "Scholarship": 0,
    "Hipertension": 1,
    "Diabetes": 0,
    "Alcoholism": 0,
    "Handcap": 0,
    "SMS_received": 0,
}


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def main():
    if len(sys.argv) != 2:
        print("Usage: python smoke_test.py <BASE_URL>")
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")

    health = requests.get(f"{base_url}/health", timeout=15)
    check(health.status_code == 200, "/health failed")

    prediction = requests.post(
        f"{base_url}/predict",
        json=SAMPLE_RECORD,
        timeout=15,
    )
    check(prediction.status_code == 200, "/predict failed")
    check("risk_level" in prediction.json(), "risk_level missing")
    check("probability" in prediction.json(), "probability missing")

    stats = requests.get(f"{base_url}/stats", timeout=15)
    check(stats.status_code == 200, "/stats failed")
    check("total_predictions" in stats.json(), "total_predictions missing")

    print("PASS")


if __name__ == "__main__":
    main()