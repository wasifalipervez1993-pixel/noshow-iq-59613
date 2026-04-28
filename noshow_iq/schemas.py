from __future__ import annotations

from pydantic import BaseModel, Field


class AppointmentRequest(BaseModel):
    PatientId: float = Field(..., example=29872499824296)
    AppointmentID: int = Field(..., example=5642903)
    Gender: str = Field(..., example="F")
    ScheduledDay: str = Field(..., example="2016-04-29T18:38:08Z")
    AppointmentDay: str = Field(..., example="2016-04-29T00:00:00Z")
    Age: int = Field(..., example=62)
    Neighbourhood: str = Field(..., example="JARDIM DA PENHA")
    Scholarship: int = Field(..., example=0)
    Hipertension: int = Field(..., example=1)
    Diabetes: int = Field(..., example=0)
    Alcoholism: int = Field(..., example=0)
    Handcap: int = Field(..., example=0)
    SMS_received: int = Field(..., example=0)


class PredictionResponse(BaseModel):
    risk_level: str
    probability: float
    recommendation: str