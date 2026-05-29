from __future__ import annotations

from pydantic import BaseModel, Field


class AppointmentRequest(BaseModel):
    PatientId: float = Field(
        ...,
        json_schema_extra={"example": 29872499824296},
    )
    AppointmentID: int = Field(
        ...,
        json_schema_extra={"example": 5642903},
    )
    Gender: str = Field(
        ...,
        json_schema_extra={"example": "F"},
    )
    ScheduledDay: str = Field(
        ...,
        json_schema_extra={"example": "2016-04-29T18:38:08Z"},
    )
    AppointmentDay: str = Field(
        ...,
        json_schema_extra={"example": "2016-04-29T00:00:00Z"},
    )
    Age: int = Field(
        ...,
        ge=0,
        le=120,
        json_schema_extra={"example": 62},
    )
    Neighbourhood: str = Field(
        ...,
        json_schema_extra={"example": "JARDIM DA PENHA"},
    )
    Scholarship: int = Field(
        ...,
        ge=0,
        le=1,
        json_schema_extra={"example": 0},
    )
    Hipertension: int = Field(
        ...,
        ge=0,
        le=1,
        json_schema_extra={"example": 1},
    )
    Diabetes: int = Field(
        ...,
        ge=0,
        le=1,
        json_schema_extra={"example": 0},
    )
    Alcoholism: int = Field(
        ...,
        ge=0,
        le=1,
        json_schema_extra={"example": 0},
    )
    Handcap: int = Field(
        ...,
        ge=0,
        le=4,
        json_schema_extra={"example": 0},
    )
    SMS_received: int = Field(
        ...,
        ge=0,
        le=1,
        json_schema_extra={"example": 0},
    )


class PredictionResponse(BaseModel):
    risk_level: str
    probability: float
    recommendation: str
    confidence: str
    action_priority: str
