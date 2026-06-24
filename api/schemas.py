"""
schemas.py
----------
Pydantic models that define the request/response "contracts" for the API.

FastAPI uses these to:
    - validate incoming JSON automatically,
    - generate the interactive Swagger docs at /docs,
    - serialise responses with the right types.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    """One loan applicant's financial profile (the POST /predict body)."""

    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    job: int = Field(
        ..., ge=0, le=3, description="Job skill level: 0=unskilled ... 3=highly skilled"
    )
    credit_amount: float = Field(..., gt=0, description="Requested credit amount")
    duration: int = Field(..., gt=0, le=120, description="Loan duration in months")

    housing: Literal["own", "rent", "free"]
    saving_accounts: Literal["little", "moderate", "quite rich", "rich"]
    checking_account: Literal["little", "moderate", "rich"]
    purpose: Literal[
        "car",
        "furniture/equipment",
        "radio/TV",
        "domestic appliances",
        "repairs",
        "education",
        "business",
        "vacation/others",
    ]

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "job": 2,
                "credit_amount": 4500,
                "duration": 24,
                "housing": "own",
                "saving_accounts": "moderate",
                "checking_account": "moderate",
                "purpose": "car",
            }
        }
    }


class PredictionResponse(BaseModel):
    """The result returned by POST /predict."""

    prediction: Literal["good", "bad"] = Field(..., description="Predicted credit risk class")
    risk_score: float = Field(..., description="Probability the applicant is 'bad' (0-1)")
    risk_level: Literal["Low", "Medium", "High"] = Field(
        ..., description="Bucketed risk band derived from risk_score"
    )


class HealthResponse(BaseModel):
    """The result returned by GET /health."""

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """The result returned by GET /model-info."""

    model_name: str
    model_type: str
    version: str
    purpose: str
    project: str
    trained_at: Optional[str] = None
    metrics: Optional[dict] = None
    model_source: Optional[str] = Field(
        None, description="Where the API loaded the model from: 'registry' or 'file'"
    )
    registry_version: Optional[str] = Field(
        None, description="MLflow Model Registry version number (when loaded from registry)"
    )
    registry_stage: Optional[str] = Field(
        None, description="MLflow alias, e.g. Production (when loaded from registry)"
    )
