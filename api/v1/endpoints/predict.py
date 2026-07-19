from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Optional
from sqlalchemy.orm import Session

from src.services.inference_service import inference_service
from src.database.session import get_db
from src.database.models import User, Alert
from src.services.auth_service import decode_access_token

router = APIRouter()
security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    token = credentials.credentials
    username = decode_access_token(token)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token"
        )
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    return user

class NetworkFlowRecord(BaseModel):
    destination_port: Optional[int] = Field(default=None, alias="Destination Port", ge=0, le=65535)
    flow_duration: Optional[int] = Field(default=None, alias="Flow Duration", ge=0)
    total_fwd_packets: Optional[int] = Field(default=None, alias="Total Fwd Packets", ge=0)
    total_backward_packets: Optional[int] = Field(default=None, alias="Total Backward Packets", ge=0)

    model_config = {
        "extra": "allow",
        "populate_by_name": True
    }

class PredictionResponse(BaseModel):
    prediction: str
    confidence: Optional[float] = None

@router.post("/predict", response_model=List[PredictionResponse], status_code=status.HTTP_200_OK)
def predict_threats(
    payload: Union[List[NetworkFlowRecord], NetworkFlowRecord],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Standardize payload to list of dicts with correct raw feature keys (by_alias=True)
        if isinstance(payload, list):
            records = [rec.model_dump(by_alias=True, exclude_none=True) for rec in payload]
        else:
            records = [payload.model_dump(by_alias=True, exclude_none=True)]

        results = inference_service.predict(records)

        # Store each prediction in the database as an Alert
        for i, (pred, conf) in enumerate(results):
            rec = records[i]
            alert = Alert(
                prediction=pred,
                confidence=conf,
                destination_port=rec.get("Destination Port"),
                flow_duration=rec.get("Flow Duration"),
                total_fwd_packets=rec.get("Total Fwd Packets"),
                total_backward_packets=rec.get("Total Backward Packets"),
                user_id=current_user.id
            )
            db.add(alert)
        db.commit()

        return [
            PredictionResponse(prediction=pred, confidence=conf)
            for pred, conf in results
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

class AlertResponse(BaseModel):
    id: int
    prediction: str
    confidence: Optional[float] = None
    destination_port: Optional[int] = None
    flow_duration: Optional[int] = None
    total_fwd_packets: Optional[int] = None
    total_backward_packets: Optional[int] = None
    created_at: Any
    user_id: Optional[int] = None

    class Config:
        from_attributes = True

@router.get("/alerts", response_model=List[AlertResponse], status_code=status.HTTP_200_OK)
def get_recent_alerts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = 100
):
    try:
        alerts = db.query(Alert).order_by(Alert.created_at.desc()).limit(limit).all()
        return alerts
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch alerts: {str(e)}"
        )

