from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, Integer, Float, Boolean, ForeignKey, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from src.database.base import Base

class User(Base):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(150), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    alerts: Mapped[List["Alert"]] = relationship(back_populates="user", cascade="all, delete-orphan")

class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (
        Index("idx_alert_created_prediction", "created_at", "prediction"),
        Index("idx_alert_port_prediction", "destination_port", "prediction"),
    )
    
    id: Mapped[int] = mapped_column(primary_key=True)
    prediction: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    destination_port: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    flow_duration: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_fwd_packets: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_backward_packets: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    user: Mapped[Optional["User"]] = relationship(back_populates="alerts")

