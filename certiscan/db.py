"""Persistence for processed certificates. SQLite by default, PostgreSQL via DATABASE_URL."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Certificate(Base):
    __tablename__ = "certificates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    image_path: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), default="extracted")  # extracted | approved
    compliance: Mapped[str] = mapped_column(String(32), default="review")  # pass | fail | review | unknown_spec
    extracted_json: Mapped[str] = mapped_column(Text)
    corrected_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    regions_json: Mapped[str] = mapped_column(Text, default="[]")
    timings_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    @property
    def document(self) -> dict:
        return json.loads(self.corrected_json or self.extracted_json)

    @property
    def regions(self) -> list:
        return json.loads(self.regions_json)


def make_session_factory(url: Optional[str] = None) -> sessionmaker:
    url = url or os.getenv("DATABASE_URL", "sqlite:///certiscan.db")
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def find_by_hash(session: Session, sha256: str) -> Optional[Certificate]:
    return session.scalar(select(Certificate).where(Certificate.sha256 == sha256))
