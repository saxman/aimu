"""
aimu.prompts.catalog: Versioned prompt storage backed by SQLite.

Prompts are keyed by (name, model_id) and versioned automatically on each
store. Metrics are stored as plain JSON dicts so they can be inspected without
deserializing a pickle.

Usage::

    from aimu.prompts import Prompt, PromptCatalog

    with PromptCatalog("prompts.db") as catalog:
        p = Prompt(name="classifier", model_id="llama3.1", prompt="Classify...")
        catalog.store_prompt(p)          # version auto-assigned (1)
        latest = catalog.retrieve_last("classifier", "llama3.1")
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, create_engine, desc
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Prompt(Base):
    __tablename__ = "prompts"
    id = Column(Integer, primary_key=True)
    parent_id = Column(Integer, ForeignKey("prompts.id"))
    name = Column(String, nullable=False)
    prompt = Column(String, nullable=False)
    model_id = Column(String)
    version = Column(Integer)
    mutation_prompt = Column(String)
    reasoning_prompt = Column(String)
    metrics = Column(JSON)
    created_at = Column(DateTime)


class PromptCatalog:
    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)

        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def close(self):
        self.session.close()
        self.engine.dispose()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.session.close()

    def store_prompt(self, prompt: Prompt) -> None:
        """Store a prompt, auto-assigning version and created_at if not set."""
        if prompt.version is None:
            last = self.retrieve_last(prompt.name, prompt.model_id)
            prompt.version = (last.version + 1) if last else 1
        if prompt.created_at is None:
            prompt.created_at = datetime.now(timezone.utc)
        try:
            self.session.add(prompt)
        except Exception:
            self.session.rollback()
            raise
        else:
            self.session.commit()

    def retrieve_last(self, name: str, model_id: str) -> Prompt | None:
        """Return the highest-versioned prompt for the given name and model."""
        return (
            self.session.query(Prompt)
            .filter(Prompt.name == name, Prompt.model_id == model_id)
            .order_by(desc(Prompt.version))
            .first()
        )

    def retrieve_all(self, name: str, model_id: str) -> list[Prompt]:
        """Return all prompt versions for the given name and model, newest first."""
        return (
            self.session.query(Prompt)
            .filter(Prompt.name == name, Prompt.model_id == model_id)
            .order_by(desc(Prompt.version))
            .all()
        )

    def delete_all(self, name: str, model_id: str) -> int:
        """Delete all versions for the given name and model. Returns row count."""
        rows_deleted = 0
        try:
            rows_deleted = self.session.query(Prompt).filter(Prompt.name == name, Prompt.model_id == model_id).delete()
        except Exception:
            self.session.rollback()
            raise
        else:
            self.session.commit()
        return rows_deleted

    def retrieve_model_ids(self) -> list[str]:
        """Return a deduplicated list of all stored model IDs."""
        return [x.model_id for x in self.session.query(Prompt.model_id).distinct()]

    def retrieve_names(self) -> list[str]:
        """Return a deduplicated list of all stored prompt names."""
        return [x.name for x in self.session.query(Prompt.name).distinct()]
