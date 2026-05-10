"""
catalog.py — Load SHL product catalog from JSON and build an in-memory
FAISS index for semantic retrieval. Built once at startup.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CATALOG_PATH = Path(os.getenv("CATALOG_PATH", "data/product.json"))
EMBED_MODEL   = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Map raw "keys" strings → short codes for the API response
CATEGORY_CODE: dict[str, str] = {
    "Knowledge & Skills":           "K",
    "Ability & Aptitude":           "A",
    "Personality & Behavior":       "P",
    "Biodata & Situational Judgment": "B",
    "Competencies":                 "C",
    "Development & 360":            "D",
    "Assessment Exercises":         "E",
}


@dataclass
class Assessment:
    entity_id:        str
    name:             str
    url:              str
    test_type:        str          # single letter code, first category wins
    description:      str
    job_levels:       list[str]
    languages:        list[str]
    duration_minutes: Optional[int]
    remote_testing:   bool
    adaptive:         bool
    categories:       list[str]    # all raw category labels

    def embed_text(self) -> str:
        """Rich text blob for FAISS embedding."""
        parts = [
            f"Assessment: {self.name}",
            f"Categories: {', '.join(self.categories)}",
            f"Description: {self.description}",
        ]
        if self.job_levels:
            parts.append(f"Job levels: {', '.join(self.job_levels)}")
        if self.duration_minutes:
            parts.append(f"Duration: {self.duration_minutes} minutes")
        if self.adaptive:
            parts.append("Adaptive/IRT scoring")
        return ". ".join(parts)

    def to_api_dict(self) -> dict:
        return {"name": self.name, "url": self.url, "test_type": self.test_type}

    def to_detail_dict(self) -> dict:
        return {
            "name":             self.name,
            "url":              self.url,
            "test_type":        self.test_type,
            "categories":       self.categories,
            "description":      self.description,
            "job_levels":       self.job_levels,
            "languages":        self.languages,
            "duration_minutes": self.duration_minutes,
            "remote_testing":   self.remote_testing,
            "adaptive":         self.adaptive,
        }


def _parse_duration(raw: str) -> Optional[int]:
    if not raw:
        return None
    m = re.search(r"(\d+)", raw)
    return int(m.group(1)) if m else None


def _load_raw_catalog() -> list[dict]:
    """Load & clean the catalog JSON (handles embedded bare CR in strings)."""
    raw = CATALOG_PATH.read_bytes().decode("utf-8", errors="replace")
    # Fix literal \r\n splits inside JSON string values
    cleaned = re.sub(r"\r\n(?!\s*[{}\[\],\"\\\\])", " ", raw)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "")
    return json.loads(cleaned)


def build_catalog() -> list[Assessment]:
    records = _load_raw_catalog()
    assessments: list[Assessment] = []
    for r in records:
        if r.get("status") != "ok":
            continue
        categories = r.get("keys") or []
        # Derive a single-letter type from the first recognised category
        test_type = "K"
        for cat in categories:
            if cat in CATEGORY_CODE:
                test_type = CATEGORY_CODE[cat]
                break

        assessments.append(Assessment(
            entity_id        = str(r.get("entity_id", "")),
            name             = r.get("name", "Unknown").strip(),
            url              = r.get("link", "https://www.shl.com/solutions/products/product-catalog/"),
            test_type        = test_type,
            description      = (r.get("description") or "").strip(),
            job_levels       = r.get("job_levels") or [],
            languages        = r.get("languages") or [],
            duration_minutes = _parse_duration(r.get("duration", "")),
            remote_testing   = str(r.get("remote", "")).lower() == "yes",
            adaptive         = str(r.get("adaptive", "")).lower() == "yes",
            categories       = categories,
        ))
    logger.info("Loaded %d assessments from catalog.", len(assessments))
    return assessments


class CatalogIndex:
    """FAISS-backed semantic index over the catalog."""

    def __init__(self):
        self.assessments: list[Assessment] = []
        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.IndexFlatIP] = None

    def build(self):
        logger.info("Loading embedding model '%s' …", EMBED_MODEL)
        self._model = SentenceTransformer(EMBED_MODEL)
        self.assessments = build_catalog()

        texts = [a.embed_text() for a in self.assessments]
        logger.info("Embedding %d assessments …", len(texts))
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        vecs = np.array(vecs, dtype="float32")

        dim = vecs.shape[1]
        self._index = faiss.IndexFlatIP(dim)   # inner-product on L2-normalised vecs = cosine
        self._index.add(vecs)
        logger.info("FAISS index built (%d vectors, dim=%d).", len(vecs), dim)

    def search(self, query: str, top_k: int = 10) -> list[Assessment]:
        if self._index is None or self._model is None:
            raise RuntimeError("Index not built yet.")
        qvec = self._model.encode([query], normalize_embeddings=True)
        qvec = np.array(qvec, dtype="float32")
        scores, indices = self._index.search(qvec, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append(self.assessments[idx])
        return results

    def get_by_name(self, name: str) -> Optional[Assessment]:
        name_lower = name.lower()
        for a in self.assessments:
            if a.name.lower() == name_lower:
                return a
        return None

    def all_names(self) -> list[str]:
        return [a.name for a in self.assessments]


# Singleton — built at startup
catalog_index = CatalogIndex()
