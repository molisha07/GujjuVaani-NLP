"""Multilingual semantic retrieval over the GUJJUVAANI knowledge base.

Pipeline
--------
1. Load ``data/knowledge_base.json``.
2. For every record build ONE *indexed passage* that combines the title, all
   five question variants (Gujarati / Hindi / Marathi / English / Roman
   Gujarati), the keywords and the short Gujarati answer.  Embedding the
   questions in every language is what makes a single multilingual encoder
   able to serve all five input languages from one index.
3. Embed the passages with ``intfloat/multilingual-e5-small`` using the E5
   ``"passage: "`` prefix and L2 normalisation.
4. Store the matrix in a FAISS ``IndexFlatIP`` index.  Because the vectors are
   unit-norm, inner product == cosine similarity.  If FAISS is unavailable we
   compute the identical thing with NumPy.
5. Embed an incoming question with the ``"query: "`` prefix and return the
   top-k records.

The model is a *pretrained* multilingual sentence-transformer.  We do not
train or fine-tune anything.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import config
from .language_utils import expand_query

# ---------------------------------------------------------------------------
# Optional FAISS
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on the local install
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SearchHit:
    """One retrieved knowledge record with its similarity score."""

    record: Dict[str, Any]
    score: float
    rank: int = 0

    @property
    def id(self) -> str:
        return str(self.record.get("id", ""))

    @property
    def title(self) -> str:
        return str(self.record.get("title", ""))

    @property
    def category(self) -> str:
        return str(self.record.get("category", ""))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "score": round(float(self.score), 4),
            "rank": self.rank,
        }


class KnowledgeBaseError(RuntimeError):
    """Raised when the knowledge base cannot be loaded or is unusable."""


# ---------------------------------------------------------------------------
# Knowledge base loading
# ---------------------------------------------------------------------------
def resolve_knowledge_base_path(explicit: Optional[Path] = None) -> Path:
    """Find ``knowledge_base.json``; raises :class:`KnowledgeBaseError`."""
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    candidates.extend(config.KNOWLEDGE_BASE_CANDIDATES)

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    # Fall back to the per-topic source files so the app works even if the
    # merged knowledge_base.json has not been generated yet.
    if config.KNOWLEDGE_TOPICS_DIR.is_dir() and any(
        config.KNOWLEDGE_TOPICS_DIR.glob("*.json")
    ):
        return config.KNOWLEDGE_TOPICS_DIR

    searched = "\n".join(f"  - {p}" for p in candidates)
    raise KnowledgeBaseError(
        "knowledge_base.json શોધવામાં નિષ્ફળતા / Could not find knowledge_base.json.\n"
        f"Searched:\n{searched}\n"
        "Fix: create data/knowledge_base.json (see README, section 'Dataset')."
    )


def _read_json_records(path: Path) -> List[Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("records") or data.get("knowledge_base") or []
    if not isinstance(data, list):
        raise KnowledgeBaseError(f"{path.name} must contain a JSON list of records.")
    return data


def load_knowledge_base(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load and validate the knowledge base (merged file or per-topic parts)."""
    kb_path = resolve_knowledge_base_path(path)
    try:
        if kb_path.is_dir():
            data: List[Any] = []
            for part in sorted(kb_path.glob("*.json")):
                data.extend(_read_json_records(part))
        else:
            data = _read_json_records(kb_path)
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError(
            f"{kb_path.name} is not valid JSON (line {exc.lineno}, col {exc.colno}): {exc.msg}"
        ) from exc
    except OSError as exc:
        raise KnowledgeBaseError(f"Could not read {kb_path}: {exc}") from exc

    if not isinstance(data, list) or not data:
        raise KnowledgeBaseError(f"{kb_path} contains no records.")

    required = ("id", "title", "answer_gu")
    cleaned: List[Dict[str, Any]] = []
    seen_ids = set()
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            continue
        missing = [key for key in required if not record.get(key)]
        if missing:
            # A record without an answer can never be useful -> skip it loudly
            # via the count, but do not crash the whole app.
            continue
        rid = str(record["id"])
        if rid in seen_ids:
            rid = f"{rid}_{index}"
            record["id"] = rid
        seen_ids.add(rid)
        record.setdefault("category", "General")
        record.setdefault("keywords", [])
        cleaned.append(record)

    if not cleaned:
        raise KnowledgeBaseError(
            "No usable records in the knowledge base. Every record needs at "
            "least 'id', 'title' and 'answer_gu'."
        )
    return cleaned


# ---------------------------------------------------------------------------
# Passage construction
# ---------------------------------------------------------------------------
#: Question fields, in the order they are concatenated into the indexed text.
QUESTION_FIELDS = (
    "question_gu",
    "question_hi",
    "question_mr",
    "question_en",
    "question_roman_gu",
    "extra_questions",
)


def build_passage(record: Dict[str, Any]) -> str:
    """Compose the single text that is embedded for one knowledge record.

    We deliberately include the *questions* in all five languages.  A
    multilingual sentence encoder maps "Rani ki Vav vishe janavo",
    "रानी की वाव के बारे में बताओ" and "રાણીની વાવ વિશે જણાવો" into nearby
    regions of the same vector space, so indexing all of them is what allows
    one FAISS index to serve every input language.
    """
    parts: List[str] = []

    title = str(record.get("title", "")).strip()
    title_en = str(record.get("title_en", "")).strip()
    category = str(record.get("category", "")).strip()
    district = str(record.get("district", "")).strip()

    if title:
        parts.append(title)
    if title_en and title_en != title:
        parts.append(title_en)
    if category:
        parts.append(category)
    if district:
        parts.append(district)

    for field_name in QUESTION_FIELDS:
        value = record.get(field_name)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            parts.extend(str(v).strip() for v in value if str(v).strip())
        else:
            parts.append(str(value).strip())

    keywords = record.get("keywords") or []
    if isinstance(keywords, (list, tuple)):
        parts.append(" ".join(str(k).strip() for k in keywords if str(k).strip()))

    short_answer = str(record.get("short_answer_gu", "")).strip()
    if short_answer:
        parts.append(short_answer)

    passage = " . ".join(p for p in parts if p)
    return f"{config.PASSAGE_PREFIX}{passage}"


# ---------------------------------------------------------------------------
# The retriever
# ---------------------------------------------------------------------------
class KnowledgeRetriever:
    """Loads the model + index ONCE and answers many queries."""

    def __init__(
        self,
        kb_path: Optional[Path] = None,
        model_name: str = config.EMBEDDING_MODEL,
        use_faiss: Optional[bool] = None,
    ) -> None:
        self.kb_path = resolve_knowledge_base_path(kb_path)
        self.records = load_knowledge_base(self.kb_path)
        self.model_name = model_name

        self._use_faiss = _FAISS_AVAILABLE if use_faiss is None else bool(use_faiss)
        if self._use_faiss and not _FAISS_AVAILABLE:
            self._use_faiss = False

        self.model = self._load_model(model_name)
        self.passages = [build_passage(r) for r in self.records]
        self.embeddings = self._embed_passages()
        self.index = self._build_index()

    # -- model -----------------------------------------------------------
    @staticmethod
    def _load_model(model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover
            raise KnowledgeBaseError(
                "sentence-transformers is not installed. Run: pip install -r requirements.txt"
            ) from exc

        try:
            return SentenceTransformer(model_name, device="cpu")
        except Exception as exc:
            raise KnowledgeBaseError(
                f"Could not load the embedding model '{model_name}'.\n"
                "This is usually a missing/corrupt Hugging Face cache or no network "
                "on first run.\n"
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

    # -- embeddings ------------------------------------------------------
    def _embed_passages(self) -> np.ndarray:
        try:
            vectors = self.model.encode(
                self.passages,
                batch_size=config.EMBED_BATCH_SIZE,
                normalize_embeddings=True,   # unit norm -> IP == cosine
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise KnowledgeBaseError(
                f"Embedding the knowledge base failed: {type(exc).__name__}: {exc}"
            ) from exc
        return np.ascontiguousarray(vectors, dtype="float32")

    def embed_query(self, query: str, language: Optional[str] = None) -> np.ndarray:
        """Embed a user question with the E5 ``query:`` prefix."""
        expanded = expand_query(query, language) or query
        if not expanded.strip():
            raise ValueError("empty query")
        vector = self.model.encode(
            f"{config.QUERY_PREFIX}{expanded}",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vector, dtype="float32").reshape(1, -1)

    # -- index -----------------------------------------------------------
    def _build_index(self):
        if not self._use_faiss:
            return None
        dim = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)          # exact search, no approximation
        index.add(self.embeddings)
        return index

    # -- search ----------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        language: Optional[str] = None,
    ) -> List[SearchHit]:
        """Return the ``top_k`` most relevant records using hybrid dense + lexical scoring."""
        import re

        top_k = top_k or config.RETRIEVAL_TOP_K
        query_vector = self.embed_query(query, language)

        # 1. Dense Cosine Similarity over all records
        sims = (self.embeddings @ query_vector.T)[:, 0]

        # 2. Extract content tokens for lexical entity alignment
        _STOPWORDS = {
            "is", "the", "a", "an", "in", "on", "of", "to", "for", "with", "about", "and", "or",
            "what", "who", "where", "when", "how", "why", "tell", "me", "give", "information",
            "kya", "hai", "hain", "kaise", "kab", "kahan", "batao", "baare", "mein", "ki", "ke", "ko",
            "shu", "chhe", "che", "su", "vishe", "janavo", "aapo", "kone", "kyare", "nu", "ni", "na", "no",
            "kaay", "aahe", "saanga", "baddal", "kashi", "koni", "ma", "chho", "chhi", "ane", "ne"
        }
        q_clean = query.lower()
        tokens = [t for t in re.findall(r"[\w]+", q_clean, flags=re.UNICODE) if t not in _STOPWORDS and len(t) >= 2]

        scored_records = []
        for idx, (rec, base_sim) in enumerate(zip(self.records, sims)):
            boost = 0.0
            if tokens:
                title_gu = str(rec.get("title", "")).lower()
                title_en = str(rec.get("title_en", "")).lower()
                kw_str = " ".join(str(k) for k in rec.get("keywords", [])).lower()
                
                for t in tokens:
                    # Direct exact title match gets strongest prior
                    if t in title_en or t in title_gu:
                        boost += 0.06
                    elif t in kw_str:
                        boost += 0.04

            final_score = min(1.0, float(base_sim) + min(0.12, boost))
            scored_records.append((idx, final_score, float(base_sim)))

        # Sort by final hybrid score
        scored_records.sort(key=lambda x: x[1], reverse=True)

        hits: List[SearchHit] = []
        for rank, (idx, final_score, base_sim) in enumerate(scored_records[:top_k], start=1):
            hits.append(
                SearchHit(
                    record=self.records[idx],
                    score=round(final_score, 4),
                    rank=rank,
                )
            )
        return hits

    # -- introspection ---------------------------------------------------
    @property
    def backend(self) -> str:
        return "FAISS IndexFlatIP" if self.index is not None else "NumPy cosine similarity"

    def stats(self) -> Dict[str, Any]:
        return {
            "records": len(self.records),
            "embedding_model": self.model_name,
            "embedding_dim": int(self.embeddings.shape[1]),
            "index_backend": self.backend,
            "passage_format": config.PASSAGE_PREFIX.strip(),
            "query_format": config.QUERY_PREFIX.strip(),
            "kb_path": str(self.kb_path),
        }

    def categories(self) -> List[str]:
        return sorted({str(r.get("category", "General")) for r in self.records})


# ---------------------------------------------------------------------------
# Process-wide singleton (used by both the Streamlit app and the CLI eval)
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_RETRIEVER: Optional[KnowledgeRetriever] = None
_ERROR: Optional[str] = None


def get_retriever(
    kb_path: Optional[Path] = None,
    model_name: str = config.EMBEDDING_MODEL,
    force_reload: bool = False,
) -> KnowledgeRetriever:
    """Return the shared retriever, building it on first use.

    Streamlit wraps this in ``@st.cache_resource``; the module-level cache
    keeps the command-line evaluation script cheap as well.
    """
    global _RETRIEVER, _ERROR

    with _LOCK:
        if _RETRIEVER is not None and not force_reload:
            return _RETRIEVER
        try:
            _RETRIEVER = KnowledgeRetriever(kb_path=kb_path, model_name=model_name)
            _ERROR = None
        except Exception as exc:  # keep the failure, do not crash on import
            _ERROR = f"{type(exc).__name__}: {exc}"
            raise
        return _RETRIEVER


def last_load_error() -> Optional[str]:
    """Message describing why the retriever could not be built (or ``None``)."""
    return _ERROR


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors, for the evaluation report."""
    va, vb = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


__all__ = [
    "KnowledgeRetriever",
    "SearchHit",
    "KnowledgeBaseError",
    "get_retriever",
    "load_knowledge_base",
    "build_passage",
    "resolve_knowledge_base_path",
    "last_load_error",
    "cosine",
]