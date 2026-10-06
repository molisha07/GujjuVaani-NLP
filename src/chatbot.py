"""The GUJJUVAANI retrieval-augmented QA pipeline.

Flow
----
::

    user query
        -> language identification        (language_utils.detect_language)
        -> query normalisation/expansion (language_utils.expand_query)
        -> multilingual embedding        (intfloat/multilingual-e5-small)
        -> semantic retrieval            (FAISS IndexFlatIP / cosine)
        -> relevance gate                (threshold + margin + domain check)
        -> Gujarati answer from the record, or an honest fallback

Design decisions worth explaining in a viva
------------------------------------------
* **No generative LLM.** The answer is the curated Gujarati text stored in the
  knowledge base. That is what makes the system a *retrieval-augmented QA*
  system rather than a chatbot with a guess, and it is why the answers can
  cite a source.
* **Relevance gate, not nearest-neighbour.** A pure vector store will always
  return *something*. We therefore combine
  (a) an absolute similarity floor,
  (b) a score-margin test (is the winner clearly ahead of the runner-up?), and
  (c) a domain-lexicon boost/penalty,
  and we refuse to answer when the question is out of scope.
* **Cosine similarity is not a probability.** ``scores`` are reported as
  "similarity" everywhere in the UI, never as a confidence percentage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from . import config
from .language_utils import LANGUAGE_NAMES, detect_language, expand_query
from .retrieval import (
    KnowledgeBaseError,
    KnowledgeRetriever,
    SearchHit,
    get_retriever,
)

# ---------------------------------------------------------------------------
# Domain lexicon
# ---------------------------------------------------------------------------
#: Words that strongly indicate a Gujarat-culture / heritage / tourism
#: question.  Used as a *signal*, never as the only decision - a question can
#: be perfectly on-topic without any of these words ("Tell me about Patan").
DOMAIN_GU = {
    "ગુજરાત", "વારસો", "સંસ્કૃતિ", "ઇતિહાસ", "પ્રવાસ", "તહેવાર", "મંદિર",
    "વાવ", "કલાકારી", "વારસાદ", "માટાજી", "ગરબા", "ડાંગિયા", "કુચ્છ", "કચ્છ",
    "સૌરાષ્ટ્ર", "રણ", "પૂર્વ", "જૂનાગઢ", "અમદાવાદ", "સુરત", "રાજકોટ", "પાટણ",
    "ભુજ", "વડોદરા", "જામનગર", "પોરબંદર", "દ્વારકા", "સોમનાથ", "મોઢેરા", "અદાલજ",
    "દ્હોલાવીરા", "ધોલાવીરા", "લોઠલ", "સ્ટેચ્યુ", "પટેલ", "ગાંધી", "સિંહ", "ગીર",
    "યુનેસ્કો", "વિશ્વ ધરોહર", "ખાવાનું", "વાસ્તુકલા", "નગર", "સંગીત", "લોકગીત",
    "સાહિત્ય", "કવિ", "અજવાયી", "પર્વત", "નદી", "ખંભાત", "ઉત્તરાયણ", "દિવાળી",
    "નવરાત્રી", "રન્ન ઉત્સવ", "બનાસકાંઠા", "દાહરાવ", "ખંભા", "નળ સરોવર",
    "સાપુતરા", "ભંગર", "ચાંપણેર", "પાવાગઢ", "અમરેલી", "ભાવનગર", "આણંદ",
    "અંક", "વસાહ", "નગરી", "રસોઈ", "શિલ્પ", "કારીગરી", "વિજ્ઞાન", "તકો",
}

DOMAIN_EN = {
    "gujarat", "gujju", "saurashtra", "kutch", "kachchh", "rajkot", "ahmedabad",
    "surat", "vadodara", "jamnagar", "porbandar", "bhuj", "junagadh", "patan",
    "dwarka", "somnath", "modhera", "adalaj", "dholavira", "lothal", "gir",
    "champaner", "pavagadh", "sidi saiyyed", "sarkhej", "statue of unity",
    "temple", "mandir", "stepwell", "vav", "mosque", "heritage", "monument",
    "fort", "palace", "museum", "unesco", "world heritage", "archaeolog",
    "festival", "navratri", "garba", "dandiya", "uptarayan", "janmashtami",
    "diwali", "holi", "rann", "desert", "lion", "wildlife", "sanctuary",
    "national park", "river", "hill", "mountain", "fortress", "jain", "hindu",
    "muslim", "sultan", "sultanate", "solanki", "chaulukya", "vaghela",
    "gandhi", "patel", "sardar", "narsinh", "mehta", "hemchandra", "meghani",
    "food", "cuisine", "dish", "snack", "sweets", "thali", "spice",
    "craft", "textile", "embroidery", "pottery", "terracotta", "bandhani",
    "patola", "dance", "folk", "culture", "history", "historic", "tourism",
    "tourist", "visit", "travel", "pilgrimage", "yatra", "poet", "literature",
    "saint", "poetry", "bhakti", "proverb", "dialect", "marriage", "wedding",
    "rann utsav", "nal sarovar", "saputara", "girnar", "barda", "sanctuary",
}

#: Topic of *known* out-of-domain queries used by the evaluation script to
#: measure the rejection rate.
OUT_OF_DOMAIN_MARKERS = {
    "python", "java", "javascript", "c++", "c programming", "sql", "machine learning",
    "bitcoin", "cryptocurrency", "stock market", "nba", "ipl", "football",
    "cooking pasta", "new york", "tokyo", "europe", "amazon web services",
    "photosynthesis", "human body", "black hole", "quantum computer",
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ChatResponse:
    """Everything the UI needs for one turn."""

    question: str
    normalized_query: str
    detected_language: str
    language_name: str
    answered: bool
    answer_gu: str
    short_answer_gu: str = ""
    curated_answer_en: str = ""
    curated_answer_hi: str = ""
    title: str = ""
    title_en: str = ""
    category: str = ""
    district: str = ""
    source: str = ""
    source_type: str = ""
    source_url: str = ""
    record_id: str = ""
    similarity: float = 0.0
    margin: float = 0.0
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    fallback_kind: str = ""   # "out_of_domain" | "low_similarity" | "ambiguous" | ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_fallback(self) -> bool:
        return not self.answered


# ---------------------------------------------------------------------------
# The chatbot
# ---------------------------------------------------------------------------
class GujjuVaaniChatbot:
    """Question answering over the curated Gujarati knowledge base."""

    def __init__(
        self,
        retriever: Optional[KnowledgeRetriever] = None,
        relevance_threshold: Optional[float] = None,
        hard_reject_threshold: Optional[float] = None,
        min_margin: Optional[float] = None,
    ) -> None:
        self.retriever = retriever or get_retriever()
        self.relevance_threshold = (
            config.RELEVANCE_THRESHOLD
            if relevance_threshold is None
            else float(relevance_threshold)
        )
        self.hard_reject_threshold = (
            config.HARD_REJECT_THRESHOLD
            if hard_reject_threshold is None
            else float(hard_reject_threshold)
        )
        self.min_margin = config.MIN_SCORE_MARGIN if min_margin is None else float(min_margin)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _domain_score(question: str, hits: List[SearchHit]) -> float:
        """Small lexical nudge from the domain lexicon (-0.15 .. +0.15)."""
        lowered = question.lower()
        domain_hits = sum(1 for term in DOMAIN_GU if term in question)
        domain_hits += sum(1 for term in DOMAIN_EN if term in lowered)
        if any(marker in lowered for marker in OUT_OF_DOMAIN_MARKERS):
            return -0.15
        # Bounded: more domain words can only help a little.
        return min(0.15, 0.03 * domain_hits)

    @staticmethod
    def _looks_out_of_domain(question: str) -> bool:
        lowered = question.lower()
        return any(marker in lowered for marker in OUT_OF_DOMAIN_MARKERS)

    def _fill_from_hit(self, response: ChatResponse, hit: SearchHit) -> None:
        record = hit.record
        response.answered = True
        response.record_id = str(record.get("id", ""))
        response.title = str(record.get("title", ""))
        response.title_en = str(record.get("title_en", "")) or str(record.get("title", ""))
        response.category = str(record.get("category", ""))
        response.district = str(record.get("district", ""))
        response.answer_gu = str(record.get("answer_gu", "")).strip()
        response.short_answer_gu = str(record.get("short_answer_gu", "")).strip()
        response.curated_answer_en = str(record.get("short_answer_en", "")).strip()
        response.curated_answer_hi = str(record.get("short_answer_hi", "")).strip()
        response.source = str(record.get("source", "")).strip() or "GujjuVaani knowledge base"
        response.source_type = str(record.get("source_type", "")).strip()
        response.source_url = str(record.get("source_url", "")).strip()

    @staticmethod
    def _make_fallback(
        question: str,
        language: str,
        reason: str,
        kind: str,
        normalized: str = "",
        hits: Optional[List[SearchHit]] = None,
        similarity: float = 0.0,
        margin: float = 0.0,
    ) -> ChatResponse:
        if language == "en":
            text = config.FALLBACK_EN
        elif language == "hi":
            text = config.FALLBACK_HI
        else:
            text = config.FALLBACK_GU

        response = ChatResponse(
            question=question,
            normalized_query=normalized,
            detected_language=language,
            language_name=LANGUAGE_NAMES.get(language, "Unknown"),
            answered=False,
            answer_gu=text,
            reason=reason,
            fallback_kind=kind,
            similarity=round(similarity, 4),
            margin=round(margin, 4),
            candidates=[h.as_dict() for h in (hits or [])[:3]],
        )
        # Show the best near-miss as "did you mean" evidence, without answering.
        if hits and not response.title:
            response.title = hits[0].title
            response.record_id = hits[0].id
            response.category = hits[0].category
        return response

    # -- main entry point -------------------------------------------------
    def ask(self, question: str, language_hint: Optional[str] = None) -> ChatResponse:
        question = (question or "").strip()
        if len(question) < config.MIN_QUERY_LENGTH:
            return self._make_fallback(
                question,
                language_hint or "gu",
                "The question is too short to search with.",
                "empty",
            )

        # 1) language identification (the UI hint is only a tie-breaker)
        language = detect_language(question, language_hint)

        # 2) normalisation / query expansion
        normalized = expand_query(question, language)

        # 3) + 4) embedding + semantic retrieval
        try:
            hits = self.retriever.search(question, top_k=config.RETRIEVAL_TOP_K, language=language)
        except Exception as exc:
            return self._make_fallback(
                question,
                language,
                f"Retrieval error: {type(exc).__name__}: {exc}",
                "error",
                normalized,
            )

        if not hits:
            return self._make_fallback(
                question, language, "The knowledge base returned no candidates.", "empty", normalized
            )

        # 5) relevance gate
        best = hits[0]
        runner_up = hits[1].score if len(hits) > 1 else 0.0
        raw_margin = best.score - runner_up
        adjusted = best.score + self._domain_score(question, hits)
        effective_margin = adjusted - (runner_up + self._domain_score(question, hits[1:]))

        candidates = [h.as_dict() for h in hits]

        if self._looks_out_of_domain(question):
            return self._make_fallback(
                question,
                language,
                "The question is clearly outside the Gujarat culture / heritage / tourism domain.",
                "out_of_domain",
                normalized,
                hits,
                best.score,
                raw_margin,
            )

        if best.score < self.hard_reject_threshold or adjusted < self.relevance_threshold:
            return self._make_fallback(
                question,
                language,
                (
                    f"Best candidate '{best.title}' scored {best.score:.3f} "
                    f"(threshold {self.relevance_threshold:.2f}) - the knowledge base "
                    "does not cover this."
                ),
                "low_similarity",
                normalized,
                hits,
                best.score,
                raw_margin,
            )

        if best.score < 0.82 and raw_margin < self.min_margin and len(hits) > 1:
            # Genuinely ambiguous: two different records are almost as good.
            return self._make_fallback(
                question,
                language,
                (
                    f"Two records scored almost the same "
                    f"('{best.title}' {best.score:.3f} vs '{hits[1].title}' {runner_up:.3f}), "
                    "so the question is too vague to answer confidently. "
                    "Please add a place name, a festival or a person."
                ),
                "ambiguous",
                normalized,
                hits,
                best.score,
                raw_margin,
            )

        # 6) answer
        response = ChatResponse(
            question=question,
            normalized_query=normalized,
            detected_language=language,
            language_name=LANGUAGE_NAMES.get(language, "Unknown"),
            answered=True,
            answer_gu="",
            similarity=round(best.score, 4),
            margin=round(raw_margin, 4),
            candidates=candidates,
        )
        self._fill_from_hit(response, best)
        response.reason = (
            f"Matched '{best.title}' (similarity {best.score:.3f}, "
            f"margin {raw_margin:.3f} over the runner-up)."
        )
        # Keep the domain-adjusted values visible for transparency.
        response.candidates[0]["adjusted_score"] = round(adjusted, 4)
        response.candidates[0]["domain_margin"] = round(effective_margin, 4)
        return response

    # -- convenience -------------------------------------------------------
    def similar_records(self, title_or_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Other knowledge-base entries about the same topic."""
        target: Optional[Dict[str, Any]] = None
        for record in self.retriever.records:
            if str(record.get("id")) == title_or_id or str(record.get("title")) == title_or_id:
                target = record
                break
        if target is None:
            return []
        hits = self.retriever.search(str(target.get("title", "")), top_k=limit + 1)
        out = []
        for hit in hits:
            if hit.id == str(target.get("id")):
                continue
            out.append(hit.as_dict())
            if len(out) >= limit:
                break
        return out

    def status(self) -> Dict[str, Any]:
        return {
            "index": self.retriever.stats(),
            "thresholds": {
                "relevance_threshold": self.relevance_threshold,
                "hard_reject_threshold": self.hard_reject_threshold,
                "min_score_margin": self.min_margin,
                "top_k": config.RETRIEVAL_TOP_K,
            },
            "fallback_examples": {
                "gu": config.FALLBACK_GU,
                "en": config.FALLBACK_EN,
                "hi": config.FALLBACK_HI,
            },
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------
_CHATBOT: Optional[GujjuVaaniChatbot] = None


def get_chatbot(relevance_threshold: Optional[float] = None) -> GujjuVaaniChatbot:
    """Shared chatbot instance (Streamlit caches this too)."""
    global _CHATBOT
    if _CHATBOT is None:
        _CHATBOT = GujjuVaaniChatbot(relevance_threshold=relevance_threshold)
    return _CHATBOT


def reset_chatbot() -> None:
    global _CHATBOT
    _CHATBOT = None


__all__ = [
    "ChatResponse",
    "GujjuVaaniChatbot",
    "get_chatbot",
    "reset_chatbot",
    "KnowledgeBaseError",
    "DOMAIN_GU",
    "DOMAIN_EN",
    "OUT_OF_DOMAIN_MARKERS",
]