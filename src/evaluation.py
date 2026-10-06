"""Offline evaluation of the GUJJUVAANI retrieval pipeline.

Run it with::

    python -m src.evaluation              # from the project root
    python src/evaluation.py              # also works

Metrics
-------
* **Top-1 accuracy** - the correct record was ranked first.
* **Top-3 accuracy** - the correct record appeared in the first three.
* **Average similarity** - mean cosine similarity of the top hit (this is a
  GEOMETRIC similarity score, *not* a probability and *not* an accuracy).
* **Fallback accuracy** - for out-of-domain questions the system correctly
  refuses to answer.
* **Per-language accuracy** - the same numbers broken down by input language.
* **Threshold sweep** - top-1 / top-3 / rejection rate for a range of
  thresholds, so the configured threshold is justified by data instead of
  guesswork.

A question counts as *correct* when the retrieved record's ``category``
(or its title) matches the expected category/topic from
``data/evaluation_questions.json``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import config
from .chatbot import GujjuVaaniChatbot
from .retrieval import KnowledgeBaseError, get_retriever

# Windows consoles default to cp1252 and would crash on Gujarati output.
for _stream in (sys.stdout, sys.stderr):
    try:  # pragma: no cover
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Out-of-domain probe set (used for the rejection metric)
# ---------------------------------------------------------------------------
OUT_OF_DOMAIN_PROBES: List[Dict[str, str]] = [
    {"question": "What is Python programming language?", "language": "en"},
    {"question": "Explain machine learning in simple words.", "language": "en"},
    {"question": "How do I install pandas on Windows?", "language": "en"},
    {"question": "Who is the current president of the United States?", "language": "en"},
    {"question": "How do I cook pasta at home?", "language": "en"},
    {"question": "What is the price of Bitcoin today?", "language": "en"},
    {"question": "Python programming kya hai?", "language": "roman_gu"},
    {"question": "Tell me about theIPL cricket tournament.", "language": "en"},
    {"question": "Write a Java program for printing hello world.", "language": "en"},
    {"question": "Tell me about New York City tourism.", "language": "en"},
    {"question": "Python kya che hai?", "language": "roman_gu"},
    {"question": "What is photosynthesis?", "language": "en"},
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class EvalQuestion:
    question: str
    expected_topic: str
    expected_category: str
    language: str

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], index: int) -> "EvalQuestion":
        question = str(raw.get("question", "")).strip()
        if not question:
            raise ValueError(f"evaluation question #{index} has no 'question' field")
        return cls(
            question=question,
            expected_topic=str(raw.get("expected_topic", "")).strip(),
            expected_category=str(raw.get("expected_category", "")).strip(),
            language=str(raw.get("language", "en")).strip(),
        )


def load_evaluation_questions(
    path: Optional[Path] = None,
) -> List[EvalQuestion]:
    path = path or config.EVALUATION_QUESTIONS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {path}\n"
            "Create data/evaluation_questions.json (see README)."
        )
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        raw = raw.get("questions", [])
    return [EvalQuestion.from_dict(item, i) for i, item in enumerate(raw)]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _matches(hit, item: EvalQuestion) -> bool:
    """A hit matches when its category or title covers the expected topic."""
    category = str(hit.record.get("category", "")).strip().lower()
    title = str(hit.record.get("title", "")).strip().lower()
    title_en = str(hit.record.get("title_en", "")).strip().lower()
    exp_cat = item.expected_category.strip().lower()
    exp_topic = item.expected_topic.strip().lower()

    if exp_cat and exp_cat == category:
        return True
    if exp_topic:
        if exp_topic in title or exp_topic in title_en:
            return True
        # tolerate a topic written with different spacing / case
        squashed = " ".join(title.split())
        if squashed == " ".join(exp_topic.split()):
            return True
    return False


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class EvalRow:
    question: str
    language: str
    expected: str
    top1_title: str
    top1_category: str
    similarity: float
    top1_hit: bool
    top3_hit: bool
    answered: bool
    fallback_kind: str
    detected_language: str = ""


@dataclass
class EvalReport:
    rows: List[EvalRow] = field(default_factory=list)
    in_domain: List[EvalRow] = field(default_factory=list)
    out_of_domain_total: int = 0
    out_of_domain_rejected: int = 0
    out_of_domain_rows: List[Dict[str, Any]] = field(default_factory=list)
    sweep: List[Dict[str, Any]] = field(default_factory=list)

    # -- aggregate metrics ------------------------------------------------
    @property
    def total(self) -> int:
        return len(self.in_domain)

    def _acc(self, rows: Iterable[EvalRow], attr: str) -> float:
        rows = list(rows)
        if not rows:
            return 0.0
        return 100.0 * sum(1 for r in rows if getattr(r, attr)) / len(rows)

    @property
    def top1(self) -> float:
        return self._acc(self.in_domain, "top1_hit")

    @property
    def top3(self) -> float:
        return self._acc(self.in_domain, "top3_hit")

    @property
    def answered_rate(self) -> float:
        return self._acc(self.in_domain, "answered")

    @property
    def avg_similarity(self) -> float:
        if not self.in_domain:
            return 0.0
        return sum(r.similarity for r in self.in_domain) / len(self.in_domain)

    @property
    def ood_rejection(self) -> float:
        if not self.out_of_domain_total:
            return 0.0
        return 100.0 * self.out_of_domain_rejected / self.out_of_domain_total

    def by_language(self) -> Dict[str, Dict[str, float]]:
        buckets: Dict[str, List[EvalRow]] = {}
        for row in self.in_domain:
            buckets.setdefault(row.language or "und", []).append(row)
        out: Dict[str, Dict[str, float]] = {}
        for lang, rows in sorted(buckets.items()):
            out[lang] = {
                "n": len(rows),
                "top1": self._acc(rows, "top1_hit"),
                "top3": self._acc(rows, "top3_hit"),
                "avg_similarity": sum(r.similarity for r in rows) / len(rows),
            }
        return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    questions: Optional[Sequence[EvalQuestion]] = None,
    chatbot: Optional[GujjuVaaniChatbot] = None,
    verbose: bool = False,
) -> EvalReport:
    """Run the benchmark and return a report (no printing)."""
    questions = list(questions or load_evaluation_questions())
    chatbot = chatbot or GujjuVaaniChatbot()
    records_by_id = build_records_by_id(chatbot)

    report = EvalReport()

    for item in questions:
        response = chatbot.ask(item.question, language_hint=item.language)
        hits = response.candidates
        row = EvalRow(
            question=item.question,
            language=item.language,
            expected=item.expected_topic or item.expected_category,
            top1_title=response.title,
            top1_category=response.category,
            similarity=response.similarity,
            top1_hit=bool(hits and _matches(_HitShim(hits[0], records_by_id), item)),
            top3_hit=bool(
                hits
                and any(
                    _matches(_HitShim(candidate, records_by_id), item)
                    for candidate in hits[:3]
                )
            ),
            answered=response.answered,
            fallback_kind=response.fallback_kind,
            detected_language=response.detected_language,
        )
        report.rows.append(row)
        report.in_domain.append(row)
        if verbose:
            mark = "OK " if row.top1_hit else ("TOP3" if row.top3_hit else "MISS")
            print(f"[{mark}] ({row.language}) {row.question}")
            print(f"       -> {row.top1_title}  sim={row.similarity:.3f}")

    # out-of-domain rejection
    for probe in OUT_OF_DOMAIN_PROBES:
        response = chatbot.ask(probe["question"], language_hint=probe.get("language"))
        report.out_of_domain_total += 1
        rejected = not response.answered
        if rejected:
            report.out_of_domain_rejected += 1
        report.out_of_domain_rows.append(
            {
                "question": probe["question"],
                "answered": response.answered,
                "similarity": response.similarity,
                "best_match": response.title,
                "fallback_kind": response.fallback_kind,
            }
        )

    report.sweep = threshold_sweep(chatbot, questions, records_by_id)
    return report


class _HitShim:
    """Adapt a plain candidate dict back to the ``.record`` shape for matching."""

    __slots__ = ("record", "score", "rank")

    def __init__(self, candidate: Dict[str, Any], records_by_id: Dict[str, Dict[str, Any]]) -> None:
        record_id = str(candidate.get("id", ""))
        full = records_by_id.get(record_id, {})
        self.record = {
            "id": record_id,
            "title": candidate.get("title", full.get("title", "")),
            "category": candidate.get("category", full.get("category", "")),
            "title_en": full.get("title_en", ""),
        }
        self.score = float(candidate.get("score", 0.0))
        self.rank = int(candidate.get("rank", 0))


def build_records_by_id(chatbot: GujjuVaaniChatbot) -> Dict[str, Dict[str, Any]]:
    return {str(record.get("id")): record for record in chatbot.retriever.records}


def threshold_sweep(
    chatbot: GujjuVaaniChatbot,
    questions: Sequence[EvalQuestion],
    records_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    thresholds: Sequence[float] = (0.60, 0.65, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.85),
) -> List[Dict[str, Any]]:
    """Measure top-1 / top-3 / rejection across candidate thresholds.

    This is what justifies the value configured in ``config.RELEVANCE_THRESHOLD``.
    """
    if records_by_id is None:
        records_by_id = build_records_by_id(chatbot)
    probes = list(questions) + [
        EvalQuestion(p["question"], "", "", p.get("language", "en")) for p in OUT_OF_DOMAIN_PROBES
    ]
    ood_start = len(questions)

    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        top1 = top3 = answered = 0
        n_in = len(questions)
        rejected = 0
        n_out = 0
        for index, item in enumerate(probes):
            saved = chatbot.relevance_threshold
            saved_hard = chatbot.hard_reject_threshold
            chatbot.relevance_threshold = threshold
            chatbot.hard_reject_threshold = min(saved_hard, threshold)
            try:
                response = chatbot.ask(item.question, language_hint=item.language)
            finally:
                chatbot.relevance_threshold = saved
                chatbot.hard_reject_threshold = saved_hard
            if index >= ood_start:
                n_out += 1
                if not response.answered:
                    rejected += 1
                continue
            if not response.answered:
                continue
            answered += 1
            hit_shim = _HitShim(response.candidates[0], records_by_id) if response.candidates else None
            if hit_shim is not None and _matches(hit_shim, item):
                top1 += 1
                if any(
                    _matches(_HitShim(c, records_by_id), item) for c in response.candidates[:3]
                ):
                    top3 += 1
        rows.append(
            {
                "threshold": threshold,
                "top1": 100.0 * top1 / n_in if n_in else 0.0,
                "top3": 100.0 * top3 / n_in if n_in else 0.0,
                "answered": 100.0 * answered / n_in if n_in else 0.0,
                "ood_rejection": 100.0 * rejected / n_out if n_out else 0.0,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------
BAR = "=" * 62


def format_report(report: EvalReport, show_sweep: bool = True) -> str:
    lines: List[str] = []
    add = lines.append

    add(BAR)
    add("GUJJUVAANI NLP EVALUATION")
    add(BAR)
    add("")
    add(f"Total questions (in-domain): {report.total}")

    add("")
    add("-- Retrieval quality (in-domain questions) " + "-" * 22)
    add(f"Top-1 Retrieval Accuracy   : {report.top1:6.2f} %")
    add(f"Top-3 Retrieval Accuracy   : {report.top3:6.2f} %")
    add(f"Answered (not refused)     : {report.answered_rate:6.2f} %")
    add(f"Average similarity score   : {report.avg_similarity:6.4f}   "
        f"(cosine similarity, NOT a probability)")
    add(f"Active threshold           : {config.RELEVANCE_THRESHOLD:.2f}")
    add(f"Hard-reject threshold      : {config.HARD_REJECT_THRESHOLD:.2f}")
    add(f"Minimum score margin       : {config.MIN_SCORE_MARGIN:.3f}")

    add("")
    add("-- Per-language accuracy " + "-" * 43)
    add(f"{'language':<20}{'n':>4}{'top-1':>10}{'top-3':>10}{'avg sim':>10}")
    for lang, stats in report.by_language().items():
        add(f"{lang:<20}{int(stats['n']):>4}{stats['top1']:>9.1f}%{stats['top3']:>9.1f}%"
            f"{stats['avg_similarity']:>10.4f}")

    add("")
    add("-- Out-of-domain rejection " + "-" * 37)
    add(f"Probes                    : {report.out_of_domain_total}")
    add(f"Correctly refused         : {report.out_of_domain_rejected}")
    add(f"Out-of-domain rejection   : {report.ood_rejection:6.2f} %")

    add("")
    add("-- Misses " + "-" * 51)
    misses = [r for r in report.in_domain if not r.top3_hit]
    if not misses:
        add("(none)")
    for row in misses:
        add(f"  [{row.language}] {row.question}")
        add(f"      expected: {row.expected}")
        add(f"      got     : {row.top1_title}  (sim={row.similarity:.3f}, "
            f"{row.fallback_kind or 'answered'})")

    if show_sweep and report.sweep:
        add("")
        add("-- Threshold sweep (justifies the configured value) " + "-" * 17)
        add(f"{'threshold':>10}{'top-1':>9}{'top-3':>9}{'answered':>10}{'ood reject':>12}")
        for row in report.sweep:
            marker = "  <-- configured" if abs(row["threshold"] - config.RELEVANCE_THRESHOLD) < 1e-9 else ""
            add(f"{row['threshold']:>10.2f}{row['top1']:>8.1f}%{row['top3']:>8.1f}%"
                f"{row['answered']:>9.1f}%{row['ood_rejection']:>11.1f}%{marker}")

    add("")
    add("Note: 'similarity' is a cosine value in [-1, 1]. It is a geometric")
    add("      measure of embedding closeness, NOT a probability that the")
    add("      answer is correct.")
    add(BAR)
    return "\n".join(lines)


def print_report(report: EvalReport, show_sweep: bool = True) -> None:
    print(format_report(report, show_sweep=show_sweep))


def print_detail_table(report: EvalReport) -> None:
    """Per-question table, handy for the project report appendix."""
    try:
        import pandas as pd
    except Exception:
        return
    df = pd.DataFrame([row.__dict__ for row in report.rows])
    cols = ["language", "detected_language", "question", "expected",
            "top1_title", "similarity", "top1_hit", "top3_hit", "answered"]
    print()
    print(df[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    verbose = "-v" in argv or "--verbose" in argv
    with_table = "--table" in argv
    show_sweep = "--no-sweep" not in argv

    print(BAR)
    print("GUJJUVAANI - loading knowledge base and embedding model...")
    print("(first run downloads intfloat/multilingual-e5-small, ~470 MB)")
    print(BAR)

    try:
        questions = load_evaluation_questions()
    except Exception as exc:
        print(f"FATAL: {exc}")
        return 2

    try:
        chatbot = GujjuVaaniChatbot()
    except KnowledgeBaseError as exc:
        print(f"FATAL: {exc}")
        return 2
    except Exception as exc:
        print(f"FATAL: could not build the retriever -> {type(exc).__name__}: {exc}")
        return 2

    stats = chatbot.retriever.stats()
    print(f"Records : {stats['records']}")
    print(f"Model   : {stats['embedding_model']} ({stats['embedding_dim']}-d)")
    print(f"Index   : {stats['index_backend']}")
    print(f"KB file : {stats['kb_path']}")
    print()

    report = evaluate(questions, chatbot=chatbot, verbose=verbose)
    print_report(report, show_sweep=show_sweep)
    if with_table:
        print_detail_table(report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EvalQuestion",
    "EvalRow",
    "EvalReport",
    "evaluate",
    "format_report",
    "print_report",
    "load_evaluation_questions",
    "threshold_sweep",
    "OUT_OF_DOMAIN_PROBES",
    "main",
]