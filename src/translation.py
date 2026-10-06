"""Machine translation for the *display* of answers (Gujarati -> EN/HI/MR).

Important architectural note
----------------------------
Translation is a **presentation layer** feature only.  The question-answering
pipeline never needs it: retrieval is multilingual because
``intfloat/multilingual-e5-small`` is multilingual, so a Hindi or Marathi
question is matched directly against Gujarati passages.  Translation is used
*after* an answer already exists in Gujarati, to show that answer in another
language.

Backend chain (tried in the order given by ``config.TRANSLATION_BACKENDS``)
---------------------------------------------------------------------------
1. ``indictrans2`` - AI4Bharat IndicTrans2, the research-preferred engine for
   Indic languages.
   * Requires ``pip install IndicTrans2``.
   * The Hugging Face repositories are **gated**, so a *free* Hugging Face
     account is needed once:  ``huggingface-cli login``.

2. ``nllb`` - Meta ``facebook/nllb-200-distilled-600M``.
   * Free, ungated, pure ``transformers``, runs on CPU.
   * One-time download of roughly 2.4 GB, then fully offline.
   * Verified working on this machine for gu->en and gu->hi.
   * Caveat: Gujarati is a low-resource language for this model, so machine
     output is rough.  The UI therefore prefers the *curated* English/Hindi
     short answers stored in the knowledge base when the caller asks for
     them (see ``chatbot.answer_curated``).

3. ``deep-translator`` - clearly isolated LAST-RESORT network fallback.  It is
   **not** part of the production architecture because of request limits and
   because it is an unofficial scraping wrapper.  Enable explicitly with
   ``GUJJUVAANI_TRANSLATOR=deep-translator``.

4. If everything fails we raise :class:`TranslationUnavailable`.  The UI catches
   it and falls back to the curated short answer, so the app never crashes.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import config

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TranslationUnavailable(RuntimeError):
    """No translation backend could be loaded or used."""


# ---------------------------------------------------------------------------
# Sentence chunking (seq2seq MT models have a ~512 token limit)
# ---------------------------------------------------------------------------
_SENTENCE_SPLIT = re.compile(r"(?<=[\u0964\u0965।!?])\s+|(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


def split_sentences(text: str, max_chars: int = config.TRANSLATION_CHUNK_CHARS) -> List[str]:
    """Split into translation-sized chunks, preserving order."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            # Pathologically long sentence: cut on commas, then hard-wrap.
            pieces = re.split(r"(?<=[,;:])", sentence)
        else:
            pieces = [sentence]
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(current) + len(piece) + 1 <= max_chars:
                current = f"{current} {piece}".strip()
            else:
                if current:
                    chunks.append(current)
                if len(piece) > max_chars:
                    for start in range(0, len(piece), max_chars):
                        chunks.append(piece[start : start + max_chars])
                    current = ""
                else:
                    current = piece
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Backend base class
# ---------------------------------------------------------------------------
class TranslationBackend:
    """Interface every backend implements."""

    name = "base"
    display_name = "base"
    requires_download_mb = 0
    note = ""

    def available(self) -> bool:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError

    def translate(self, text: str, source: str, target: str) -> str:
        raise NotImplementedError

    # -- convenience -------------------------------------------------
    def translate_long(self, text: str, source: str, target: str) -> str:
        """Translate chunk-by-chunk and re-join."""
        if source == target:
            return text
        if len(text) <= config.TRANSLATION_CHUNK_CHARS:
            return self.translate(text, source, target)
        parts = [self.translate(chunk, source, target) for chunk in split_sentences(text)]
        joined = " ".join(p.strip() for p in parts if p and p.strip())
        return _WHITESPACE.sub(" ", joined).strip() or text

    def status(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "display_name": self.display_name,
            "available": self.available(),
            "loaded": False,
            "requires_download_mb": self.requires_download_mb,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Backend 1: AI4Bharat IndicTrans2
# ---------------------------------------------------------------------------
class IndicTrans2Backend(TranslationBackend):
    name = "indictrans2"
    display_name = "AI4Bharat IndicTrans2"
    requires_download_mb = 4400
    note = (
        "Needs `pip install IndicTrans2` and a free Hugging Face login "
        "(`huggingface-cli login`) because the model repos are gated."
    )

    def __init__(self) -> None:
        self._transformer = None
        self._tokenizer = None
        self._cache: Dict[str, object] = {}

    def available(self) -> bool:
        try:
            import IndicTrans2  # noqa: F401
            import transformers  # noqa: F401
            return True
        except Exception:
            return False

    def _resolve_model(self, source: str, target: str) -> str:
        key = f"{source}>{target}"
        if key not in config.INDICTRANS2_MODELS:
            raise TranslationUnavailable(
                f"IndicTrans2 has no published checkpoint for {source}->{target}."
            )
        return config.INDICTRANS2_MODELS[key]

    def load(self) -> None:
        if self._transformer is not None:
            return
        try:
            from IndicTrans2 import converter  # type: ignore
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:
            raise TranslationUnavailable(
                "IndicTrans2 is not installed. Run: pip install IndicTrans2"
            ) from exc

        def _get(model_name: str):
            if model_name not in self._cache:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                model.eval()
                converter.checkpoint_size[model_name] = {"vocab_size": 18, "hidden_size": 18}
                self._cache[model_name] = (tokenizer, model)
            return self._cache[model_name]

        self._load_pair = _get
        self._transformer = True  # marker: ready

    def translate(self, text: str, source: str, target: str) -> str:
        self.load()
        if target == "en":
            model_name = self._resolve_model(source, "en")
            tokenizer, model = self._load_pair(model_name)
            tokenizer.src_lang = source
        else:
            model_name = self._resolve_model("en", target)
            tokenizer, model = self._load_pair(model_name)
            tokenizer.src_lang = "en"

        import torch  # local import: only needed when this backend is active

        with torch.inference_mode():
            batch = tokenizer([text], return_tensors="pt", padding=True)
            generated = model.generate(
                **batch,
                num_beams=5,
                max_length=256,
                early_stopping=True,
            )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def status(self) -> Dict[str, object]:
        data = super().status()
        data["loaded"] = self._transformer is not None
        return data


# ---------------------------------------------------------------------------
# Backend 2: Meta NLLB-200 distilled 600M
# ---------------------------------------------------------------------------
class NLLBBackend(TranslationBackend):
    name = "nllb"
    display_name = "Meta NLLB-200 distilled 600M"
    requires_download_mb = config.NLLB_APPROX_DOWNLOAD_MB
    note = (
        "Free and offline after a one-time ~2.4 GB download. Gujarati output "
        "quality is moderate, so the UI prefers the curated short answers."
    )

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            return hasattr(transformers, "AutoModelForSeq2SeqLM") and bool(AutoTokenizer) and bool(AutoModelForSeq2SeqLM)
        except Exception:
            return False

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:
            raise TranslationUnavailable("transformers is not installed.") from exc

        config.ensure_dirs()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(config.NLLB_MODEL)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(config.NLLB_MODEL)
            self._model.eval()
        except Exception as exc:
            self._tokenizer = None
            self._model = None
            raise TranslationUnavailable(
                f"Could not load {config.NLLB_MODEL}. First run downloads "
                f"~{config.NLLB_APPROX_DOWNLOAD_MB} MB, so check the network "
                f"connection. ({type(exc).__name__}: {exc})"
            ) from exc

    def translate(self, text: str, source: str, target: str) -> str:
        self.load()
        if source not in config.NLLB_LANG_CODES or target not in config.NLLB_LANG_CODES:
            raise TranslationUnavailable(
                f"NLLB does not support {source}->{target} in this project."
            )

        import torch

        self._tokenizer.src_lang = config.NLLB_LANG_CODES[source]
        batch = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=480)
        bos = self._tokenizer.convert_tokens_to_ids(config.NLLB_LANG_CODES[target])
        with torch.inference_mode():
            generated = self._model.generate(
                **batch,
                forced_bos_token_id=bos,
                max_new_tokens=320,
                num_beams=2,
                no_repeat_ngram_size=3,
            )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def status(self) -> Dict[str, object]:
        data = super().status()
        data["loaded"] = self._model is not None
        return data


# ---------------------------------------------------------------------------
# Backend 3: deep-translator (isolated last resort, opt-in)
# ---------------------------------------------------------------------------
class DeepTranslatorBackend(TranslationBackend):
    name = "deep-translator"
    display_name = "deep-translator (network fallback)"
    requires_download_mb = 0
    note = (
        "Unofficial online fallback with request limits. Not the production "
        "architecture - enable with GUJJUVAANI_TRANSLATOR=deep-translator."
    )

    _TARGET_CODES = {"en": "en", "hi": "hi", "mr": "mr", "gu": "gu"}

    def __init__(self) -> None:
        self._translator = None

    def available(self) -> bool:
        try:
            import deep_translator  # noqa: F401

            return True
        except Exception:
            return False

    def load(self) -> None:
        if self._translator is not None:
            return
        try:
            from deep_translator import GoogleTranslator
        except Exception as exc:
            raise TranslationUnavailable(
                "deep-translator is not installed (pip install deep-translator)."
            ) from exc
        self._translator = GoogleTranslator

    def translate(self, text: str, source: str, target: str) -> str:
        self.load()
        try:
            result = self._translator(
                source=self._TARGET_CODES.get(source, source),
                target=self._TARGET_CODES.get(target, target),
            ).translate(text)
        except Exception as exc:
            raise TranslationUnavailable(f"deep-translator failed: {exc}") from exc
        if not result:
            raise TranslationUnavailable("deep-translator returned an empty result.")
        return str(result)

    def status(self) -> Dict[str, object]:
        data = super().status()
        data["loaded"] = self._translator is not None
        return data


# ---------------------------------------------------------------------------
# Backend registry + lazy singleton
# ---------------------------------------------------------------------------
_BACKENDS = {
    "indictrans2": IndicTrans2Backend,
    "nllb": NLLBBackend,
    "deep-translator": DeepTranslatorBackend,
}

_LOCK = threading.Lock()
_ACTIVE: Optional[TranslationBackend] = None
_TRIED = False
_LAST_ERROR: Optional[str] = None


def _build_active() -> Optional[TranslationBackend]:
    """Instantiate and ``load()`` the first usable backend."""
    global _ACTIVE, _TRIED, _LAST_ERROR

    for name in config.TRANSLATION_BACKENDS:
        name = name.strip().lower()
        backend_cls = _BACKENDS.get(name)
        if backend_cls is None:
            continue
        backend = backend_cls()
        try:
            if not backend.available():
                continue
            backend.load()
            _ACTIVE = backend
            _LAST_ERROR = None
            return backend
        except Exception as exc:
            _LAST_ERROR = f"{name}: {type(exc).__name__}: {exc}"
    return None


def get_translator(force_reload: bool = False) -> TranslationBackend:
    """Return the active translation backend, raising if none is usable."""
    global _ACTIVE, _TRIED

    with _LOCK:
        if _ACTIVE is not None and not force_reload:
            return _ACTIVE
        backend = _build_active()
        if backend is None:
            raise TranslationUnavailable(
                "No translation backend is available.\n"
                f"Tried: {', '.join(config.TRANSLATION_BACKENDS)}\n"
                f"Last error: {_LAST_ERROR or 'none of them is installed'}\n"
                "Fix (free, offline): install torch + transformers and let the app "
                "download facebook/nllb-200-distilled-600M once (~2.4 GB), or "
                "run `pip install IndicTrans2 && huggingface-cli login`."
            )
        return backend


def translation_status() -> Dict[str, object]:
    """A dict describing every backend - rendered by the UI status panel."""
    rows: List[Dict[str, object]] = []
    active_name = None
    try:
        active_name = get_translator().name
    except Exception:
        active_name = None

    for name in config.TRANSLATION_BACKENDS:
        name = name.strip().lower()
        backend_cls = _BACKENDS.get(name)
        if backend_cls is None:
            rows.append(
                {
                    "backend": name,
                    "display_name": name,
                    "available": False,
                    "active": False,
                    "requires_download_mb": 0,
                    "note": "unknown backend name in config",
                }
            )
            continue
        backend = backend_cls()
        try:
            available = backend.available()
        except Exception:
            available = False
        rows.append(
            {
                "backend": name,
                "display_name": backend.display_name,
                "available": available,
                "active": name == active_name,
                "requires_download_mb": backend.requires_download_mb,
                "note": backend.note,
            }
        )

    return {
        "active": active_name,
        "backends": rows,
        "last_error": _LAST_ERROR,
    }


def translate(
    text: str,
    target: str,
    source: str = "gu",
    raise_on_error: bool = True,
) -> Tuple[str, str]:
    """Translate ``text`` from ``source`` to ``target``.

    Returns ``(translated_text, backend_name)``.

    If ``raise_on_error`` is False the original text is returned together with
    an empty backend name instead of raising, so the UI can degrade silently.
    """
    text = (text or "").strip()
    if not text:
        return "", ""
    if source == target:
        return text, "identity"

    try:
        backend = get_translator()
        return backend.translate_long(text, source, target), backend.name
    except Exception as exc:
        if raise_on_error:
            if isinstance(exc, TranslationUnavailable):
                raise
            raise TranslationUnavailable(f"{type(exc).__name__}: {exc}") from exc
        return text, ""


def reset_backend_cache() -> None:
    """Forget the cached backend (used by tests)."""
    global _ACTIVE, _TRIED, _LAST_ERROR
    with _LOCK:
        _ACTIVE = None
        _TRIED = False
        _LAST_ERROR = None


__all__ = [
    "TranslationUnavailable",
    "TranslationBackend",
    "IndicTrans2Backend",
    "NLLBBackend",
    "DeepTranslatorBackend",
    "translate",
    "translation_status",
    "get_translator",
    "reset_backend_cache",
    "split_sentences",
]