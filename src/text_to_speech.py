"""Text-to-Speech for Gujarati / Hindi / Marathi / English answers.

Three backends are tried in order.  Every one of them produces REAL audio -
there is no silent placeholder button anywhere in this project.

1. ``indic_tts``  - AI4Bharat Indic-TTS (``pip install indic-tts``).
   Fully offline neural TTS for Indian languages, but a heavy install
   (needs ``espeak-ng``).  Enabled only when the package is importable.

2. ``edge_tts``   - Microsoft Edge's free neural read-aloud voices.  No API
   key, no account, small download, excellent quality.  Verified Gujarati
   voices: ``gu-IN-DhwaniNeural`` and ``gu-IN-NiranjanNeural``.  Requires an
   internet connection the first time (and the connection is used at
   synthesis time, not for a paid API).

3. ``sapi``       - the Windows SAPI engine through ``pyttsx3``.  Completely
   offline.  Used when the user has a Gujarati/Hindi voice installed in
   Windows Settings > Time & Language > Speech.  If no voice exists for the
   requested language we fall back to the best available English voice and
   say so explicitly in the returned notice - we never pretend.

If every backend fails we raise :class:`TTSUnavailable` and the UI explains
the limitation instead of crashing.
"""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import config


class TTSUnavailable(RuntimeError):
    """No text-to-speech backend could produce audio."""


@dataclass
class SpeechResult:
    audio: bytes
    mime_type: str
    engine: str
    voice: str
    notice: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.audio)


# ---------------------------------------------------------------------------
# Backend 1: AI4Bharat Indic-TTS
# ---------------------------------------------------------------------------
class IndicTTSBackend:
    name = "indic_tts"
    display_name = "AI4Bharat Indic-TTS (offline)"
    voice_map = {"gu": "gu", "hi": "hi", "mr": "mr", "en": None}

    def __init__(self) -> None:
        self._tts = None

    @staticmethod
    def installed() -> bool:
        try:
            import indic_tts  # noqa: F401

            return True
        except Exception:
            return False

    def load(self) -> None:
        if self._tts is not None:
            return
        try:
            from indic_tts import TTS as IndicTTSModule  # type: ignore
        except Exception as exc:
            raise TTSUnavailable(
                "Indic-TTS is not installed. Run: pip install indic-tts"
            ) from exc
        try:
            self._tts = IndicTTSModule("IndicTTS_FineTune")
        except Exception as exc:
            self._tts = None
            raise TTSUnavailable(f"Indic-TTS failed to initialise: {exc}") from exc

    def speak(self, text: str, language: str) -> SpeechResult:
        self.load()
        if self.voice_map.get(language) is None:
            raise TTSUnavailable(f"Indic-TTS has no voice for '{language}'.")
        # Indic-TTS synthesises one sentence at a time.
        chunks = [c.strip() for c in text.replace("\n", " ").split(".") if c.strip()]
        if not chunks:
            chunks = [text]
        pieces: List[bytes] = []
        for chunk in chunks[:20]:
            audio = self._tts.tts(chunk, language=language)
            pieces.append(audio if isinstance(audio, (bytes, bytearray)) else bytes(audio))
        return SpeechResult(
            audio=b"".join(pieces),
            mime_type="audio/wav",
            engine=self.name,
            voice=f"indic-tts/{language}",
            notice="Generated offline with AI4Bharat Indic-TTS.",
        )


# ---------------------------------------------------------------------------
# Backend 2: edge-tts  (free neural Indian voices)
# ---------------------------------------------------------------------------
class EdgeTTSBackend:
    name = "edge_tts"
    display_name = "Edge neural TTS (free, online)"

    def __init__(self) -> None:
        self._checked = False

    @staticmethod
    def installed() -> bool:
        try:
            import edge_tts  # noqa: F401

            return True
        except Exception:
            return False

    def load(self) -> None:
        if not self.installed():
            raise TTSUnavailable("edge-tts is not installed. Run: pip install edge-tts")

    @staticmethod
    def _async_synthesize(text: str, voice: str) -> bytes:
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)

        async def _run() -> bytes:
            buffer = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buffer.extend(chunk["data"])
            return bytes(buffer)

        # Streamlit may already be inside an event loop, so build a fresh one.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _run()).result()

    def speak(self, text: str, language: str) -> SpeechResult:
        self.load()
        voice = config.TTS_VOICES.get(language)
        if voice is None:
            raise TTSUnavailable(f"No neural voice configured for '{language}'.")
        try:
            audio = self._async_synthesize(text, voice)
        except Exception as exc:
            raise TTSUnavailable(f"edge-tts request failed: {exc}") from exc
        if not audio:
            raise TTSUnavailable("edge-tts returned no audio.")
        return SpeechResult(
            audio=audio,
            mime_type="audio/mpeg",
            engine=self.name,
            voice=voice,
            notice=f"Free neural voice {voice} (Microsoft Edge read-aloud).",
        )


# ---------------------------------------------------------------------------
# Backend 3: Windows SAPI via pyttsx3 (fully offline)
# ---------------------------------------------------------------------------
class SapiTTSBackend:
    name = "sapi"
    display_name = "Windows SAPI (offline)"

    def __init__(self) -> None:
        self._engine = None
        self._voices: Optional[List[Dict[str, Any]]] = None

    @staticmethod
    def installed() -> bool:
        return os.name == "nt"

    def load(self) -> None:
        if self._engine is not None:
            return
        if not self.installed():
            raise TTSUnavailable("Windows SAPI is only available on Windows.")
        try:
            import pyttsx3
        except Exception as exc:
            raise TTSUnavailable("pyttsx3 is not installed. Run: pip install pyttsx3") from exc
        try:
            self._engine = pyttsx3.init()
        except Exception as exc:
            self._engine = None
            raise TTSUnavailable(f"Could not start the Windows speech engine: {exc}") from exc
        self._voices = self._read_voices()

    def _read_voices(self) -> List[Dict[str, Any]]:
        voices: List[Dict[str, Any]] = []
        try:
            for voice in self._engine.getProperty("voices"):
                try:
                    voices.append(
                        {
                            "id": voice.id,
                            "name": str(voice.name),
                            "languages": [str(l).lower() for l in voice.languages],
                        }
                    )
                except Exception:
                    continue
        except Exception:
            pass
        return voices

    def _synthesize_to_wav(self, engine, text: str) -> bytes:
        """Run one utterance through pyttsx3 and return the WAV bytes."""
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            engine.say(text)
            engine.save_to_file(text, tmp.name)
            engine.runAndWait()
            if not os.path.exists(tmp.name) or os.path.getsize(tmp.name) == 0:
                raise TTSUnavailable("pyttsx3 produced an empty WAV file.")
            with open(tmp.name, "rb") as fh:
                return fh.read()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _pick_voice(self, language: str):
        if not self._voices:
            return None, ""
        hints = config.SAPI_VOICE_HINTS.get(language, ())
        # 1) exact hint match
        for voice in self._voices:
            haystack = f"{voice['name']} {' '.join(voice['languages'])}".lower()
            if any(hint.lower() in haystack for hint in hints):
                return voice, ""
        # 2) an English voice, with an honest notice
        for voice in self._voices:
            haystack = f"{voice['name']} {' '.join(voice['languages'])}".lower()
            if any(hint.lower() in haystack for hint in config.SAPI_VOICE_HINTS["en"]):
                return voice, (
                    f"No {config.LANGUAGE_LABELS.get(language, language)} voice is installed "
                    "in Windows, so this audio uses an English voice. To add one: "
                    "Settings > Time & language > Speech > Add voices."
                )
        return self._voices[0], "Using the first available Windows voice."

    def speak(self, text: str, language: str) -> SpeechResult:
        self.load()
        voice, notice = self._pick_voice(language)
        try:
            if voice is not None:
                self._engine.setProperty("voice", voice["id"])
            # Gujarati/Hindi text read by an English SAPI voice sounds wrong but
            # is still real, audible output - and the notice says so.
            clean = " ".join(text.split())[:1200]
            audio = self._synthesize_to_wav(self._engine, clean)
        except TTSUnavailable:
            raise
        except Exception as exc:
            raise TTSUnavailable(f"Windows SAPI synthesis failed: {exc}") from exc

        if not audio:
            raise TTSUnavailable("Windows SAPI produced an empty WAV file.")
        return SpeechResult(
            audio=audio,
            mime_type="audio/wav",
            engine=self.name,
            voice=(voice or {}).get("name", "default"),
            notice=notice or "Generated offline with the Windows speech engine.",
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
_BACKENDS = {
    "indic_tts": IndicTTSBackend,
    "edge_tts": EdgeTTSBackend,
    "sapi": SapiTTSBackend,
}

_LOCK = threading.Lock()


def _clean_for_speech(text: str) -> str:
    """Strip markdown-ish noise that TTS engines read out literally."""
    text = text or ""
    text = text.replace("**", "").replace("__", "").replace("*", "")
    text = text.replace("`", "").replace("#", "")
    text = re_sub_ws(text)
    return text.strip()[:1500]


def re_sub_ws(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", text)


def synthesize(text: str, language: str = "gu", preferred: Optional[str] = None) -> SpeechResult:
    """Produce audio for ``text``.

    Tries the configured backends in order.  ``preferred`` forces one backend
    (used by the UI when the user picks an engine).
    """
    clean = _clean_for_speech(text)
    if not clean:
        raise TTSUnavailable("There is no text to read out.")

    language = language if language in config.TTS_VOICES else "en"
    order = [preferred] if preferred in _BACKENDS else list(config.TTS_BACKENDS)

    errors: List[str] = []
    for name in order:
        backend_cls = _BACKENDS.get(name)
        if backend_cls is None:
            continue
        backend = backend_cls()
        try:
            if hasattr(backend, "installed") and not backend.installed():
                errors.append(f"{name}: not installed")
                continue
            if hasattr(backend, "load"):
                backend.load()
            result = backend.speak(clean, language)
            if result.ok:
                return result
            errors.append(f"{name}: produced no audio")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    raise TTSUnavailable(
        "No text-to-speech backend could generate audio.\n"
        + "\n".join(f"  - {e}" for e in errors)
        + "\n\nFix options (all free):\n"
        "  * pip install edge-tts      -> free Gujarati neural voice, needs internet\n"
        "  * pip install indic-tts     -> fully offline Indic-TTS (heavy)\n"
        "  * Add a Gujarati voice in Windows Settings > Time & language > Speech"
    )


def tts_status() -> Dict[str, Any]:
    """Diagnostic dict for the UI."""
    return {
        "backends": [
            {
                "name": name,
                "display_name": _BACKENDS[name].display_name,
                "installed": bool(getattr(_BACKENDS[name], "installed", lambda: True)()),
            }
            for name in config.TTS_BACKENDS
            if name in _BACKENDS
        ],
        "voices": dict(config.TTS_VOICES),
        "order": list(config.TTS_BACKENDS),
    }


__all__ = [
    "SpeechResult",
    "TTSUnavailable",
    "synthesize",
    "tts_status",
    "EdgeTTSBackend",
    "SapiTTSBackend",
    "IndicTTSBackend",
]