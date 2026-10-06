"""Speech-to-Text using ``faster-whisper`` (CTranslate2 Whisper).

Why faster-whisper
------------------
``faster-whisper`` runs the OpenAI Whisper weights through CTranslate2, which
gives int8 quantisation on CPU.  That means:

* no GPU required,
* the ``small`` model (~480 MB) transcribes on an ordinary laptop,
* multilingual support for gu / hi / mr / en (Whisper's Gujarati is the
  weakest of the four, so the UI always shows the transcript and lets the
  user correct it before asking).

The model is loaded once per process and reused.
"""

from __future__ import annotations

import io
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from . import config


class SpeechToTextUnavailable(RuntimeError):
    """faster-whisper is not installed or the model could not be loaded."""


class TranscriptionError(RuntimeError):
    """The audio could not be decoded or transcribed."""


@dataclass
class Transcription:
    text: str
    language: Optional[str]
    language_probability: float
    duration: float
    model: str


# ---------------------------------------------------------------------------
# Model loading (once)
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_MODEL = None
_MODEL_NAME: Optional[str] = None


def whisper_is_installed() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except Exception:
        return False


def get_whisper_model(model_name: Optional[str] = None):
    """Return the cached Whisper model, loading it on first use."""
    global _MODEL, _MODEL_NAME

    if not whisper_is_installed():
        raise SpeechToTextUnavailable(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        )

    wanted = model_name or config.WHISPER_MODEL
    with _LOCK:
        if _MODEL is not None and _MODEL_NAME == wanted:
            return _MODEL
        from faster_whisper import WhisperModel

        config.ensure_dirs()
        try:
            _MODEL = WhisperModel(
                wanted,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
                download_root=str(config.WHISPER_CACHE_DIR),
            )
            _MODEL_NAME = wanted
        except Exception as exc:
            raise SpeechToTextUnavailable(
                f"Could not load the Whisper model '{wanted}'.\n"
                "The first run downloads the weights into models/whisper/ and needs "
                f"the network. ({type(exc).__name__}: {exc})"
            ) from exc
        return _MODEL


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def decode_audio(audio_bytes: bytes, filename: str = "recording.wav") -> Any:
    """Decode uploaded audio into a mono 16 kHz float32 NumPy array.

    Handles whatever container the browser produced (webm/opus, wav, mp4/m4a, ogg).
    """
    import numpy as np

    # 1. First attempt: faster_whisper's built-in PyAV decoder
    try:
        from faster_whisper.audio import decode_audio as _decode

        arr = _decode(io.BytesIO(audio_bytes))
        if arr is not None and len(arr) > 0:
            return np.ascontiguousarray(arr, dtype=np.float32)
    except Exception:
        pass

    # 2. Second attempt: SoundFile with automatic 16kHz resampling
    try:
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample to 16,000 Hz if needed
        if sr != 16000 and len(data) > 0:
            import scipy.signal

            num_samples = int(round(len(data) * 16000.0 / float(sr)))
            if num_samples > 0:
                data = scipy.signal.resample(data, num_samples).astype(np.float32)

        if len(data) > 0:
            return np.ascontiguousarray(data, dtype=np.float32)
    except Exception:
        pass

    # 3. Third attempt: Direct PyAV container stream reading
    try:
        import av

        container = av.open(io.BytesIO(audio_bytes))
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is not None:
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
            frames = []
            for packet in container.demux(stream):
                for frame in packet.decode():
                    for resampled_frame in resampler.resample(frame):
                        frames.append(resampled_frame.to_ndarray())
            if frames:
                data = np.concatenate(frames, axis=1).squeeze()
                return np.ascontiguousarray(data, dtype=np.float32)
    except Exception:
        pass

    raise TranscriptionError(
        f"Could not decode the recorded audio ({filename}). "
        "Please ensure your microphone is working or upload a valid audio file (.wav, .mp3, .ogg, .m4a, .webm)."
    )


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
def transcribe_bytes(
    audio_bytes: bytes,
    filename: str = "recording.wav",
    language: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Transcription:
    """Transcribe raw audio bytes (e.g. a Streamlit ``UploadedFile``).

    ``language`` accepts GujjuVaani codes (``gu``, ``hi``, ``mr``, ``en``).
    Passing ``None`` lets Whisper auto-detect, which is ideal for Roman
    Gujarati / multilingual voice input.
    """
    if not audio_bytes or len(audio_bytes) < 64:
        raise TranscriptionError("The recording was empty - please click record and speak your question.")

    audio = decode_audio(audio_bytes, filename)
    if audio is None or len(audio) == 0:
        raise TranscriptionError("No audio samples were found in the recording. Please try speaking again.")

    # Check for near-total silence
    import numpy as np
    max_amp = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    if max_amp < 1e-4:
        raise TranscriptionError("The recording appears to be silent. Please check your microphone volume.")

    whisper_lang = config.WHISPER_LANGUAGE_CODES.get(language, None)
    if whisper_lang is None and language not in (None, "und"):
        whisper_lang = config.WHISPER_LANGUAGE_CODES.get(language)

    model = get_whisper_model(model_name)

    # Transcribe with vad_filter=False to avoid external Silero VAD download hanging
    try:
        segments, info = model.transcribe(
            audio,
            language=whisper_lang,
            task="transcribe",
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        parts = []
        for segment in segments:
            text = (segment.text or "").strip()
            if text:
                parts.append(text)
    except Exception as exc:
        raise TranscriptionError(
            f"Transcription failed ({type(exc).__name__}: {exc}). "
            "Please try speaking again or upload a short audio clip."
        ) from exc

    text = " ".join(parts).strip()
    return Transcription(
        text=text,
        language=getattr(info, "language", whisper_lang),
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        duration=float(getattr(info, "duration", 0.0) or 0.0),
        model=model_name or config.WHISPER_MODEL,
    )


def transcribe_uploaded(
    uploaded_file,
    language: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Transcription:
    """Convenience wrapper for a ``st.file_uploader`` / ``st.audio_input`` file."""
    if uploaded_file is None:
        raise TranscriptionError("No audio was received from the microphone.")
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    name = getattr(uploaded_file, "name", "recording.wav") or "recording.wav"
    return transcribe_bytes(data, filename=name, language=language, model_name=model_name)


def stt_status() -> Dict[str, Any]:
    """Diagnostic dict for the UI."""
    return {
        "installed": whisper_is_installed(),
        "model": config.WHISPER_MODEL,
        "device": config.WHISPER_DEVICE,
        "compute_type": config.WHISPER_COMPUTE_TYPE,
        "loaded": _MODEL is not None,
        "cache_dir": str(config.WHISPER_CACHE_DIR),
        "languages": ["gu", "hi", "mr", "en"],
    }


__all__ = [
    "SpeechToTextUnavailable",
    "TranscriptionError",
    "Transcription",
    "transcribe_bytes",
    "transcribe_uploaded",
    "get_whisper_model",
    "decode_audio",
    "whisper_is_installed",
    "stt_status",
]