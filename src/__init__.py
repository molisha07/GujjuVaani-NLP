"""GUJJUVAANI - Multilingual Gujarati Culture, Heritage, History & Tourism assistant.

Package layout
--------------
``config``            every tunable value lives here
``language_utils``    script-based language identification + text normalisation
``retrieval``         multilingual embeddings + FAISS/cosine semantic search
``chatbot``           the retrieval-augmented QA pipeline + relevance gate
``translation``       machine translation (IndicTrans2 -> NLLB -> M2M100 -> web)
``speech_to_text``    faster-whisper speech recognition
``text_to_speech``    Indic-TTS -> edge-tts -> Windows SAPI
``evaluation``        offline retrieval benchmark
"""

__all__ = [
    "config",
    "language_utils",
    "retrieval",
    "chatbot",
    "translation",
    "speech_to_text",
    "text_to_speech",
    "evaluation",
]

__version__ = "1.0.0"
__all_names__ = "GujjuVaani - Explore Gujarat. In Gujarati."