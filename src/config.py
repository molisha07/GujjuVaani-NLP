"""Central configuration for GUJJUVAANI.

Every tunable number of the project lives here so that thresholds, model ids
and file locations can be changed in exactly ONE place.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

#: Candidate locations for the knowledge base, in priority order.
KNOWLEDGE_BASE_CANDIDATES = [
    DATA_DIR / "knowledge_base.json",
    PROJECT_ROOT / "knowledge_base.json",
]

#: The knowledge base is authored as one JSON file per topic under this folder.
#: ``data/knowledge_base.json`` is generated from these parts by
#: ``python -m src.build_kb``; if that merged file is missing, the loader
#: merges the parts directly so the app still runs.
KNOWLEDGE_TOPICS_DIR = DATA_DIR / "topics"

EVALUATION_QUESTIONS_PATH = DATA_DIR / "evaluation_questions.json"

#: Local caches for the (large) downloaded models.
WHISPER_CACHE_DIR = MODELS_DIR / "whisper"
HF_HOME = MODELS_DIR / "hf"

# ---------------------------------------------------------------------------
# Embedding / retrieval model
# ---------------------------------------------------------------------------
#: Pretrained multilingual sentence-embedding model (NOT trained by us).
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384

#: E5 models are trained with these asymmetric prefixes. Omitting them
#: noticeably degrades retrieval quality.
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

#: How many candidate records the retriever returns before the re-ranker
#: decides whether the question is answerable at all.
RETRIEVAL_TOP_K = 5

#: Use FAISS when available, else exact NumPy cosine similarity.
PREFER_FAISS = True

#: Batch size used while building the index (keeps RAM usage modest on laptops).
EMBED_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Relevance / fallback thresholds  (tuned on data/evaluation_questions.json)
# ---------------------------------------------------------------------------
#: Cosine similarity above which a candidate is considered possibly relevant.
#: NOTE: a cosine similarity is a GEOMETRIC similarity, NOT a probability.
RELEVANCE_THRESHOLD = 0.72

#: Gap between the best and the 2nd-best candidate. A *large* gap means the
#: query clearly points at one record; a *tiny* gap means the query is vague
#: and we should not confidently answer.
MIN_SCORE_MARGIN = 0.015

#: If the top candidate is below this score we never answer, no matter what.
HARD_REJECT_THRESHOLD = 0.66

#: Guard against degenerate input (single character, repeated punctuation...).
MIN_QUERY_LENGTH = 2

# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------
LANGUAGES = ["gu", "hi", "mr", "en", "roman_gu"]

LANGUAGE_LABELS = {
    "gu": "ગુજરાતી (Gujarati)",
    "hi": "हिन्दी (Hindi)",
    "mr": "मराठी (Marathi)",
    "en": "English",
    "roman_gu": "Roman Gujarati",
}

#: Languages the answer can be translated into.
TRANSLATION_TARGETS = ["en", "hi", "mr"]

#: Whisper language codes.
WHISPER_LANGUAGE_CODES = {
    "gu": "gu",
    "hi": "hi",
    "mr": "mr",
    "en": "en",
    "roman_gu": None,  # auto-detect
}

# ---------------------------------------------------------------------------
# Speech-to-Text (faster-whisper)
# ---------------------------------------------------------------------------
WHISPER_MODEL = os.environ.get("GUJJUVAANI_WHISPER_MODEL", "small")
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = os.environ.get("GUJJUVAANI_WHISPER_COMPUTE", "int8")

# ---------------------------------------------------------------------------
# Text-to-Speech
# ---------------------------------------------------------------------------
#: Order in which TTS backends are attempted.
TTS_BACKENDS = ["indic_tts", "edge_tts", "sapi"]

#: Neural voices (free, no API key) verified to exist on the Edge read-aloud
#: service. gu-IN / hi-IN / mr-IN are real Indian locales.
TTS_VOICES = {
    "gu": "gu-IN-DhwaniNeural",
    "hi": "hi-IN-SwaraNeural",
    "mr": "mr-IN-AarohiNeural",
    "en": "en-IN-NeerjaNeural",
}

#: Windows SAPI voice name fragments we look for, best first.
SAPI_VOICE_HINTS = {
    "gu": ("gujarati", "guj", "गुज", "ગુજ"),
    "hi": ("hindi", "हिंदी", "स्वरा", "madhur"),
    "mr": ("marathi", "मराठी"),
    "en": ("english", "zira", "david"),
}

# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
#: Backend preference order. Each entry is attempted in turn until one loads.
#: See src/translation.py for what every backend requires.
#:
#:   indictrans2 - AI4Bharat IndicTrans2. The academically preferred engine for
#:                 Indic languages, but the Hugging Face repos are now gated, so
#:                 it needs a FREE Hugging Face account:  huggingface-cli login
#:                 plus  pip install IndicTrans2
#:   nllb         - Meta NLLB-200 distilled 600M. Ungated, free, runs on CPU and
#:                 is fully offline after a one-time ~2.4 GB download.
#:                 Covers gu / hi / mr / en.  Verified working.
#:   deep-translator - clearly isolated LAST-RESORT network fallback. Never the
#:                 primary architecture; enable explicitly with the env var
#:                 GUJJUVAANI_TRANSLATOR=deep-translator.
#:   none         - only the curated short answers stored in the knowledge base
#:                 are shown.  The chat itself never needs translation.
TRANSLATION_BACKENDS = os.environ.get(
    "GUJJUVAANI_TRANSLATOR", "indictrans2,nllb,deep-translator"
).split(",")

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_APPROX_DOWNLOAD_MB = 2450

#: IndicTrans2 repositories (need `huggingface-cli login`).
INDICTRANS2_MODELS = {
    ("gu", "en"): "ai4Bharat/IndicTrans2-Indic2En",
    ("hi", "en"): "ai4Bharat/IndicTrans2-Indic2En",
    ("mr", "en"): "ai4Bharat/IndicTrans2-Indic2En",
    ("en", "gu"): "ai4Bharat/IndicTrans2-enIndic",
    ("en", "hi"): "ai4Bharat/IndicTrans2-enIndic",
    ("en", "mr"): "ai4Bharat/IndicTrans2-enIndic",
    ("gu", "hi"): "ai4Bharat/IndicTrans2-Indic2Indic",
    ("gu", "mr"): "ai4Bharat/IndicTrans2-Indic2Indic",
    ("hi", "gu"): "ai4Bharat/IndicTrans2-Indic2Indic",
    ("hi", "mr"): "ai4Bharat/IndicTrans2-Indic2Indic",
    ("mr", "gu"): "ai4Bharat/IndicTrans2-Indic2Indic",
    ("mr", "hi"): "ai4Bharat/IndicTrans2-Indic2Indic",
}

#: NLLB uses FLORES-200 style language codes.
NLLB_LANG_CODES = {
    "gu": "guj_Gujr",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "en": "eng_Latn",
}

#: Marian/seq2seq models truncate at ~512 tokens, so we translate sentence by
#: sentence. This is the soft character budget for one chunk.
TRANSLATION_CHUNK_CHARS = 900

# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------
APP_TITLE = "GujjuVaani"
APP_ICON = "🪔"
APP_TAGLINE = "Explore Gujarat. In Gujarati."
APP_SUBTITLE_GU = "ગુજરાતની સંસ્કૃતિ, ઇતિહાસ અને વારસો જાણો."

SUGGESTED_QUESTIONS = [
    ("🏛️", "Tell me about Rani ki Vav."),
    ("🦁", "Where can I see Asiatic lions in Gujarat?"),
    ("🎭", "How is Navratri celebrated in Gujarat?"),
    ("🌞", "Tell me about the Modhera Sun Temple."),
    ("🏰", "What is Champaner-Pavagadh Archaeological Park?"),
    ("🍲", "What are famous Gujarati foods?"),
    ("📜", "Tell me about Narsinh Mehta."),
    ("🛕", "ગુજરાતની ઉત્તરપાદિક વાવ વિશે જણાવો."),
]

FALLBACK_GU = (
    "માફ કરશો, હું ગુજરાતની સંસ્કૃતિ, ઇતિહાસ, વારસો અને પ્રવાસન "
    "વિશે માહિતી આપવા માટે બનાવવામાં આવ્યો છું. તમારો પ્રશ્ન મારા "
    "જ્ઞાનઆધારિત વિષયોની અંદર નથી. કૃપા કરીને ગુજરાતના સંસ્કૃતિ, "
    "વારસા, ઐતિહાસિક સ્થળો, તહેવારો, વરસાદ કે પ્રવાસન વિષે પૂછો."
)

FALLBACK_EN = (
    "Sorry, I am designed to answer questions about Gujarat's culture, "
    "history, heritage and tourism."
)

FALLBACK_HI = (
    "माफ कीजिए, मैं गुजरात की संस्कृति, इतिहास, विरासत और पर्यटन से जुड़ी "
    "जानकारी देने के लिए बनाया गया हूँ। आपका प्रश्न मेरे ज्ञान आधारित विषयों "
    "में नहीं है।"
)

#: Used when a record is found but simply does not carry enough information.
INSUFFICIENT_GU = (
    "માફ કરશો, આ માહિતી હાલમાં મારી જ્ઞાન આધારિત માહિતીમાં "
    "પૂરતી માત્રામાં ઉપલબ્ધ નથી."
)


def ensure_dirs() -> None:
    """Create the local cache directories (safe to call repeatedly)."""
    for path in (MODELS_DIR, WHISPER_CACHE_DIR, HF_HOME):
        path.mkdir(parents=True, exist_ok=True)