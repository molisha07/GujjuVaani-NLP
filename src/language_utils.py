"""Language identification and text normalisation.

GUJJUVAANI accepts five input "languages".  Three of them are written in a
distinct script, which makes identification almost deterministic:

===========  ====================  =========================================
code         script               example
===========  ====================  =========================================
``gu``       Gujarati block U+0A80 "રાણીની વાવ વિશે જણાવો"
``hi``       Devanagari U+0900    "रानी की वाव के बारे में बताओ"
``mr``       Devanagari U+0900    "राणी की वाव बद्दल माहिती द्या"
``en``       Latin                "Tell me about Rani ki Vav"
``roman_gu`` Latin                "Rani ki Vav kyare banavvama aavi hati?"
===========  ====================  =========================================

Hindi and Marathi share the Devanagari script, so for those we score function
words / orthographic markers.  English and Roman Gujarati share the Latin
script, so for those we score function words and Gujarati romanisation
markers.  This is a light-weight, fully offline, rule + lexicon based LID
system: no extra model download is required.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Script ranges
# ---------------------------------------------------------------------------
GUJARATI_RANGE = (0x0A80, 0x0AFF)
DEVANAGARI_RANGE = (0x0900, 0x097F)

_GUJARATI_CHARS = re.compile(r"[\u0a80-\u0aff]")
_DEVANAGARI_CHARS = re.compile(r"[\u0900-\u097f]")
_LATIN_CHARS = re.compile(r"[A-Za-z]")
_DIGITS = re.compile(r"[0-9]")

# ---------------------------------------------------------------------------
# Lexicons (deliberately small and high-precision)
# ---------------------------------------------------------------------------

#: Marathi-only markers. Marathi uses "आहे / आहेत / करा / काय / माहिती /
#: बद्दल / सांगा" where Hindi uses "है / हैं / करो / क्या / जानकारी / बारे /
#: बताओ". The presence of "आहे" or "माहिती" is a very strong Marathi signal.
MARATHI_MARKERS = {
    "आहे", "आहेत", "आले", "माहिती", "काय", "करा", "सांगा", "बद्दल",
    "मला", "तुम्ही", "कुठे", "कधी", "आणि", "पण", "म्हणून", "यांनी", "त्या",
}

#: Hindi markers.
HINDI_MARKERS = {
    "है", "हैं", "हो", "हूँ", "करो", "क्या", "कौन", "कब", "कहाँ", "कैसे",
    "जानकारी", "बताओ", "बताइए", "बारे", "मुझे", "आप", "यह", "वह", "क्यों",
    "और", "लिए", "वाला", "वाली",
}

#: High-signal Gujarati transliteration markers. Roman Gujarati (sometimes
#: called "Hinglish-style Gujarati") is written with Latin letters but Gujarati
#: phonology and syntax, e.g. "chhe", "che", "shu", "kem", "kyare", "aavu",
#: "karjo", "vishe", "janavo", "malo", "bhai", "tamne", "humne".
ROMAN_GUJARATI_MARKERS = {
    # verb / auxiliary forms
    "chhe", "che", "chhena", "hati", "hato", "havu", "havy", "karjo",
    "karyu", "karie", "karu", "kare", "karo", "karta", "jani", "janavu",
    "janav", "janie", "tamne", "tamaro", "humne", "humari", "jo", "je",
    # question words
    "shu", "su", "kem", "ken", "kyare", "kya", "kyu", "kuthare", "kahan",
    "shyamal", "kidhu", "kitli", "kevu",
    # postpositions / conjunctions / pronouns
    "mate", "matre", "na", "no", "ni", "nu", "ke", "ken", "la", "vala",
    "vishe", "vishesh", "baddha", "baddhi", "pachi", "par", "tathi", "aavyo",
    "aavyu", "aavo", "aavse", "gaya", "gayi", "thay", "thaya", "karine",
    "sathiyana", "sathi", "bhi", "pan", "pn",
    # pronouns
    "hum", "tame", "me", "tui", "teno", "tenu", "maru", "taru",
    # common verbs
    "khavo", "khelo", "gamvo", "gau", "gamtu", "rahe", "rahu", "vadh",
    "dikh", "dikhu", "shu", "veshe", "vadhare", "pelu", "pachi",
}

#: High-signal English function words. Used to separate ``en`` from ``roman_gu``.
ENGLISH_MARKERS = {
    "the", "is", "are", "was", "were", "of", "and", "to", "in", "on", "for",
    "with", "about", "tell", "me", "what", "which", "where", "when", "how",
    "why", "who", "can", "you", "i", "it", "this", "that", "there", "from",
    "do", "does", "did", "please", "known", "knowns", "famous", "give",
    "explain", "describe", "have", "has", "they", "we", "he", "she", "his",
    "her", "their", "our", "your", "my", "me", "an", "as", "at", "by", "be",
    "been", "also", "any", "some", "not", "but", "if", "so", "than", "then",
}

# ---------------------------------------------------------------------------
# Roman Gujarati -> Gujarati-script normalisation lexicon
# ---------------------------------------------------------------------------
# Roman Gujarati is not a separate language model.  We do not transliterate the
# whole query (that would need a trained transducer).  Instead we append the
# Gujarati-script equivalent of a few high-value function words to the query,
# purely as a *lexical prior* that helps the embedding model.  The semantic
# match itself is done by the multilingual encoder.
ROMAN_TO_GUJARATI: Dict[str, str] = {
    "chhe": "છે", "che": "છે", "chhena": "છે", "hati": "હતી", "hato": "હતો",
    "havu": "હવું", "havy": "હવું", "karjo": "કરજો", "karyu": "કરું",
    "kare": "કરે", "karo": "કરો", "karta": "કરતા", "janavi": "જણાવો",
    "janavo": "જણાવો", "janav": "જણાવો", "janie": "જણાવો", "jani": "જણાવો",
    "tamne": "તમને", "tamaro": "તમારું", "humne": "હમને", "humari": "હમારી",
    "shu": "શું", "su": "શું", "kem": "કેમ", "ken": "કેન", "kyare": "ક્યારે",
    "kya": "કયા", "kyu": "ક્યું", "kuthare": "ક્યુંથી", "kahan": "ક્યાં",
    "kidhu": "કિઠં", "kitli": "કેટલી", "kevu": "કેવું",
    "mate": "માટે", "matre": "માટે", "no": "નો", "na": "ના", "ni": "ની",
    "nu": "નું", "vala": "વાલા", "vishe": "વિશે", "baddha": "બધું",
    "baddhi": "બધી", "pachi": "પછી", "par": "પર", "tathi": "ત્યાં",
    "aavyo": "આવ્યો", "aavyu": "આવ્યું", "aavo": "આવો", "gaya": "ગયા",
    "gayi": "ગઈ", "thay": "થયા", "thaya": "થયા", "sathi": "સાથે",
    "sathiyana": "સાથે", "bhi": "પણ", "pan": "પણ", "pn": "પણ",
    "hum": "હમે", "tame": "તમે", "me": "મેં", "tui": "તું", "teno": "તેનો",
    "maru": "મારું", "vadhare": "વધારે", "pelu": "પેલું", "dikh": "દેખ",
    "dikhu": "દેખાય", "veshe": "વિશે",
}

#: Frequent English words that hint at a Gujarati entity even in Roman script,
#: mapped to the Gujarati-script spelling used in the knowledge base.
ENTITY_SPELLING_HINTS: Dict[str, str] = {
    "rani": "રાણી", "vav": "વાવ", "rani ki vav": "રાણીની વાવ",
    "rani-ki-vav": "રાણીની વાવ", "rani ki vaav": "રાણીની વાવ",
    "modhera": "મોઢેરા", "sun temple": "સૂર્ય મંદિર", "surya mandir": "સૂર્ય મંદિર",
    "somnath": "સોમનાથ", "dwarka": "દ્વારકા", "patan": "પાટણ",
    "ahmedabad": "અમદાવાદ", "surat": "સુરત", "vadodara": "વડોદરા",
    "rajkot": "રાજકોટ", "jamnagar": "જામનગર", "porbandar": "પોરબંદર",
    "bhuj": "ભુજ", "junagadh": "જૂનાગઢ", "gir": "ગીર", "kutch": "કચ્છ",
    "kutchi": "કચ્છી", "dholavira": "ધોલાવીરા", "lothal": "લોઠલ",
    "adalaj": "અદાલજ", "sidi saiyyed": "સિદી સૈયદ", "sarkhej": "સરખેજ",
    "champaner": "ચાંપાનેર", "pavagadh": "પાવાગઢ", "nal sarovar": "નળ સરોવર",
    "saputara": "સાપુતારા", "statue of unity": "સ્ટેચ્યુ ઓફ યુનિટી",
    "sardar patel": "સરદાર વલ્લભભાઈ પટેલ", "gandhi": "ગાંધી",
    "mahatma gandhi": "મહાત્મા ગાંધી", "narsinh mehta": "નરસિંહ મહેતા",
    "hemchandra": "હેમચંદ્ર", "navratri": "નવરાત્રી", "garba": "ગરબા",
    "dandiya": "દાંડિયા", "uttarayan": "ઉત્તરાયણ", "janmashtami": "જન્માષ્ટમી",
    "diwali": "દિવાળી", "rann": "રણ", "asiatic lion": "એશિયાઈ સિંહ",
    "lion": "સિંહ", "patola": "પટોળા", "bandhani": "બાંધણી",
    "sabarmati": "સાબરમતી", "sabarmati ashram": "સાબરમતી આશ્રમ", "ashram": "આશ્રમ",
    "kutchi embroidery": "કચ્છી એમ્બ્રોડરી", "undhiyu": "ઊંધિયું",
    "tarnetar": "તરણેતર", "bhavnath": "ભવનાથ", "vautha": "વૌઠા",
    "madhavpur": "માધવપુર", "rath yatra": "રથયાત્રા", "dang darbar": "ડાંગ દરબાર",
    "darshak": "દર્શક મનુભાઈ પંચોળી", "manubhai pancholi": "મનુભાઈ પંચોળી દર્શક",
    "meghani": "ઝવેરચંદ મેઘાણી", "premanand": "પ્રેમાનંદ ભટ્ટ",
    "dayaram": "દયારામ", "akho": "અખો", "kavi narmad": "કવિ નર્મદ",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def script_profile(text: str) -> Dict[str, float]:
    """Return the relative share of Gujarati / Devanagari / Latin characters."""
    total = max(len([c for c in text if c.isalpha()]), 1)
    gu = len(_GUJARATI_CHARS.findall(text))
    dv = len(_DEVANAGARI_CHARS.findall(text))
    lat = len(_LATIN_CHARS.findall(text))
    return {
        "gujarati": gu / total,
        "devanagari": dv / total,
        "latin": lat / total,
        "gujarati_raw": gu,
        "devanagari_raw": dv,
        "latin_raw": lat,
    }


def is_gujarati(text: str) -> bool:
    return _GUJARATI_CHARS.search(text) is not None


def is_devanagari(text: str) -> bool:
    return _DEVANAGARI_CHARS.search(text) is not None


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[\w']+", text.lower(), flags=re.UNICODE) if t]


def _score_lexicon(tokens: List[str], lexicon) -> int:
    return sum(1 for t in tokens if t in lexicon)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_language(text: str, hint: Optional[str] = None) -> str:
    """Identify the language of ``text``.

    ``hint`` is the language the user *selected* in the UI.  The hint is used
    only as a tie-breaker - we deliberately do not trust it blindly, because
    the UI selector is frequently wrong or left on the default.

    Returns one of ``gu``, ``hi``, ``mr``, ``en``, ``roman_gu``, ``und``.
    """
    if not text or not text.strip():
        return "und"

    prof = script_profile(text)
    tokens = _tokens(text)

    # 1) Gujarati script -> unambiguous.
    if prof["gujarati_raw"] >= 2 or prof["gujarati"] > 0.30:
        return "gu"

    # 2) Devanagari -> Hindi or Marathi, decided by function words.
    if prof["devanagari_raw"] >= 2 or prof["devanagari"] > 0.30:
        mr_score = _score_lexicon(tokens, MARATHI_MARKERS)
        hi_score = _score_lexicon(tokens, HINDI_MARKERS)
        # Generic Marathi orthography signals.
        if re.search(r"[\u0900-\u097F]ाहे|ाहेत|माहिती|काय|करा", text):
            mr_score += 2
        if re.search(r"[\u0900-\u097F]मध्ये|ला|ची|च्या", text):
            mr_score += 1
        if mr_score > hi_score:
            return "mr"
        if hi_score > mr_score:
            return "hi"
        return "mr" if hint == "mr" else ("hi" if hint == "hi" else "hi")

    # 3) Latin script -> English or Roman Gujarati.
    if prof["latin_raw"] >= 2:
        rg_score = _score_lexicon(tokens, ROMAN_GUJARATI_MARKERS)
        en_score = _score_lexicon(tokens, ENGLISH_MARKERS)
        # Gujarati romanisation has doubled consonants ("chhe", "vavv",
        # "matra", "aavyo") and its own question words. Nudge it up a little.
        if re.search(r"\b\w*chh\w*|\b\w*vav\b|\bmatra\b|\baav[a-z]+\b", text.lower()):
            rg_score += 1
        if rg_score > en_score:
            return "roman_gu"
        if en_score > rg_score:
            return "en"
        if rg_score == 0 and en_score == 0:
            # No signal at all: one or two tokens, likely a proper noun.
            return "en" if len(tokens) <= 2 else "roman_gu"
        return "roman_gu" if hint == "roman_gu" else "en"

    # 4) Digits / punctuation only.
    return "und" if not _DIGITS.search(text) else "en"


def normalize_text(text: str) -> str:
    """NFKC-normalise, collapse whitespace and lowercase for lexical work."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def roman_gujarati_gloss(query: str) -> str:
    """Append Gujarati-script glosses for Roman-Gujarati input.

    This is *not* transliteration and it is not a second model.  It is a small
    lexicon-based expansion that puts a handful of Gujarati tokens next to the
    user's query, which measurably helps the (already multilingual) encoder
    place the query near Gujarati passages.
    """
    tokens = _tokens(normalize_text(query))
    glosses: List[str] = []

    for token in tokens:
        gloss = ROMAN_TO_GUJARATI.get(token)
        if gloss and gloss not in glosses:
            glosses.append(gloss)

    lowered = normalize_text(query)
    for phrase, gloss in ENTITY_SPELLING_HINTS.items():
        if phrase in lowered and gloss not in glosses:
            glosses.append(gloss)

    return " ".join(glosses)


def expand_query(query: str, language: Optional[str] = None) -> str:
    """Language-aware query normalisation used by the retrieval layer.

    * strips filler politeness words that only add noise to embeddings
    * appends a Gujarati gloss when the input is Roman Gujarati
    * appends a couple of well-chosen Gujarati synonyms for common intents
    """
    cleaned = normalize_text(query)
    if not cleaned:
        return ""

    tokens = [t for t in cleaned.split() if t not in _FILLER_WORDS]
    if tokens:
        cleaned = " ".join(tokens)

    language = language or detect_language(query)

    if language == "roman_gu":
        gloss = roman_gujarati_gloss(query)
        if gloss:
            cleaned = f"{cleaned} {gloss}"
    elif language == "en":
        # A small, high-value intent vocabulary.  This is *query expansion*,
        # a standard IR technique - it does not add facts to the answer.
        hints = []
        lowered = cleaned
        if "temple" in lowered or "mandir" in lowered:
            hints.append("મંદિર")
        if "stepwell" in lowered or "vav" in lowered or "well" in lowered:
            hints.append("વાવ વિશે ઐતિહાસિક સ્થળ")
        if "lion" in lowered or "animal" in lowered or "wildlife" in lowered:
            hints.append("વન્યજીવો અને સ્થળ")
        if "festival" in lowered or "celebrat" in lowered:
            hints.append("તહેવાર ઉજવણી")
        if "food" in lowered or "dish" in lowered or "cuisine" in lowered:
            hints.append("ગુજરાતી ખાવાનું")
        if "who" in lowered or "who was" in lowered:
            hints.append("વ્યક્તિ જીવનચરિત્ર")
        if hints:
            cleaned = f"{cleaned} " + " ".join(dict.fromkeys(hints))

    return cleaned.strip()


#: Politeness / meta words that carry no topical signal.
_FILLER_WORDS = {
    "please", "kindly", "hi", "hello", "hey", "bhai", "yaar", "sir", "madam",
    "tell", "me", "about", "give", "some", "information", "info", "details",
    "detail", "explain", "describe", "what", "who", "when", "where", "why",
    "how", "is", "are", "the", "a", "an", "of", "and", "to", "in", "on", "for",
    "pls", "plz", "can", "you", "i", "want", "need", "know", "help",
    "krupa", "karjo", "please", "janavi", "janavo", "ma", "मुझे", "बताओ",
    "મને", "કરી", "જણાવો", "કરો",
}

#: Language display names, re-exported for convenience.
LANGUAGE_NAMES = {
    "gu": "Gujarati",
    "hi": "Hindi",
    "mr": "Marathi",
    "en": "English",
    "roman_gu": "Roman Gujarati",
    "und": "Unknown",
}


@lru_cache(maxsize=512)
def cached_detection(text: str, hint: Optional[str] = None) -> str:
    """Memoised :func:`detect_language` (the UI calls it repeatedly)."""
    return detect_language(text, hint)


def language_distribution(questions: List[Tuple[str, str]]) -> Dict[str, int]:
    """Count detected languages - used by the evaluation report."""
    counts: Dict[str, int] = {}
    for text, hint in questions:
        lang = detect_language(text, hint)
        counts[lang] = counts.get(lang, 0) + 1
    return counts