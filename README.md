# 🪔 GUJJUVAANI
### Multilingual Gujarati Culture, Heritage, History & Tourism AI Assistant
> **"Explore Gujarat. In Gujarati." • ગુજરાતની સંસ્કૃતિ, ઇતિહાસ અને વારસો જાણો.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Sentence Transformers](https://img.shields.io/badge/NLP-Sentence--Transformers-orange.svg)](https://www.sbert.net/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-blueviolet.svg)](https://github.com/facebookresearch/faiss)
[![Faster Whisper](https://img.shields.io/badge/STT-Faster--Whisper-green.svg)](https://github.com/SYSTRAN/faster-whisper)
[![Edge TTS](https://img.shields.io/badge/TTS-Neural%20Edge%20TTS-teal.svg)](https://github.com/rany2/edge-tts)

---

## 📋 1. Project Overview & Problem Statement

### 1.1 Problem Statement
General-purpose conversational AI models and commercial chatbots (like GPT-4 or Gemini) frequently suffer from hallucinations, lack deep regional historical context, and perform sub-optimally on low-resource Indic languages and regional scripts like Gujarati. Furthermore, students and academic institutions often cannot rely on expensive, paid APIs that require credit cards or cloud GPUs.

### 1.2 Motivation & Objective
**GUJJUVAANI** is a dedicated, **100% free-to-run, open-source Retrieval-Augmented Question Answering (RAG) assistant** engineered specifically for the domain of **Gujarat's culture, heritage, architecture, historical personalities, festivals, cuisine, folk arts, literature, wildlife, and tourism**.

The core objective is to allow users to interact in **English, Hindi, Marathi, Gujarati, or Roman Gujarati (Latin transliteration)** via **Text or Voice**, understand their intent using multilingual transformer sentence embeddings, retrieve ground-truth verified historical records, and deliver authentic answers primarily in **natural Gujarati script**, with on-demand **translation** and **neural speech synthesis**.

---

## 🏛️ 2. Domain Coverage

GUJJUVAANI covers 13 deep regional knowledge categories across **140+ fact-checked records**:

1. **Historical Places & Stepwells:** Rani ki Vav (Patan), Sun Temple (Modhera), Dholavira, Lothal, Champaner-Pavagadh, Adalaj Stepwell, Sidi Saiyyed Mosque, Sarkhej Roza, Uparkot Fort, Lakhpat Fort, Bhadra Fort, Jhulta Minar, Diu Fort, Taranga Jain Temples, Shamlaji Temple, Kirti Mandir Porbandar, Hira Bhagol Dabhoi, Tambekar Wada, etc.
2. **Heritage Sites & Royal Palaces:** Historic Walled City of Ahmedabad (UNESCO), Lukshmi Vilas Palace (Vadodara), Prag Mahal & Aina Mahal (Bhuj), Vijay Vilas Palace (Mandvi), Naulakha Palace (Gondal), Ranjit Vilas Palace (Wankaner), Lakhota Fort (Jamnagar), Khambhalida Buddhist Caves, etc.
3. **Tourism & Scenic Destinations:** Statue of Unity (182m, Kevadia/Ekta Nagar), Great Rann of Kutch (White Desert / Dhordo), Saputara Hill Station, Polo Forest (Vijaynagar), Shivrajpur Blue Flag Beach (Dwarka), Gira Waterfalls, Kalo Dungar (Black Hill), etc.
4. **Religious & Spiritual Shrines:** Somnath Jyotirlinga (1st of 12 Jyotirlingas), Dwarkadhish Jagat Mandir (Char Dham), Shatrunjaya Palitana Jain Temples (860+ marble temples), Ambaji Shaktipeeth, Akshardham (Gandhinagar), Udvada Atash Behram (sacred Zoroastrian fire), Nageshwar Jyotirlinga, Bahucharaji Temple, Sarangpur Kashtbhanjan Hanuman Temple, Virpur Jalaram Bapa Temple (Sadavrat), Bhalka Tirth, Koteshwar Mahadev & Narayan Sarovar.
5. **Historical Personalities:** Mahatma Gandhi (Father of the Nation), Sardar Vallabhbhai Patel (Iron Man of India), Narsinh Mehta (Adi Kavi), Jhaverchand Meghani (Rashtriya Shayar), Acharya Hemachandra (Kalikala-Sarvajna), Dr. Vikram Sarabhai (ISRO Founder), Shyamji Krishna Varma (India House), Morarji Desai (4th PM & Bharat Ratna), Swami Dayananda Saraswati (Arya Samaj), Ravishankar Maharaj (Muk Sevak), Umashankar Joshi (Jnanpith 'Nishith'), Kavi Kalapi, Akho (Chhappa), Premanand Bhatt (Manbhatt), Kavi Dalpatram, Kavi Narmad, Dhirubhai Ambani, Dr. Verghese Kurien & Tribhuvandas Patel (Amul / White Revolution), Kasturba Gandhi, Maharaja Sayajirao Gaekwad III, Manubhai Pancholi 'Darshak'.
6. **Festivals & Folk Fairs:** Navratri & Garba (UNESCO Intangible Cultural Heritage 2023), Uttarayan & International Kite Festival, Tarnetar Folk Fair, Bhavnath Mahadev Fair (Girnar Naga Sadhus), Vautha Animal Fair (Saptasangam), Madhavpur Ghed Fair, Ahmedabad Jagannath Rath Yatra, Dang Darbar (Ahwa), Modhera Classical Dance Festival, Chitra Vichitra Fair (Gunbhakhari).
7. **Gujarati Cuisine & Food Culture:** Traditional Gujarati Thali, Surti Undhiyu & Puri, Dhokla vs Nylon Khaman, Fafda-Jalebi (Dussehra tradition), Thepla & Khakhra (travel food), Kutchi Dabeli, Handvo, Surti Locho, Dal Dhokli, Mohanthal & Shrikhand, Surti Ghari (Chandi Padvo), Kathiyawadi Bajra Rotlo & Ringna no Olo, Amdavadi Dalwada, Bhavnagari Ganthiya, Mahudi Sukhdi, Patra & Muthia, Sev Khamani & Sev Usal.
8. **Handicrafts & Folk Arts:** Patan Patola (Double Ikat Silk - GI Tag), Rogan Art (Nirona Kutch), Lippan Kaam (Mud Mirror Work & Bhungas), Bandhani (Tie & Dye - GI Tag), Sankheda Lacquered Furniture (GI Tag), Mata ni Pachedi (Sacred Textile - GI Tag), Tangaliya Shawl (Daana Weaving - GI Tag), Khambhat Agate/Akik Craft (GI Tag), Kutch Embroidery (Rabari, Ahir, Mutwa), Kutch Copper Bells (Zura).
9. **Gujarati Literature & Folklore:** 'Jai Jai Garvi Gujarat' (State Anthem by Narmad), 'Saraswatichandra' (Govardhanram Tripathi), 'Manvini Bhavai' (Pannalal Patel - Jnanpith 1985), 'Saurashtra ni Rasdhar' (Jhaverchand Meghani), 'Vaishnav Jan To' (Narsinh Mehta), Dayaram's Garbi, Akha na Chhappa, 'Nishith' (Umashankar Joshi).
10. **Wildlife & National Parks:** Gir National Park (only natural habitat of Asiatic Lions), Little Rann of Kutch Wild Ass Sanctuary (Indian Wild Ass - Khur), Nal Sarovar Bird Sanctuary (Ramsar Site), Velavadar Blackbuck National Park, Marine National Park Jamnagar (Pirotan Island Corals), Thol Lake Bird Sanctuary (Ramsar Site), Khijadiya Bird Sanctuary (Ramsar Site), Vansda National Park, Ratanmahal Sloth Bear Sanctuary.
11. **Geography & Natural Wonders:** Mount Girnar (1117m, Gorakhnath Peak & Ropeway), Narmada River & Sardar Sarovar Dam, Gujarat Coastline (1600 km - longest in India), Bhal Region & Bhalia Wheat (GI Tag), Charotar Region (Land of Golden Leaf), Sabarmati & Tapi Rivers, Gulf of Kutch & Khambhat.
12. **Historical Events & Movements:** Dandi March / Salt Satyagraha (1930), Mahagujarat Movement & Gujarat State Formation (1 May 1960), Bardoli Satyagraha (1928), Kheda Satyagraha (1918), Integration of Junagadh & Aarzi Hukumat (1947), Navnirman Student Movement (1974).
13. **Gujarati Culture & Social Values:** Gujarati Language & Dialects (Kathiyawadi, Surti, Charotari, Kutchi), Gujarati Wedding Customs (Pithi, Mameru, Hastamelap), Traditional Attire (Chaniya Choli, Kedia, Paghdi), 'Atithi Devo Bhava' & Gujarati Hospitality.

---

## 🏗️ 3. System Architecture & NLP Pipeline

```
                              USER
                                |
                    ┌───────────┴───────────┐
                    |                       |
                  TEXT                    VOICE
                    |                       |
                    |              Faster-Whisper (STT)
                    |                       |
                    └───────────┬───────────┘
                                |
                    Language Identification (LID)
                    [gu, hi, mr, en, roman_gu]
                                |
                    Query Normalization & Expansion
                    [Entity Glossing & Canonical Form]
                                |
                    Multilingual Dense Embedding
                    [intfloat/multilingual-e5-small]
                    "query: <normalized_text>"
                                |
                    Semantic Retrieval (FAISS IndexFlatIP)
                    [Cosine Similarity Matrix]
                                |
                    Knowledge Base Records (140+ Topics)
                    "passage: <title . questions . answer>"
                                |
                    Relevance Gate & Domain Classifier
                       /                         \
         Score >= 0.72 & In-Domain        Score < 0.66 / OOD
                     /                             \
        Relevant Gujarati Answer          Honest Cultural Fallback
                    |                      "માફ કરશો, હું ગુજરાત..."
          ┌─────────┴─────────┐
          |                   |
    Neural Translation    Neural Text-to-Speech
    [IndicTrans2/NLLB]    [Edge Neural / SAPI]
          |                   |
    English / Hindi      Gujarati Audio Playback
          |                   |
          └─────────┬─────────┘
                    |
                  USER
```

---

## ⚙️ 4. Technology Stack & Design Decisions

| Component | Technology | Rationale & Design Decision |
| :--- | :--- | :--- |
| **Frontend** | Streamlit (Python) | High-productivity interactive UI with custom CSS design tokens inspired by Gujarat's culture. |
| **Language** | Python 3.11 | Modern typing, performance, and cross-platform compatibility. |
| **Sentence Embeddings** | `intfloat/multilingual-e5-small` | 384-dimensional dense multilingual embedding space covering Gujarati, Hindi, Marathi, and English. Pretrained on asymmetric `query:` / `passage:` prefixes. |
| **Vector Retrieval** | `faiss-cpu` (IndexFlatIP) / Cosine Similarity | Exact inner product search on L2-normalized vectors mathematically equivalent to cosine similarity. |
| **Speech-to-Text** | `faster-whisper` (CTranslate2 int8) | Runs Whisper models locally on CPU with int8 quantization. |
| **Text-to-Speech** | `edge-tts` / Windows SAPI / Indic-TTS | Free neural Indian voices (`gu-IN-DhwaniNeural`, `hi-IN-SwaraNeural`, `en-IN-NeerjaNeural`). |
| **Machine Translation** | AI4Bharat `IndicTrans2` / Meta `NLLB-200` / Fallback | Open-source Indic translation presentation layer for English/Hindi toggles. |
| **Knowledge Store** | JSON (`data/knowledge_base.json`) | Clean, human-auditable, version-controlled structured knowledge base. |

---

## 🚀 5. Installation & Setup Instructions (Windows / Linux / macOS)

### 5.1 Prerequisites
- Python 3.11 installed
- Git installed
- Microphone (for voice input demo)

### 5.2 Step-by-Step Setup

```bash
# 1. Clone or navigate to the project directory
cd "c:\Users\Molisha Jain\Desktop\mini-nlp\GujjuVaani"

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Windows Command Prompt:
.venv\Scripts\activate.bat
# On Linux / macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify the Knowledge Base
python -m src.generate_full_kb

# 6. Run the Streamlit Application
streamlit run app.py
```

---

## 🧪 6. Running the NLP Evaluation Benchmark

To run the offline evaluation benchmark over the test dataset:

```bash
# Run evaluation benchmark
python -m src.evaluation

# Run with verbose match details
python -m src.evaluation --verbose
```

### 6.1 Actual Benchmark Results

```
==============================================================
GUJJUVAANI NLP EVALUATION
==============================================================

Total questions (in-domain): 35

-- Retrieval quality (in-domain questions) ----------------------
Top-1 Retrieval Accuracy   : 100.00 %
Top-3 Retrieval Accuracy   : 100.00 %
Answered (not refused)     : 100.00 %
Average similarity score   : 0.8785   (cosine similarity, NOT a probability)
Active threshold           : 0.72
Hard-reject threshold      : 0.66
Minimum score margin       : 0.015

-- Per-language accuracy -------------------------------------------
language               n     top-1     top-3   avg sim
en                     7    100.0%    100.0%    0.8838
gu                    10    100.0%    100.0%    0.8763
hi                     6    100.0%    100.0%    0.8706
mr                     5    100.0%    100.0%    0.8746
roman_gu               7    100.0%    100.0%    0.8858

-- Out-of-domain rejection -------------------------------------
Probes                    : 12
Correctly refused         : 12
Out-of-domain rejection   : 100.00 %
==============================================================
```

---

## 💡 7. Academic Concepts & Viva Defense Guide

### 7.1 Multilingual Dense Semantic Retrieval
- **Pretrained vs Trained:** We do **not** train a transformer model from scratch; we leverage the pretrained `intfloat/multilingual-e5-small` model.
- **Asymmetric E5 Prefixing:** E5 models are trained with asymmetric prefixes: `passage: ` for knowledge passages and `query: ` for user questions. Omitting these degrades cosine similarity by 10-15%.
- **Vector Normalization:** Embedding vectors are L2-normalized so that inner product $\mathbf{u} \cdot \mathbf{v}$ is identical to cosine similarity $\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$.

### 7.2 Why Nearest-Neighbor is Not Enough (Relevance Gate)
A naive vector database will always return the closest vector, even for unrelated questions (e.g. "What is Python?" might return Gir National Park because Python is also a snake). GujjuVaani implements:
1. **Hard Reject Floor ($0.66$):** Any top hit below $0.66$ is immediately rejected.
2. **Relevance Threshold ($0.72$):** Cosine score floor for confident answers.
3. **Out-of-Domain Marker Lexicon:** Detects programming/finance/unrelated queries and triggers the cultural fallback.

### 7.3 Speech-to-Text & Text-to-Speech
- **STT:** `faster-whisper` uses CTranslate2 to perform int8 quantized inference of Whisper on CPU.
- **TTS:** `edge-tts` streams neural Indian voices (`gu-IN-DhwaniNeural` for Gujarati, `hi-IN-SwaraNeural` for Hindi) with authentic regional prosody and zero API costs.

---

## 📁 8. Project Structure

```
GujjuVaani/
├── app.py                          # Streamlit UI with 4 interactive tabs
├── knowledge_base.json             # Root copy of the 140+ knowledge records
├── requirements.txt                # Dependency list
├── README.md                       # Comprehensive documentation
├── .gitignore                      # Git ignore rules
│
├── data/
│   ├── knowledge_base.json         # Master 140+ verified Gujarati records
│   └── evaluation_questions.json   # 35 multilingual test questions
│
├── models/
│   ├── whisper/                    # Local cache for STT models
│   └── hf/                         # Local cache for HF models
│
├── src/
│   ├── __init__.py                 # Module initializer
│   ├── config.py                   # Central paths and threshold configs
│   ├── retrieval.py                # E5 Embedding & FAISS vector search
│   ├── chatbot.py                  # Relevance gate & RAG answer composer
│   ├── language_utils.py           # LID, script profiler, query expander
│   ├── translation.py              # IndicTrans2 / NLLB / Fallback MT
│   ├── speech_to_text.py           # Faster-Whisper audio transcription
│   ├── text_to_speech.py           # Edge Neural TTS / Windows SAPI
│   ├── evaluation.py               # Evaluation benchmark suite
│   ├── build_kb.py                 # Core knowledge base authoring
│   ├── build_full_kb.py            # Expanded knowledge builder
│   └── generate_full_kb.py         # Complete 140+ record generator
│
└── assets/
    └── logo/
        └── gujjuvaani_logo.jpg     # Official GujjuVaani emblem
```

---

## 📚 9. References & Sources
- **UNESCO World Heritage Centre:** Rani ki Vav (1307), Historic City of Ahmedabad (1551), Champaner-Pavagadh (1101), Dholavira (1644), Garba of Gujarat (Intangible Cultural Heritage 01962).
- **Archaeological Survey of India (ASI):** ASI Monuments of National Importance in Gujarat Circle (Vadodara Circle).
- **Gujarat Tourism (TCGL):** Official Tourism Portal of Gujarat (`gujarattourism.com`).
- **Ministry of Environment, Forest and Climate Change (MoEFCC):** Ramsar Convention Wetlands of Gujarat (Nal Sarovar, Thol, Khijadiya, Wadhwana).
- **Gujarati Sahitya Parishad & Sahitya Akademi:** Gujarati literary history, biographies, and editions.

---

**🪔 GujjuVaani • Explore Gujarat. In Gujarati.**
