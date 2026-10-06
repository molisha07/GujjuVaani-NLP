# 🏛️ GujjuVaani (ગુજ્જુવાણી)
> **Multilingual AI Assistant for Gujarat Culture, Heritage, History & Tourism**

GujjuVaani is a free, open-source conversational AI assistant dedicated to Gujarat. You can ask questions in **Gujarati, English, Hindi, Marathi, or Hinglish (Roman script)** using **Text or Voice**, and get accurate, verified cultural answers with neural voice playback and instant translations.

---

## ✨ Key Features

- 🏛️ **140+ Curated Topics:** Verified knowledge covering monuments, temples, history, festivals, wildlife, food, and famous personalities.
- 🗣️ **Multilingual & Hinglish:** Ask in English, Gujarati (`ગુજરાતી`), Hindi (`हिंदी`), Marathi (`मराठी`), or Roman script (`"navratri kya hai"`).
- 🎙️ **Voice In & Voice Out:** Built-in microphone recording, offline **Faster-Whisper** speech-to-text, and natural neural voice speech synthesis.
- ⚡ **Hybrid Semantic Search:** Combines 384-dimensional **Multilingual E5** embeddings with lexical keyword matching for high accuracy (97%+ Top-1 match).
- 🌐 **1-Click Translations:** Translate any answer to English, Hindi, or Marathi with audio playback.
- 💡 **Ready-to-Ask Question Library:** Clickable English & Gujarati sample questions across monuments, culture, cuisine, and wildlife.
- 💻 **100% Free & Local:** Runs completely on CPU with no paid APIs required.

---

## 🚀 Quick Start (How to Run)

### 1. Clone the Repository
```bash
git clone https://github.com/molisha07/GujjuVaani-NLP.git
cd GujjuVaani-NLP
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🧠 How It Works

```
User Question (Text or Voice)
  ├── 🎙️ Voice Input → Transcribed locally with faster-whisper
  ├── 🌐 Language Identification → Auto-detects gu, hi, mr, en, or Hinglish
  ├── ⚡ Hybrid Vector Search → Multilingual E5 embeddings + Keyword matching
  ├── 🏛️ Curated Knowledge Base → Finds exact verified heritage answer
  └── 🔊 Output → Gujarati natural answer + Neural audio playback + Multi-language translations
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **UI Frontend** | Streamlit | Clean, modern light-theme interface |
| **Embeddings** | `multilingual-e5-small` | 384-d dense vector semantic search |
| **Vector Index** | FAISS / NumPy | Fast exact cosine similarity retrieval |
| **Speech-to-Text** | `faster-whisper` (int8) | Fast offline CPU speech recognition |
| **Text-to-Speech** | `edge-tts` / Neural Voices | Authentic regional speech playback |
| **Translation** | Curated KB + NLLB / IndicTrans | Multi-language translation support |

---

## 📁 Project Structure

```
GujjuVaani/
├── app.py                          # Streamlit application UI
├── data/
│   ├── knowledge_base.json         # 140+ curated Gujarati heritage records
│   └── evaluation_questions.json   # 35 test questions for benchmarking
├── src/
│   ├── chatbot.py                  # Core QA assistant & relevance gating
│   ├── retrieval.py                # Hybrid E5 dense & lexical vector search
│   ├── speech_to_text.py           # Offline faster-whisper speech recognition
│   ├── text_to_speech.py           # Neural voice synthesis engine
│   ├── translation.py              # Multilingual translation module
│   ├── language_utils.py           # Language detection & query expansion
│   ├── evaluation.py               # Live benchmark test suite
│   └── config.py                   # Global configuration & paths
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🧪 Running the Benchmark Test

To run the offline NLP accuracy evaluation over the test dataset:

```bash
python -m src.evaluation
```

**Benchmark Accuracy:**
- **Top-1 Retrieval Accuracy:** `97.1%`
- **Top-3 Retrieval Accuracy:** `100.0%`
- **Out-of-Domain Rejection:** `83.3%`

---

## 📄 License

This project is licensed under the **MIT License**.
