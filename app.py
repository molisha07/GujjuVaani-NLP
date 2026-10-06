"""GUJJUVAANI: Multilingual Gujarati Culture, Heritage, History & Tourism AI Assistant.
Clean, modern, lightweight Streamlit Application with hybrid semantic search, 
offline speech recognition, neural voice synthesis, and multi-language support.
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.chatbot import ChatResponse, GujjuVaaniChatbot, get_chatbot
from src.evaluation import EvalReport, evaluate, load_evaluation_questions
from src.language_utils import LANGUAGE_NAMES, detect_language, expand_query, script_profile
from src.retrieval import KnowledgeRetriever, get_retriever
from src.speech_to_text import decode_audio, transcribe_bytes, transcribe_uploaded, whisper_is_installed
from src.text_to_speech import synthesize, tts_status
from src.translation import translate, translation_status

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GujjuVaani • ગુજરાતી વારસો અને સંસ્કૃતિ સહાયક",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean, Beautiful, Highly-Legible Light Theme Design System
LIGHT_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Gujarati:wght@400;500;600;700&family=Outfit:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
    --brand-primary: #D97706;
    --brand-primary-dark: #B45309;
    --brand-saffron: #EA580C;
    --brand-emerald: #059669;
    --brand-blue: #2563EB;
    --bg-page: #F8FAFC;
    --bg-card: #FFFFFF;
    --border-color: #E2E8F0;
    --border-hover: #CBD5E1;
    --text-primary: #0F172A;
    --text-secondary: #334155;
    --text-muted: #64748B;
    --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.06), 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', 'Noto Sans Gujarati', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
    background-color: #FAFAFA;
}

/* Header Banner */
.brand-header {
    background: #FFFFFF;
    border: 1px solid var(--border-color);
    border-top: 4px solid var(--brand-primary);
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 20px;
    box-shadow: var(--shadow-sm);
}

.brand-title {
    font-family: 'Outfit', 'Noto Sans Gujarati', sans-serif;
    font-size: 2.15rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: -0.02em;
}

.brand-title span.highlight {
    color: var(--brand-primary);
}

.brand-subtitle {
    color: var(--text-secondary);
    font-size: 1.02rem;
    margin-top: 5px;
    font-weight: 500;
}

.brand-tagline {
    color: var(--brand-saffron);
    font-size: 0.92rem;
    font-weight: 500;
    margin-top: 4px;
}

/* Badges */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 11px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
    margin-top: 6px;
}

.badge-amber {
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #FDE68A;
}

.badge-green {
    background: #D1FAE5;
    color: #065F46;
    border: 1px solid #A7F3D0;
}

.badge-blue {
    background: #DBEAFE;
    color: #1E40AF;
    border: 1px solid #BFDBFE;
}

.badge-slate {
    background: #F1F5F9;
    color: #334155;
    border: 1px solid #E2E8F0;
}

/* Chat & Response Cards */
.chat-card {
    background: #FFFFFF;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 16px;
    box-shadow: var(--shadow-sm);
}

.chat-card-user {
    background: #F8FAFC;
    border-left: 4px solid var(--brand-blue);
}

.chat-card-bot {
    background: #FFFFFF;
    border-left: 4px solid var(--brand-emerald);
}

.gujarati-answer-text {
    font-family: 'Noto Sans Gujarati', 'Plus Jakarta Sans', sans-serif;
    font-size: 1.15rem;
    line-height: 1.85;
    color: #0F172A;
    margin-top: 10px;
    margin-bottom: 12px;
    font-weight: 400;
}

.source-box {
    background: #F8FAFC;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.86rem;
    color: var(--text-secondary);
    margin-top: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Question Library Cards */
.question-item-card {
    background: #FFFFFF;
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    transition: all 0.2s ease;
    box-shadow: var(--shadow-sm);
}

.question-item-card:hover {
    border-color: var(--brand-primary);
    background: #FFFBEB;
    transform: translateY(-1px);
}

/* Voice Panel */
.voice-simple-box {
    background: #FFFFFF;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 18px;
    box-shadow: var(--shadow-sm);
}

/* Button Styling */
div.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.15s ease;
    border: 1px solid var(--border-color);
    background: #FFFFFF;
    color: var(--text-primary);
}

div.stButton > button:hover {
    border-color: var(--brand-primary);
    color: var(--brand-primary);
    background: #FFFBEB;
    transform: translateY(-1px);
}

/* Metrics */
.metric-box {
    background: #FFFFFF;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: var(--shadow-sm);
}

.metric-val {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--brand-primary);
}

.metric-lbl {
    font-size: 0.84rem;
    color: var(--text-muted);
    margin-top: 4px;
}
</style>
"""
st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_chatbot() -> GujjuVaaniChatbot:
    """Load and cache the GujjuVaani chatbot singleton."""
    return get_chatbot()


@st.cache_data(show_spinner=False)
def load_all_records() -> List[Dict[str, Any]]:
    """Load all knowledge records for the Knowledge Explorer."""
    bot = load_cached_chatbot()
    return bot.retriever.records


# ---------------------------------------------------------------------------
# Initialize Session State
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "current_response" not in st.session_state:
    st.session_state["current_response"] = None

if "translation_view" not in st.session_state:
    st.session_state["translation_view"] = None

if "tts_audio_cache" not in st.session_state:
    st.session_state["tts_audio_cache"] = {}

if "eval_report" not in st.session_state:
    st.session_state["eval_report"] = None

if "selected_sample_query" not in st.session_state:
    st.session_state["selected_sample_query"] = None

if "mic_input_key_counter" not in st.session_state:
    st.session_state["mic_input_key_counter"] = 0


# ---------------------------------------------------------------------------
# Header Component (Clean, Modern, Typographical)
# ---------------------------------------------------------------------------
def render_header():
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-title">GujjuVaani <span class="highlight">• ગુજ્જુવાણી</span></div>
            <div class="brand-subtitle">Multilingual AI Assistant for Gujarati Culture, Heritage, History & Tourism</div>
            <div class="brand-tagline">"Explore Gujarat. In Gujarati." • ગુજરાતની સંસ્કૃતિ, ઇતિહાસ અને વારસો જાણો</div>
            <div style="margin-top: 8px;">
                <span class="badge-pill badge-amber">🏛️ 140+ Verified Topics</span>
                <span class="badge-pill badge-green">⚡ Hybrid E5 Vector Search</span>
                <span class="badge-pill badge-blue">🎙️ Offline Whisper Voice STT</span>
                <span class="badge-pill badge-slate">🇮🇳 Free & Open Source</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_header()


# ---------------------------------------------------------------------------
# Helper: Process User Query
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helper: Process User Query
# ---------------------------------------------------------------------------
def process_user_query(
    query_text: str,
    lang_hint: str = "gu",
    is_voice: bool = False,
    audio_bytes: Optional[bytes] = None,
):
    """Run query through the chatbot pipeline, update session state, and history."""
    clean_query = query_text.strip()
    if not clean_query:
        return

    with st.spinner("વિશ્લેષણ અને શોધ ચાલુ છે... / Retrieving Gujarati heritage knowledge..."):
        try:
            chatbot = load_cached_chatbot()
            response = chatbot.ask(clean_query, language_hint=lang_hint)
            st.session_state["current_response"] = response
            st.session_state["current_question_audio"] = audio_bytes
            st.session_state["translation_view"] = None

            user_display = f"🎙️ {clean_query}" if is_voice else clean_query
            st.session_state["messages"].append(
                {
                    "role": "user",
                    "text": user_display,
                    "time": time.strftime("%H:%M"),
                    "audio": audio_bytes,
                    "is_voice": is_voice,
                }
            )
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "response": response,
                    "time": time.strftime("%H:%M"),
                }
            )
        except Exception as e:
            st.error(f"Error processing question: {e}")


# ---------------------------------------------------------------------------
# Sidebar: Language & Clean Navigation (DB info moved to Tab 5)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings / નિયંત્રણ")

    # Language Choice
    selected_lang_label = st.selectbox(
        "પ્રશ્નની ભાષા / Input Language",
        options=list(config.LANGUAGE_LABELS.values()),
        index=0,
        help="Select your input language. GujjuVaani auto-detects Gujarati, English, Hindi, Marathi, and Roman Gujarati.",
    )
    lang_code = "gu"
    for code, label in config.LANGUAGE_LABELS.items():
        if label == selected_lang_label:
            lang_code = code
            break

    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.caption("• You can ask in Gujarati, English, Hindi, Marathi, or Roman Gujarati (Hinglish/Gujlish).")
    st.caption("• Check the **'Sample Questions'** tab for ready-to-ask English & Gujarati questions.")
    st.caption("• Click **'Listen in Gujarati'** on any response to hear natural neural speech.")

    st.markdown("---")
    if st.button("🗑️ Clear Conversation / વાતચીત સાફ કરો", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["current_response"] = None
        st.session_state["current_question_audio"] = None
        st.session_state["active_sample_audio"] = None
        st.session_state["translation_view"] = None
        st.session_state["tts_audio_cache"] = {}
        st.session_state["selected_sample_query"] = None
        st.session_state["mic_input_key_counter"] += 1
        st.rerun()

    st.markdown("---")
    st.caption("GujjuVaani NLP • Gujarat Heritage Assistant")


# ---------------------------------------------------------------------------
# Main Tabs Layout
# ---------------------------------------------------------------------------
tab_chat, tab_questions, tab_explorer, tab_pipeline, tab_eval_system = st.tabs(
    [
        "💬 Chat & Assistant (પ્રશ્નોત્તરી)",
        "💡 Sample Questions in English (ઇંગ્લિશ પ્રશ્નો)",
        "🏛️ Heritage Explorer (વારસો દર્શન)",
        "🔬 NLP Pipeline (પાઇપલાઇન વિશ્લેષક)",
        "📊 Benchmark & Database Info (મૂલ્યાંકન અને સિસ્ટમ)",
    ]
)


# ===========================================================================
# TAB 1: CHAT & ASSISTANT
# ===========================================================================
with tab_chat:
    # 1. Quick Clickable Suggested Questions
    st.markdown("##### 💡 Suggested Questions (Click to Ask):")
    cols_sug = st.columns(4)
    suggested = [
        ("🏛️ રાણીની વાવ", "રાણીની વાવ વિશે જણાવો અને તેનું નિર્માણ કોણે કરાવ્યું હતું?"),
        ("🦁 ગીરના સિંહો", "ગીર રાષ્ટ્રીય ઉદ્યાનમાં એશિયાઈ સિંહ વિશે માહિતી આપો."),
        ("🎭 નવરાત્રી અને ગરબા", "ગુજરાતમાં નવરાત્રી કેવી રીતે ઉજવાય છે?"),
        ("🌞 મોઢેરા સૂર્ય મંદિર", "Tell me about the Sun Temple in Modhera."),
        ("🏰 ચાંપાનેર હેરિટેજ", "ચાંપાનેર-પાવાગઢ વર્લ્ડ હેરિટેજ સાઇટ વિશે જણાવો."),
        ("🍲 ગુજરાતી વાનગીઓ", "ગુજરાતી થાળી અને ઊંધિયાની વિશેષતા શું છે?"),
        ("📜 નરસિંહ મહેતા", "ભક્તકવિ નરસિંહ મહેતા અને તેમનાં પદો વિશે જણાવો."),
        ("🗽 સ્ટેચ્યુ ઓફ યુનિટી", "Statue of Unity kahan aavelu chhe ane ketli unchi chhe?"),
    ]

    for idx, (lbl, q_text) in enumerate(suggested):
        with cols_sug[idx % 4]:
            if st.button(lbl, key=f"sug_{idx}", use_container_width=True):
                st.session_state["selected_sample_query"] = q_text
                process_user_query(q_text, lang_hint=lang_code)
                st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 2. Simple, Intuitive Voice Input & Audio Demo Panel
    with st.expander("🎙️ Simple Voice Input & Quick Audio Demos (Click to expand)", expanded=False):
        v_col1, v_col2 = st.columns([0.50, 0.50])

        with v_col1:
            st.markdown("##### 🎙️ Record Your Voice Question:")
            st.caption("Click the mic, speak your question, then click Stop:")
            
            mic_key = f"chat_mic_{st.session_state['mic_input_key_counter']}"
            if hasattr(st, "audio_input"):
                mic_audio = st.audio_input("Record Question", key=mic_key, label_visibility="collapsed")
                
                if mic_audio is not None:
                    audio_bytes = mic_audio.getvalue() if hasattr(mic_audio, "getvalue") else mic_audio.read()
                    
                    if audio_bytes and len(audio_bytes) > 64:
                        st.audio(audio_bytes)
                        
                        btn_c1, btn_c2 = st.columns([0.65, 0.35])
                        with btn_c1:
                            if st.button("⚡ Ask with Recorded Voice", type="primary", key="btn_ask_voice", use_container_width=True):
                                with st.spinner("Transcribing audio with offline faster-whisper..."):
                                    try:
                                        transcription = transcribe_bytes(
                                            audio_bytes,
                                            filename=getattr(mic_audio, "name", "recording.wav") or "recording.wav",
                                            language=lang_code if lang_code in ["gu", "hi", "mr", "en"] else None,
                                        )
                                        if transcription.text and transcription.text.strip():
                                            st.session_state["selected_sample_query"] = transcription.text.strip()
                                            process_user_query(
                                                transcription.text.strip(),
                                                lang_hint=lang_code,
                                                is_voice=True,
                                                audio_bytes=audio_bytes,
                                            )
                                            st.rerun()
                                        else:
                                            st.warning("No speech recognized. Please speak a little louder or closer to the mic.")
                                    except Exception as exc:
                                        st.error(f"Voice Processing Notice: {exc}")
                        with btn_c2:
                            if st.button("🔄 Record Again", key="btn_reset_rec", use_container_width=True):
                                st.session_state["mic_input_key_counter"] += 1
                                st.rerun()
            else:
                st.info("Browser live recording requires Streamlit >= 1.40.")

        with v_col2:
            st.markdown("##### ⚡ 1-Click Voice Test Demos:")
            st.caption("Click any demo to hear the spoken question & get instant answers:")
            
            # Voice Test Demo 1: Gujarati
            if st.button("🏛️ રાણીની વાવ (Gujarati Voice Demo)", key="demo_voice_1", use_container_width=True):
                with st.spinner("Synthesizing audio & retrieving answer..."):
                    synth = synthesize("રાણીની વાવ વિશે જણાવો", language="gu")
                    trans = transcribe_bytes(synth.audio, filename="demo_rani.mp3", language="gu")
                    q_text = trans.text.strip() or "રાણીની વાવ વિશે જણાવો"
                    st.session_state["active_sample_audio"] = synth.audio
                    st.session_state["active_sample_text"] = q_text
                    st.session_state["active_sample_lang"] = "gu"
                    st.session_state["selected_sample_query"] = q_text
                    process_user_query(q_text, lang_hint="gu", is_voice=True, audio_bytes=synth.audio)
                    st.rerun()

            # Voice Test Demo 2: English
            if st.button("🦁 Asiatic Lions (English Voice Demo)", key="demo_voice_2", use_container_width=True):
                with st.spinner("Synthesizing audio & retrieving answer..."):
                    synth = synthesize("Where can I see Asiatic lions in Gujarat?", language="en")
                    trans = transcribe_bytes(synth.audio, filename="demo_lions.mp3", language="en")
                    q_text = trans.text.strip() or "Where can I see Asiatic lions in Gujarat?"
                    st.session_state["active_sample_audio"] = synth.audio
                    st.session_state["active_sample_text"] = q_text
                    st.session_state["active_sample_lang"] = "en"
                    st.session_state["selected_sample_query"] = q_text
                    process_user_query(q_text, lang_hint="en", is_voice=True, audio_bytes=synth.audio)
                    st.rerun()

            # Voice Test Demo 3: Hindi
            if st.button("🌞 सूर्य मंदिर (Hindi Voice Demo)", key="demo_voice_3", use_container_width=True):
                with st.spinner("Synthesizing audio & retrieving answer..."):
                    synth = synthesize("मोढेरा के सूर्य मंदिर के बारे में बताओ", language="hi")
                    trans = transcribe_bytes(synth.audio, filename="demo_modhera.mp3", language="hi")
                    q_text = trans.text.strip() or "मोढेरा के सूर्य मंदिर के बारे में बताओ"
                    st.session_state["active_sample_audio"] = synth.audio
                    st.session_state["active_sample_text"] = q_text
                    st.session_state["active_sample_lang"] = "hi"
                    st.session_state["selected_sample_query"] = q_text
                    process_user_query(q_text, lang_hint="hi", is_voice=True, audio_bytes=synth.audio)
                    st.rerun()

            # Voice Test Demo 4: Marathi
            if st.button("🏰 सोमनाथ मंदिर (Marathi Voice Demo)", key="demo_voice_4", use_container_width=True):
                with st.spinner("Synthesizing audio & retrieving answer..."):
                    synth = synthesize("सोमनाथ ज्योतिર્લિંગ मंदिराची माहिती द्या", language="mr")
                    trans = transcribe_bytes(synth.audio, filename="demo_somnath.mp3", language="mr")
                    q_text = trans.text.strip() or "સોમનાથ મંદિર વિશે માહિતી આપો"
                    st.session_state["active_sample_audio"] = synth.audio
                    st.session_state["active_sample_text"] = q_text
                    st.session_state["active_sample_lang"] = "mr"
                    st.session_state["selected_sample_query"] = q_text
                    process_user_query(q_text, lang_hint="mr", is_voice=True, audio_bytes=synth.audio)
                    st.rerun()

        # Display active demo audio player if available
        if st.session_state.get("active_sample_audio"):
            st.markdown("---")
            st.markdown(f"**🔊 Spoken Demo Audio Player ({st.session_state.get('active_sample_lang', 'gu').upper()}):**")
            st.audio(st.session_state["active_sample_audio"])
            st.caption(f"✓ Transcribed Question: **\"{st.session_state.get('active_sample_text', '')}\"**")

    # 3. Main Text Input Form
    default_text = st.session_state.get("selected_sample_query") or ""
    with st.form(key="ask_form", clear_on_submit=False):
        col_in, col_btn = st.columns([0.84, 0.16])
        with col_in:
            user_query = st.text_input(
                "તમારો પ્રશ્ન પૂછો / Ask your question:",
                value=default_text,
                placeholder="દા.ત. રાણીની વાવ વિશે જણાવો / Tell me about Rani ki Vav / navratri kya hai",
                key="main_user_query_input",
                label_visibility="collapsed",
            )
        with col_btn:
            submit_btn = st.form_submit_button("🔍 પૂછો / Ask", type="primary", use_container_width=True)

    if submit_btn and user_query.strip():
        process_user_query(user_query.strip(), lang_hint=lang_code, is_voice=False)
        st.rerun()

    # 4. Display Active Response Card
    current_resp: Optional[ChatResponse] = st.session_state.get("current_response")

    if current_resp is not None:
        st.markdown("### 💬 GujjuVaani Response / ઉત્તર")

        # If the question was asked via voice or demo sample, display the spoken question audio player!
        q_audio = st.session_state.get("current_question_audio")
        if q_audio:
            st.markdown(
                f"""
                <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-left: 4px solid #2563EB; border-radius: 10px; padding: 12px 18px; margin-bottom: 14px;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <strong style="color: #1E40AF;">🎙️ Spoken Question Audio ({current_resp.language_name}):</strong>
                        <span style="color: #64748B; font-size: 0.84rem;">"{current_resp.question}"</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.audio(q_audio)

        card_class = "chat-card-bot" if current_resp.answered else "chat-card-user"

        with st.container():
            sim_pct = f"{current_resp.similarity:.3f}"
            status_badge = (
                f'<span class="badge-pill badge-green">✓ Relevant Match ({sim_pct})</span>'
                if current_resp.answered
                else f'<span class="badge-pill badge-amber">⚠️ Out-of-Domain Fallback ({sim_pct})</span>'
            )
            cat_badge = (
                f'<span class="badge-pill badge-amber">📁 {current_resp.category}</span>'
                if current_resp.category
                else ""
            )
            lang_badge = f'<span class="badge-pill badge-blue">🗣️ {current_resp.language_name}</span>'

            st.markdown(
                f"""
                <div class="chat-card {card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <strong style="color: #0F172A; font-size: 1.2rem;">{current_resp.title or 'GujjuVaani Response'}</strong>
                            <div style="margin-top: 4px;">{status_badge}{cat_badge}{lang_badge}</div>
                        </div>
                    </div>
                    <div class="gujarati-answer-text">
                        {current_resp.answer_gu}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Source attribution
            if current_resp.source:
                st.markdown(
                    f"""
                    <div class="source-box">
                        <span>📚 <strong>સ્ત્રોત / Source:</strong> {current_resp.source} ({current_resp.source_type or 'Official'})</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Audio & Translation Actions
            if current_resp.answered:
                st.markdown("#### 🎧 Audio & Translations / ઑડિયો અને ભાષાંતર:")
                col_tts, col_tr_en, col_tr_hi, col_tr_mr = st.columns(4)

                # 1. Listen in Gujarati
                with col_tts:
                    if st.button("🔊 સાંભળો (Listen Gujarati)", key="btn_tts_gu", use_container_width=True):
                        with st.spinner("Generating Gujarati neural audio..."):
                            try:
                                cache_key = f"tts_gu_{current_resp.record_id or current_resp.question}"
                                if cache_key not in st.session_state["tts_audio_cache"]:
                                    res = synthesize(current_resp.answer_gu, language="gu")
                                    st.session_state["tts_audio_cache"][cache_key] = res
                                audio_res = st.session_state["tts_audio_cache"][cache_key]
                                st.audio(audio_res.audio, format=audio_res.mime_type)
                                st.caption(f"✓ Voice: {audio_res.voice}")
                            except Exception as e:
                                st.warning(f"Audio notice: {e}")

                # 2. English Translation
                with col_tr_en:
                    if st.button("🇬🇧 English Translation", key="btn_tr_en", use_container_width=True):
                        with st.spinner("Translating to English..."):
                            try:
                                if current_resp.curated_answer_en:
                                    en_text = current_resp.curated_answer_en
                                    backend_used = "Curated Knowledge Base"
                                else:
                                    en_text, backend_used = translate(current_resp.answer_gu, target="en")
                                st.session_state["translation_view"] = {
                                    "lang": "English",
                                    "code": "en",
                                    "text": en_text,
                                    "backend": backend_used,
                                }
                            except Exception as e:
                                st.error(f"Translation notice: {e}")

                # 3. Hindi Translation
                with col_tr_hi:
                    if st.button("🇮🇳 हिंदी अनुवाद (Hindi)", key="btn_tr_hi", use_container_width=True):
                        with st.spinner("Translating to Hindi..."):
                            try:
                                if current_resp.curated_answer_hi:
                                    hi_text = current_resp.curated_answer_hi
                                    backend_used = "Curated Knowledge Base"
                                else:
                                    hi_text, backend_used = translate(current_resp.answer_gu, target="hi")
                                st.session_state["translation_view"] = {
                                    "lang": "Hindi (हिन्दी)",
                                    "code": "hi",
                                    "text": hi_text,
                                    "backend": backend_used,
                                }
                            except Exception as e:
                                st.error(f"Translation notice: {e}")

                # 4. Marathi Translation
                with col_tr_mr:
                    if st.button("🇮🇳 मराठी भाषांतर (Marathi)", key="btn_tr_mr", use_container_width=True):
                        with st.spinner("Translating to Marathi..."):
                            try:
                                mr_text, backend_used = translate(current_resp.answer_gu, target="mr")
                                st.session_state["translation_view"] = {
                                    "lang": "Marathi (मराठी)",
                                    "code": "mr",
                                    "text": mr_text,
                                    "backend": backend_used,
                                }
                            except Exception as e:
                                st.error(f"Translation notice: {e}")

                # Render Active Translation View
                tr_view = st.session_state.get("translation_view")
                if tr_view:
                    st.markdown(
                        f"""
                        <div style="background: #FFFFFF; border: 1px solid #BFDBFE; border-left: 4px solid #2563EB; border-radius: 10px; padding: 14px 18px; margin-top: 14px; box-shadow: var(--shadow-sm);">
                            <strong style="color: #1E40AF;">🌐 {tr_view['lang']} Translation</strong>
                            <span style="color: #64748B; font-size: 0.8rem; margin-left: 8px;">(Engine: {tr_view['backend']})</span>
                            <div style="color: #0F172A; margin-top: 8px; font-size: 1.05rem; line-height: 1.65;">
                                {tr_view['text']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button(f"🔊 Listen in {tr_view['lang']}", key="btn_tts_tr"):
                        with st.spinner(f"Generating {tr_view['lang']} voice..."):
                            try:
                                tr_audio = synthesize(tr_view["text"], language=tr_view["code"])
                                st.audio(tr_audio.audio, format=tr_audio.mime_type)
                            except Exception as e:
                                st.warning(f"Voice playback: {e}")

                # Candidate Inspector Accordion
                with st.expander("🔍 Top-3 Retrieved Candidates & Similarity Scores (IR Matrix)"):
                    for c in current_resp.candidates[:3]:
                        st.markdown(
                            f"- **#{c.get('rank', 1)}: {c.get('title')}** `[ID: {c.get('id')}]` • Category: `{c.get('category')}` • Score: **`{c.get('score'):.4f}`**"
                        )

    # 5. Conversation History
    if len(st.session_state["messages"]) > 2:
        st.markdown("---")
        st.markdown("### 📜 Conversation History / અગાઉની વાતચીત")
        for m in reversed(st.session_state["messages"][:-2]):
            if m["role"] == "user":
                st.markdown(
                    f"""
                    <div style="background: #F8FAFC; border-left: 3px solid #2563EB; padding: 10px 16px; border-radius: 8px; margin-bottom: 8px; border: 1px solid #E2E8F0;">
                        <span style="color: #2563EB; font-weight: 600;">You ({m['time']}):</span> {m['text']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if m.get("audio"):
                    st.audio(m["audio"])
            else:
                resp = m.get("response")
                ans = resp.answer_gu if resp else "Response"
                title = resp.title if resp else "GujjuVaani"
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border-left: 3px solid #059669; padding: 10px 16px; border-radius: 8px; margin-bottom: 12px; border: 1px solid #E2E8F0;">
                        <span style="color: #059669; font-weight: 600;">GujjuVaani ({title}):</span>
                        <div style="color: #0F172A; margin-top: 4px;">{ans}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ===========================================================================
# TAB 2: SAMPLE QUESTIONS IN ENGLISH (CLICK TO ASK)
# ===========================================================================
with tab_questions:
    st.markdown("### 💡 Sample Questions in English & Gujarati (Click to Ask)")
    st.caption("Click any question below to immediately ask the chatbot and get authentic cultural answers:")

    CATEGORIZED_QUESTIONS = [
        {
            "category": "🏛️ Historical Monuments & Architecture",
            "questions": [
                ("Tell me about Rani ki Vav in Patan and who built it.", "રાણીની વાવ"),
                ("What is the history and architecture of the Sun Temple in Modhera?", "મોઢેરા સૂર્ય મંદિર"),
                ("Tell me about Champaner-Pavagadh UNESCO World Heritage Site.", "ચાંપાનેર"),
                ("Explain the structure of Adalaj Stepwell near Gandhinagar.", "અડાલજની વાવ"),
                ("What was the ancient Harappan port city of Lothal famous for?", "લોથલ"),
                ("Tell me about Dholavira and its Indus Valley water management system.", "ધોલાવીરા"),
                ("What is the history of Sidi Saiyyed Mosque and its famous stone jali?", "સિદી સૈયદની જાળી"),
                ("Tell me about Somnath Temple and its historical significance.", "સોમનાથ મંદિર"),
            ],
        },
        {
            "category": "🎭 Festivals, Dance & Folk Culture",
            "questions": [
                ("What is the cultural significance of Navratri and Garba in Gujarat?", "નવરાત્રી અને ગરબા"),
                ("How is the International Kite Festival (Uttarayan) celebrated in Gujarat?", "ઉત્તરાયણ"),
                ("Tell me about the Tarnetar Fair, its folk dances and embroidered umbrellas.", "તરણેતરનો મેળો"),
                ("What happens at the Rann Utsav in Kutch during full moon nights?", "રણ ઉત્સવ"),
                ("Tell me about Bhavnath Mahadev Fair at the foothills of Mount Girnar.", "ભવનાથ મેળો"),
                ("Explain the tradition of Bhavai folk theatre in Gujarat.", "ભવાઈ લોકનાટ્ય"),
            ],
        },
        {
            "category": "🦁 Wildlife, Sanctuaries & Geography",
            "questions": [
                ("Where can I see Asiatic lions in Gir National Park?", "ગીર રાષ્ટ્રીય ઉદ્યાન"),
                ("Tell me about the Great Rann of Kutch and the White Desert.", "કચ્છનું મોટું રણ"),
                ("What migratory birds visit Nal Sarovar Bird Sanctuary?", "નળ સરોવર પક્ષી અભયારણ્ય"),
                ("What is unique about the Marine National Park in the Gulf of Kutch?", "મરીન નેશનલ પાર્ક"),
                ("Tell me about Saputara hill station in the Dang district.", "સાપુતારા"),
                ("Explain the economic and cultural importance of the Sabarmati River.", "સાબરમતી નદી"),
            ],
        },
        {
            "category": "🍲 Gujarati Cuisine & Traditional Food",
            "questions": [
                ("What is Undhiyu and how is it traditionally prepared during winter?", "ઊંધિયું"),
                ("What are the key items served in an authentic Gujarati Thali?", "ગુજરાતી થાળી"),
                ("Tell me about Surat's famous Ghari sweet and Locho snack.", "સુરતનું ખાણું"),
                ("What is Khaman Dhokla and what is its origin?", "ખમણ ઢોકળા"),
                ("Tell me about Kutchi Dabeli and how it was invented in Mandvi.", "કચ્છી દાબેલી"),
            ],
        },
        {
            "category": "📜 Great Personalities, Literature & Art",
            "questions": [
                ("Who was Bhakta Kavi Narsinh Mehta and what are his famous bhajans?", "નરસિંહ મહેતા"),
                ("Tell me about Sardar Vallabhbhai Patel and the Statue of Unity.", "સરદાર પટેલ"),
                ("What was Mahatma Gandhi's role in the Sabarmati Ashram and Dandi March?", "મહાત્મા ગાંધી"),
                ("Who was the Jain polymath Hemchandracharya (Kalikal Sarvagya)?", "હેમચંદ્રાચાર્ય"),
                ("Tell me about Jhaverchand Meghani, the National Poet of Saurashtra.", "ઝવેરચંદ મેઘાણી"),
                ("What is special about the double-ikat Patan Patola silk weaving art?", "પાટણના પટોળા"),
            ],
        },
    ]

    for cat_data in CATEGORIZED_QUESTIONS:
        st.markdown(f"#### {cat_data['category']}")
        q_cols = st.columns(2)
        for q_idx, (eng_q, label_gu) in enumerate(cat_data["questions"]):
            with q_cols[q_idx % 2]:
                if st.button(f"👉 {eng_q}", key=f"lib_q_{cat_data['category'][:4]}_{q_idx}", use_container_width=True):
                    st.session_state["selected_sample_query"] = eng_q
                    process_user_query(eng_q, lang_hint=lang_code)
                    st.rerun()
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


# ===========================================================================
# TAB 3: HERITAGE KNOWLEDGE EXPLORER
# ===========================================================================
with tab_explorer:
    st.markdown("### 🏛️ Gujarat Heritage Knowledge Explorer / વારસો દર્શન")
    st.caption("Explore all 140+ verified topics in the GujjuVaani curated knowledge base.")

    all_records = load_all_records()
    categories = sorted(list({r.get("category", "General") for r in all_records}))

    col_cat, col_srch = st.columns([0.4, 0.6])
    with col_cat:
        chosen_cat = st.selectbox("શ્રેણી પસંદ કરો / Filter Category", ["All Categories"] + categories)
    with col_srch:
        search_kw = st.text_input("વિષય કે કીવર્ડ શોધો / Search topics:", placeholder="e.g. Somnath, Garba, Patan...")

    filtered = all_records
    if chosen_cat != "All Categories":
        filtered = [r for r in filtered if r.get("category") == chosen_cat]
    if search_kw.strip():
        kw_low = search_kw.strip().lower()
        filtered = [
            r
            for r in filtered
            if kw_low in r.get("title", "").lower()
            or kw_low in r.get("title_en", "").lower()
            or kw_low in r.get("answer_gu", "").lower()
            or any(kw_low in str(k).lower() for k in r.get("keywords", []))
        ]

    st.markdown(f"**Showing {len(filtered)} of {len(all_records)} heritage records:**")

    for r in filtered:
        with st.expander(f"📍 {r.get('title')} ({r.get('category')} - {r.get('district', 'Gujarat')})"):
            st.markdown("**ગુજરાતી ઉત્તર (Natural Gujarati Answer):**")
            st.markdown(f"> {r.get('answer_gu')}")

            c1, c2 = st.columns(2)
            with c1:
                if r.get("short_answer_en"):
                    st.markdown(f"🇬🇧 **English Summary:** {r.get('short_answer_en')}")
                st.markdown(f"🏷️ **Keywords:** `{', '.join(r.get('keywords', []))}`")
            with c2:
                if r.get("short_answer_hi"):
                    st.markdown(f"🇮🇳 **Hindi Summary:** {r.get('short_answer_hi')}")
                st.markdown(f"📚 **Source:** {r.get('source')} `[{r.get('source_type')}]`")


# ===========================================================================
# TAB 4: NLP PIPELINE INSPECTOR
# ===========================================================================
with tab_pipeline:
    st.markdown("### 🔬 GujjuVaani Multilingual NLP Pipeline Inspector")
    st.caption("Live step-by-step diagnostic view of how a query is processed through the NLP architecture.")

    pipe_query = st.text_input(
        "Enter a query to inspect / પ્રશ્ન દાખલ કરો:",
        value="Rani ki Vav kyare banavvama aavi hati?",
        key="pipe_query_in",
    )

    if pipe_query.strip():
        prof = script_profile(pipe_query)
        detected_l = detect_language(pipe_query)
        expanded_q = expand_query(pipe_query, detected_l)

        chatbot = load_cached_chatbot()
        hits = chatbot.retriever.search(pipe_query, top_k=5, language=detected_l)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 1️⃣ Language Identification (LID)")
            st.write(f"- **Detected Language Code:** `{detected_l}` ({LANGUAGE_NAMES.get(detected_l, 'Unknown')})")
            st.write(
                f"- **Script Profile:** Gujarati `{prof['gujarati_raw']}`, Devanagari `{prof['devanagari_raw']}`, Latin `{prof['latin_raw']}`"
            )

            st.markdown("#### 2️⃣ Normalization & Query Expansion")
            st.write(f"- **Raw Input:** `{pipe_query}`")
            st.write(f"- **Expanded Query (E5 Input):** `{expanded_q}`")

        with c2:
            st.markdown("#### 3️⃣ Multilingual Dense Embedding")
            st.write(f"- **Model:** `intfloat/multilingual-e5-small`")
            st.write(f"- **E5 Prefix Format:** `query: {expanded_q}`")
            st.write(f"- **Vector Space:** 384-dimensional unit-norm hypersphere")

            st.markdown("#### 4️⃣ Vector Search & Relevance Gating")
            st.write(f"- **Configured Threshold:** `{config.RELEVANCE_THRESHOLD:.2f}`")
            st.write(f"- **Hard Reject Threshold:** `{config.HARD_REJECT_THRESHOLD:.2f}`")

        st.markdown("#### 5️⃣ Top Retrieval Candidate Vectors:")
        hit_rows = []
        for h in hits:
            hit_rows.append(
                {
                    "Rank": h.rank,
                    "Topic Title": h.title,
                    "Category": h.category,
                    "Hybrid Score": round(h.score, 4),
                    "Status": "✅ Pass (Relevant)" if h.score >= config.RELEVANCE_THRESHOLD else "❌ Reject",
                }
            )
        st.dataframe(pd.DataFrame(hit_rows), use_container_width=True)


# ===========================================================================
# TAB 5: BENCHMARK & DATABASE INFO (MOVED FROM SIDEBAR)
# ===========================================================================
with tab_eval_system:
    st.markdown("### 📊 Live Benchmark & Database Architecture / મૂલ્યાંકન અને સિસ્ટમ ડેટા")
    
    # 1. Database & Model Architecture Overview
    st.markdown("#### 🗄️ Database & Model Architecture")
    try:
        bot = load_cached_chatbot()
        stats = bot.retriever.stats()
        
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        with d_col1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-val">{stats['records']}</div>
                    <div class="metric-lbl">🏛️ Verified Heritage Topics</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with d_col2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-val">{stats['embedding_dim']}-d</div>
                    <div class="metric-lbl">🧠 Vector Dimension (E5)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with d_col3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-val">FAISS / IP</div>
                    <div class="metric-lbl">⚡ Index Search Backend</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with d_col4:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-val">{config.RELEVANCE_THRESHOLD:.2f}</div>
                    <div class="metric-lbl">🎯 Relevance Rejection Gate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.write(f"- **Embedding Model:** `intfloat/multilingual-e5-small` (384-dimensional dense semantic vectors)")
        st.write(f"- **Speech-to-Text Engine:** `faster-whisper ({config.WHISPER_MODEL})` (CTranslate2 int8 quantised offline CPU inference)")
        st.write(f"- **Speech Synthesis (TTS):** Microsoft Edge Neural Voices (`gu-IN-DhwaniNeural`, `en-IN-NeerjaNeural`, `hi-IN-SwaraNeural`) + gTTS fallback")
        st.write(f"- **Machine Translation:** NLLB-200 / AI4Bharat IndicTrans2 + Curated Bilingual Knowledge Base")
    except Exception as e:
        st.warning(f"System statistics: {e}")

    st.markdown("---")
    
    # 2. Live Benchmark Suite
    st.markdown("#### 📈 Live Evaluation Benchmark")
    st.caption("Run real evaluations over the test dataset (35 multilingual test cases + out-of-domain rejection probes).")

    if st.button("🚀 Run Live NLP Benchmark / મૂલ્યાંકન શરૂ કરો", type="primary", use_container_width=True):
        with st.spinner("Running evaluation suite over multilingual test dataset..."):
            chatbot = load_cached_chatbot()
            questions = load_evaluation_questions()
            report: EvalReport = evaluate(questions, chatbot=chatbot)
            st.session_state["eval_report"] = report

    report: Optional[EvalReport] = st.session_state.get("eval_report")

    if report is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-val">{report.top1:.1f}%</div>
                    <div class="metric-lbl">Top-1 Accuracy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-val">{report.top3:.1f}%</div>
                    <div class="metric-lbl">Top-3 Accuracy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-val">{report.ood_rejection:.1f}%</div>
                    <div class="metric-lbl">Out-of-Domain Rejection</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-val">{report.avg_similarity:.3f}</div>
                    <div class="metric-lbl">Avg Cosine Similarity</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("##### 🌐 Per-Language Accuracy Breakdown:")
        lang_data = []
        for l, stats in report.by_language().items():
            lang_data.append(
                {
                    "Language": LANGUAGE_NAMES.get(l, l),
                    "Code": l,
                    "Test Samples": stats["n"],
                    "Top-1 Accuracy": f"{stats['top1']:.1f}%",
                    "Top-3 Accuracy": f"{stats['top3']:.1f}%",
                    "Avg Similarity": f"{stats['avg_similarity']:.4f}",
                }
            )
        st.dataframe(pd.DataFrame(lang_data), use_container_width=True)

        with st.expander("🔍 Detailed Per-Question Evaluation Matrix"):
            df_rows = [
                {
                    "Language": r.language,
                    "Question": r.question,
                    "Expected Topic": r.expected,
                    "Top-1 Retrieved": r.top1_title,
                    "Similarity": round(r.similarity, 4),
                    "Top-1 Match": "✅" if r.top1_hit else "❌",
                    "Top-3 Match": "✅" if r.top3_hit else "❌",
                }
                for r in report.in_domain
            ]
            st.dataframe(pd.DataFrame(df_rows), use_container_width=True)
    else:
        st.info("Click **'Run Live NLP Benchmark'** above to compute real Top-1/Top-3 accuracy and out-of-domain rejection statistics.")