"""Voice Demo — speak and see your words transcribed using Whisper (offline).

Uses `audio_recorder_streamlit` for a one-click record button in the browser,
then transcribes with OpenAI Whisper running locally (no API/internet needed).

Install deps:
    pip install streamlit openai-whisper audio-recorder-streamlit torch

Run:
    streamlit run app/voice_demo.py

How it works:
    1. Click the microphone button in the browser
    2. Speak (in English or Nepali)
    3. Click stop
    4. Whisper transcribes locally (no internet needed)
    5. Text appears on screen
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Voice Demo", page_icon="🎤", layout="centered")
st.title("🎤 Voice to Text Demo (Whisper, offline)")
st.caption("Speak into your microphone. Whisper transcribes locally — no internet needed.")

# ---------------------------------------------------------------------------
# Model loading (cached so it loads only once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_whisper_model(size: str = "base"):
    """Load Whisper model. Sizes: tiny (~40MB), base (~140MB), small (~460MB)."""
    import whisper
    return whisper.load_model(size)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox("Whisper model size", ["tiny", "base", "small"],
                              index=1,
                              help="Larger = more accurate but slower. "
                                   "'base' is the sweet spot for a demo.")
    language = st.selectbox("Language", ["auto-detect", "en", "ne", "hi"],
                            index=0,
                            help="Set to 'ne' for Nepali, 'hi' for Hindi, "
                                 "or let Whisper auto-detect.")
    st.divider()
    st.markdown(
        "**How to use:**\n"
        "1. Click the microphone button below\n"
        "2. Speak clearly\n"
        "3. Click stop\n"
        "4. Wait for transcription"
    )

# ---------------------------------------------------------------------------
# Audio recorder
# ---------------------------------------------------------------------------
try:
    from audio_recorder_streamlit import audio_recorder
except ImportError:
    st.error(
        "Missing package. Install it:\n\n"
        "```bash\npip install audio-recorder-streamlit\n```"
    )
    st.stop()

st.subheader("Record your voice")
audio_bytes = audio_recorder(
    text="Click to record",
    recording_color="#e74c3c",
    neutral_color="#2ecc71",
    pause_threshold=2.0,
)

# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    with st.spinner("Transcribing with Whisper..."):
        model = load_whisper_model(model_size)

        # Write audio to a temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Transcribe
        lang = None if language == "auto-detect" else language
        result = model.transcribe(tmp_path, language=lang)

        # Clean up
        Path(tmp_path).unlink(missing_ok=True)

    # Display result
    st.divider()
    st.subheader("Transcription")
    st.markdown(f"### \"{result['text'].strip()}\"")

    with st.expander("Details"):
        st.write(f"**Detected language:** {result.get('language', 'unknown')}")
        if result.get("segments"):
            for seg in result["segments"]:
                st.text(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
else:
    st.info("Click the microphone button above, speak, then wait for transcription.")
