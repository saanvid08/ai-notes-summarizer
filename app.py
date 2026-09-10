import io
import re
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="NoteLens — AI Notes Summarizer",
    page_icon="📝",
    layout="wide",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #eef6ff 0%, #f8fbff 55%, #f1f7ff 100%);
        border: 1px solid #dcecff;
        margin-bottom: 1.5rem;
    }
    .hero h1 {font-size: 3rem; margin-bottom: .3rem;}
    .hero p {font-size: 1.1rem; color: #526173;}
    .card {
        padding: 1.2rem;
        border: 1px solid #e6eaf0;
        border-radius: 18px;
        background: white;
        box-shadow: 0 4px 18px rgba(20,40,70,.05);
    }
    .tag {
        display:inline-block; padding:.3rem .65rem; margin:.15rem;
        border-radius:999px; background:#edf5ff; color:#2865a6;
        font-size:.82rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>📝 NoteLens</h1>
<p>Turn handwritten class notes into clean, structured study guides — locally and for free.</p>
<span class="tag">Computer Vision</span>
<span class="tag">OCR</span>
<span class="tag">NLP</span>
<span class="tag">Python</span>
</div>
""", unsafe_allow_html=True)

# ---------- OCR ----------
@st.cache_resource
def load_ocr():
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception:
        return None

def preprocess(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Adaptive threshold helps with uneven notebook lighting.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )
    return binary

def run_ocr(image: Image.Image):
    reader = load_ocr()
    if reader is None:
        return None, "EasyOCR is not installed. Run: pip install -r requirements.txt"

    processed = preprocess(image)
    # Upscale for small handwriting.
    processed = cv2.resize(processed, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(processed, detail=1, paragraph=False)

    # Sort by approximate reading order.
    results.sort(key=lambda item: (min(p[1] for p in item[0]), min(p[0] for p in item[0])))

    lines = []
    for _, text, confidence in results:
        if confidence >= 0.25 and text.strip():
            lines.append(text.strip())

    return "\n".join(lines), None

# ---------- Extractive NLP summarizer ----------
STOPWORDS = set("""
a an and are as at be been but by for from had has have he her his i if in into is it
its me more most my no not of on or our she so than that the their them then there these
they this those to was we were what when where which who will with you your
""".split())

def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return re.split(r"(?<=[.!?])\s+", text)

def summarize(text, max_sentences=5):
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return sentences

    words = re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower())
    freq = {}
    for word in words:
        if word not in STOPWORDS and len(word) > 2:
            freq[word] = freq.get(word, 0) + 1

    if not freq:
        return sentences[:max_sentences]

    max_freq = max(freq.values())
    for word in freq:
        freq[word] /= max_freq

    scored = []
    for idx, sentence in enumerate(sentences):
        sw = re.findall(r"[A-Za-z][A-Za-z'-]+", sentence.lower())
        score = sum(freq.get(w, 0) for w in sw if w not in STOPWORDS)
        # Slightly reward early sentences, common in lecture-note structure.
        score += max(0, (len(sentences) - idx) / len(sentences)) * 0.15
        scored.append((score, idx, sentence))

    top = sorted(scored, reverse=True)[:max_sentences]
    return [x[2] for x in sorted(top, key=lambda x: x[1])]

def make_bullets(text, max_sentences):
    summary = summarize(text, max_sentences)
    return "\n".join(f"- {s}" for s in summary)

# ---------- UI ----------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("1. Add your notes")
    uploaded = st.file_uploader(
        "Upload a clear photo of handwritten notes",
        type=["png", "jpg", "jpeg"],
        help="Good lighting and a straight-on photo produce better OCR.",
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Original notes", use_container_width=True)

with right:
    st.subheader("2. Generate a study guide")
    sentence_count = st.slider("Summary length", 3, 10, 5)
    process = st.button("✨ Extract & Summarize", use_container_width=True)

    if process:
        if not uploaded:
            st.warning("Upload a note image first.")
        else:
            with st.spinner("Reading handwriting and organizing your notes..."):
                text, error = run_ocr(image)

            if error:
                st.error(error)
            elif not text:
                st.warning("I couldn't confidently read this page. Try a clearer photo.")
            else:
                st.session_state["ocr_text"] = text
                st.session_state["summary"] = make_bullets(text, sentence_count)
                st.success("Notes processed!")

if "ocr_text" in st.session_state:
    st.divider()
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔎 Extracted notes")
        st.text_area("OCR output", st.session_state["ocr_text"], height=330)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📚 Study summary")
        st.markdown(st.session_state["summary"])
        st.download_button(
            "Download study guide",
            data=st.session_state["summary"],
            file_name="study_guide.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.caption("NoteLens is an independent student project. OCR accuracy depends on handwriting, image quality, and page layout.")
