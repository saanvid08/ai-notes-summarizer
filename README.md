# 📝 NoteLens — AI Notes Summarizer

NoteLens is a free, student-built computer vision + NLP tool that turns photos of handwritten class notes into structured study guides.

## What it does

**1. Image preprocessing**
- Converts the page to grayscale
- Reduces noise
- Uses adaptive thresholding to improve contrast
- Upscales small handwriting

**2. OCR**
- Uses EasyOCR to detect and recognize handwritten/printed English text
- Sorts detected text into a readable order

**3. NLP summarization**
- Cleans the OCR output
- Scores sentences using word-frequency features
- Produces a concise bullet-point study guide
- Lets students download the result as a `.txt` file

## Tech stack

- Python
- Streamlit
- OpenCV
- EasyOCR / PyTorch
- NumPy
- Pillow
- NLP-based extractive summarization

## Run locally

```bash
git clone YOUR_REPOSITORY_URL
cd ai-notes-summarizer

python -m venv .venv
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The first OCR run may download EasyOCR's model files.

## Project structure

```text
ai-notes-summarizer/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Important project note

This project is designed to be genuinely usable, but OCR quality varies significantly with handwriting, lighting, camera angle, and page layout. Test it on real notes and report actual usage/results rather than inventing performance numbers.

## Future improvements

- Automatic heading detection
- Flashcard generation
- Key-term extraction
- Multi-page note merging
- Handwritten math recognition
- Side-by-side image/text correction
- Local history of previous study guides
