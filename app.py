"""
Streamlit demo UI for the fake news detector.
Accepts input as pasted text, a URL, or a screenshot (OCR).

Run with:
    streamlit run app.py

Make sure you've already run `python src/train.py` first so models/ exists.
"""

import sys
from pathlib import Path

import joblib
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from preprocessing import clean_text  # noqa: E402
from extract import extract_text_from_url, extract_text_from_image  # noqa: E402

MODELS_DIR = Path(__file__).parent / "models"

MODEL_FILES = {
    "Naive Bayes": "naive_bayes_model.joblib",
    "SVM (Linear, calibrated)": "svm_model.joblib",
    "Logistic Regression": "logistic_regression_model.joblib",
}


@st.cache_resource
def load_vectorizer():
    return joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")


@st.cache_resource
def load_model(filename: str):
    return joblib.load(MODELS_DIR / filename)


def classify(text: str, model_choice: str):
    cleaned = clean_text(text)
    if not cleaned:
        st.warning("No usable content left after cleaning - try a longer/different input.")
        return

    vectorizer = load_vectorizer()
    model = load_model(MODEL_FILES[model_choice])

    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = proba[pred] * 100
    label = "REAL" if pred == 1 else "FAKE"

    if label == "REAL":
        st.success(f"Prediction: **{label}**")
    else:
        st.error(f"Prediction: **{label}**")

    st.metric("Confidence", f"{confidence:.2f}%")
    st.progress(min(int(confidence), 100))

    with st.expander("Class probabilities"):
        st.write({"fake": f"{proba[0]*100:.2f}%", "real": f"{proba[1]*100:.2f}%"})

    with st.expander("Extracted text used for classification"):
        st.write(text[:2000] + ("..." if len(text) > 2000 else ""))


def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="📰")
    st.title("📰 Fake News Detector")
    st.caption("TF-IDF + classical ML (Naive Bayes / SVM / Logistic Regression) — trained on the ISOT dataset")

    if not (MODELS_DIR / "tfidf_vectorizer.joblib").exists():
        st.error("No trained models found. Run `python src/train.py` first, then reload this page.")
        return

    model_choice = st.selectbox("Model", list(MODEL_FILES.keys()))

    tab_text, tab_url, tab_image = st.tabs(["✍️ Paste text", "🔗 URL", "🖼️ Screenshot"])

    with tab_text:
        text = st.text_area("Paste a news headline or article", height=200, key="text_input")
        if st.button("Classify", type="primary", key="classify_text"):
            if not text.strip():
                st.warning("Please paste some text first.")
            else:
                classify(text, model_choice)

    with tab_url:
        url = st.text_input("Article URL", placeholder="https://example.com/some-article")
        if st.button("Fetch & Classify", type="primary", key="classify_url"):
            if not url.strip():
                st.warning("Please enter a URL first.")
            else:
                with st.spinner("Fetching article..."):
                    try:
                        extracted = extract_text_from_url(url)
                    except Exception as e:
                        st.error(f"Couldn't fetch/parse that URL: {e}")
                        extracted = ""
                if extracted:
                    if len(extracted) < 100:
                        st.warning("Very little text was extracted - this site may block scraping or use JS rendering. Try pasting the text manually instead.")
                    classify(extracted, model_choice)

    with tab_image:
        st.caption("Requires the Tesseract OCR engine installed on this machine - see README.")
        uploaded = st.file_uploader("Upload a screenshot", type=["png", "jpg", "jpeg"])
        if uploaded and st.button("Extract & Classify", type="primary", key="classify_image"):
            with st.spinner("Running OCR..."):
                try:
                    extracted = extract_text_from_image(uploaded)
                except Exception as e:
                    st.error(
                        f"OCR failed: {e}\n\n"
                        "Make sure Tesseract OCR is installed and on your PATH (see README)."
                    )
                    extracted = ""
            if extracted:
                if len(extracted) < 30:
                    st.warning("Very little text was extracted from the image - try a clearer/higher-res screenshot.")
                classify(extracted, model_choice)

    st.divider()
    st.caption(
        "Note: this model recognizes writing-style/vocabulary patterns from its training "
        "data - it does not fact-check. Short, unusual, or poorly-extracted inputs may get "
        "low-confidence predictions, which is expected."
    )


if __name__ == "__main__":
    main()
