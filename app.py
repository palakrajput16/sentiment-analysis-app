import streamlit as st
from transformers import pipeline
import pandas as pd
import time
import re
import io
from datetime import datetime

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    layout="wide"
)

# ----------------------------------------------------
# CONSTANTS
# ----------------------------------------------------

MAX_CHARS = 3000              # hard input limit for the single-text box
MAX_BATCH_ROWS = 500           # hard limit on rows processed in batch mode
MODEL_MAX_TOKENS = 512          # DistilBERT's hard token limit
UNCERTAIN_THRESHOLD = 60.0     # below this confidence, label as "Uncertain" instead of forcing Positive/Negative

# ----------------------------------------------------
# CSS
# ----------------------------------------------------

st.markdown("""
<style>

/* ---------- Force light backgrounds everywhere, regardless of ---------- */
/* ---------- the visitor's OS/browser dark mode setting          ---------- */

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stMain"],
.stApp {
    background-color: #FAFAF9 !important;
    color: #1C1C1C !important;
    font-family: "Times New Roman", serif !important;
}

[class*="css"] {
    font-family: "Times New Roman", serif;
}

/* ---------- Layout width ---------- */

.block-container {
    max-width: 950px;
    padding-top: 3rem;
    padding-bottom: 4rem;
    background-color: #FAFAF9 !important;
}

/* ---------- Headings ---------- */

h1, h1 * {
    color: #1C1C1C !important;
    font-family: "Times New Roman", serif;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}

h2, h2 *, h3, h3 * {
    color: #1C1C1C !important;
    font-family: "Times New Roman", serif;
    font-weight: 600;
}

/* Subtitle under title */

.subtitle {
    color: #5C5C5C !important;
    font-size: 17px;
    margin-bottom: 2rem;
    max-width: 640px;
}

/* ---------- Divider ---------- */

hr {
    border: none;
    border-top: 1px solid #E0E0DC;
    margin: 2rem 0;
}

/* ---------- Text area ---------- */

.stTextArea textarea {
    font-family: "Times New Roman", serif !important;
    font-size: 17px !important;
    border: 1px solid #D6D3CE !important;
    border-radius: 4px !important;
    background-color: #FFFFFF !important;
    color: #1C1C1C !important;
    -webkit-text-fill-color: #1C1C1C !important;
}

.stTextArea textarea::placeholder {
    color: #9A9A94 !important;
    opacity: 1 !important;
}

.stTextArea textarea:focus {
    border: 1px solid #1C1C1C !important;
    box-shadow: none !important;
}

label, label *,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
[data-testid="stWidgetLabel"] p {
    font-family: "Times New Roman", serif !important;
    color: #1C1C1C !important;
    font-size: 15px !important;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

/* ---------- Buttons ---------- */

.stButton>button {
    background-color: #1C1C1C !important;
    color: #FAFAF9 !important;
    width: 100%;
    height: 50px;
    border: none;
    border-radius: 4px;
    font-size: 16px;
    font-family: "Times New Roman", serif;
    letter-spacing: 0.5px;
    transition: background 0.15s ease;
}

.stButton>button p, .stButton>button div, .stButton>button span {
    color: #FAFAF9 !important;
    font-family: "Times New Roman", serif !important;
}

.stButton>button:hover {
    background: #3A3A3A !important;
    color: #FAFAF9 !important;
}

.stButton>button:hover p, .stButton>button:hover div, .stButton>button:hover span {
    color: #FAFAF9 !important;
}

/* ---------- Secondary / download buttons ---------- */

.stDownloadButton>button {
    background-color: #FFFFFF !important;
    color: #1C1C1C !important;
    border: 1px solid #1C1C1C !important;
    width: 100%;
    height: 46px;
    border-radius: 4px;
    font-family: "Times New Roman", serif !important;
}

.stDownloadButton>button p, .stDownloadButton>button div, .stDownloadButton>button span {
    color: #1C1C1C !important;
}

.stDownloadButton>button:hover {
    background-color: #F1F0EC !important;
}

/* ---------- Result card ---------- */

.result-card {
    border: 1px solid #E0E0DC;
    border-radius: 6px;
    padding: 28px 32px;
    background-color: #FFFFFF;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}

.result-label {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8A8A85 !important;
    margin-bottom: 6px;
}

.result-value {
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 4px;
    color: #1C1C1C !important;
}

.result-value.positive {
    border-left: 3px solid #1C1C1C;
    padding-left: 14px;
}

.result-value.negative {
    border-left: 3px solid #8A8A85;
    padding-left: 14px;
}

.result-value.uncertain {
    border-left: 3px solid #C7A24A;
    padding-left: 14px;
}

.result-note {
    font-size: 14px;
    color: #8A8A85 !important;
    margin-top: 10px;
}

/* ---------- Progress bar ---------- */

.stProgress > div > div {
    background-color: #1C1C1C !important;
}

.stProgress > div {
    background-color: #E5E5E0 !important;
}

/* ---------- Metric cards ---------- */

[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #E0E0DC;
    padding: 18px 20px;
    border-radius: 6px;
}

[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    font-family: "Times New Roman", serif !important;
    color: #5C5C5C !important;
    text-transform: uppercase;
    font-size: 13px !important;
    letter-spacing: 0.5px;
}

[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    font-family: "Times New Roman", serif !important;
    color: #1C1C1C !important;
}

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"] {
    background: #F1F0EC !important;
    border-right: 1px solid #E0E0DC;
}

section[data-testid="stSidebar"] * {
    font-family: "Times New Roman", serif !important;
    color: #1C1C1C !important;
}

section[data-testid="stSidebar"] hr {
    border-top: 1px solid #D6D3CE;
}

/* ---------- Tabs ---------- */

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #E0E0DC;
}

.stTabs [data-baseweb="tab"] {
    font-family: "Times New Roman", serif !important;
    color: #5C5C5C !important;
    font-size: 16px;
    padding: 10px 18px;
}

.stTabs [aria-selected="true"] {
    color: #1C1C1C !important;
    border-bottom: 2px solid #1C1C1C !important;
    font-weight: 600;
}

/* ---------- Dataframes / tables ---------- */

[data-testid="stDataFrame"] {
    font-family: "Times New Roman", serif !important;
    border: 1px solid #E0E0DC !important;
}

/* ---------- Code blocks (example inputs) ---------- */

.stCodeBlock, pre, .stCodeBlock > div {
    font-family: "Times New Roman", serif !important;
    background-color: #FFFFFF !important;
    border: 1px solid #E0E0DC !important;
    border-radius: 4px !important;
}

.stCodeBlock code, pre code, .stCodeBlock span {
    font-family: "Times New Roman", serif !important;
    color: #1C1C1C !important;
    background-color: transparent !important;
}

/* ---------- Alerts (warning/error/info boxes) ---------- */

.stAlert, .stAlert * {
    font-family: "Times New Roman", serif !important;
    border-radius: 4px;
    color: #1C1C1C !important;
}

.stAlert {
    background-color: #F1F0EC !important;
    border: 1px solid #D6D3CE !important;
}

/* ---------- File uploader ---------- */

[data-testid="stFileUploader"] section {
    background-color: #FFFFFF !important;
    border: 1px dashed #D6D3CE !important;
    border-radius: 4px;
}

[data-testid="stFileUploader"] * {
    color: #1C1C1C !important;
    font-family: "Times New Roman", serif !important;
}

/* ---------- Paragraph / markdown text ---------- */

p {
    font-size: 17px;
    color: #2E2E2E !important;
}

[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * {
    color: #2E2E2E !important;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

# ----------------------------------------------------
# SESSION STATE
# ----------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# ----------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------

def guess_text_column(df):
    """Best-effort guess at which column holds the text to analyze,
    so the selector doesn't just default to column 0 (often an ID
    column) and silently produce meaningless results."""
    keywords = ["text", "review", "comment", "sentence", "content", "feedback", "message", "description"]

    for col in df.columns:
        if any(k in str(col).lower() for k in keywords):
            return col

    object_cols = [c for c in df.columns if df[c].dtype == object]
    if object_cols:
        avg_lengths = {c: df[c].astype(str).str.len().mean() for c in object_cols}
        return max(avg_lengths, key=avg_lengths.get)

    return df.columns[0]


def has_meaningful_content(text):
    """True if the text contains at least one letter or digit.
    Filters out inputs that are technically non-empty but carry no
    real content (e.g. '...', '???', a lone comma), which the model
    would otherwise score with a misleadingly confident prediction."""
    return bool(re.search(r'[A-Za-z0-9]', text))


def split_sentences(text):
    """Naive sentence splitter. Good enough for short-to-medium text
    without pulling in a full NLP sentence tokenizer dependency."""
    pieces = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in pieces if p.strip()]


def is_truncated(text):
    """Check whether the model's tokenizer would cut this text off
    before it ever reaches the classifier, so we can warn the user
    instead of silently scoring a partial sentence."""
    try:
        tokens = classifier.tokenizer.encode(text, add_special_tokens=True)
        return len(tokens) > MODEL_MAX_TOKENS
    except Exception:
        return False


def classify_text(text):
    """Run the classifier on a single string and return a normalized
    result dict. Applies the uncertain-confidence band and truncation
    handling, and never lets a raw exception hit the UI."""
    try:
        truncated = is_truncated(text)
        raw = classifier(text, truncation=True, max_length=MODEL_MAX_TOKENS)[0]
        confidence = raw["score"] * 100

        if confidence < UNCERTAIN_THRESHOLD:
            label = "UNCERTAIN"
        else:
            label = raw["label"]

        return {
            "success": True,
            "label": label,
            "raw_label": raw["label"],
            "confidence": confidence,
            "truncated": truncated,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "label": None,
            "raw_label": None,
            "confidence": None,
            "truncated": False,
            "error": str(e)
        }


def label_display(label):
    return {"POSITIVE": "Positive", "NEGATIVE": "Negative", "UNCERTAIN": "Uncertain"}.get(label, label)


def label_css_class(label):
    return {"POSITIVE": "positive", "NEGATIVE": "negative", "UNCERTAIN": "uncertain"}.get(label, "")


def add_to_history(text, result):
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Text": text if len(text) <= 80 else text[:77] + "...",
        "Prediction": label_display(result["label"]),
        "Confidence": f"{result['confidence']:.2f}%"
    })


# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

st.sidebar.title("Project Information")

st.sidebar.write("""
This application performs sentiment analysis
using a pretrained DistilBERT model from
Hugging Face Transformers.

The model classifies text as Positive or
Negative and provides a confidence score.
Predictions below the confidence threshold
are labeled Uncertain rather than forced
into a category.
""")

st.sidebar.divider()

st.sidebar.subheader("Technology Stack")

st.sidebar.write("""
Python

Streamlit

Hugging Face

Transformers

PyTorch

Pandas
""")

st.sidebar.divider()

st.sidebar.subheader("Model")

st.sidebar.write("distilbert-base-uncased-finetuned-sst-2-english")

st.sidebar.divider()

st.sidebar.subheader("Limitations")

st.sidebar.write(f"""
Trained on movie reviews (SST-2), so it can
misread sarcasm, mixed sentiment, or text
from very different domains (e.g. financial
or medical writing).

Inputs longer than {MODEL_MAX_TOKENS} tokens
are truncated before scoring.

Predictions under {UNCERTAIN_THRESHOLD:.0f}%
confidence are shown as Uncertain.

Text with no letters or numbers (e.g. blank
input or stray punctuation) is treated as
Invalid rather than scored.
""")

# ----------------------------------------------------
# TITLE
# ----------------------------------------------------

st.title("AI Sentiment Analyzer")

st.markdown(
    '<p class="subtitle">Analyze the sentiment of text using a pretrained '
    'DistilBERT transformer model. Run a single sentence, break a paragraph '
    'down sentence by sentence, or upload a CSV to process text in bulk.</p>',
    unsafe_allow_html=True
)

# ----------------------------------------------------
# TABS
# ----------------------------------------------------

tab_single, tab_batch, tab_history = st.tabs(["Single Text", "Batch Upload", "History"])

# ----------------------------------------------------
# TAB 1: SINGLE TEXT
# ----------------------------------------------------

with tab_single:

    text = st.text_area(
        "Input Text",
        placeholder="Example: I absolutely loved this movie.",
        height=160,
        max_chars=MAX_CHARS
    )

    st.caption(f"{len(text)} / {MAX_CHARS} characters")

    breakdown_mode = st.checkbox("Break down sentence by sentence")

    if st.button("Analyze", key="analyze_single"):

        if text.strip() == "":

            st.warning("Please enter text.")

        elif not has_meaningful_content(text):

            st.warning("This doesn't look like actual text — please enter a real word or sentence.")

        elif not breakdown_mode:

            start = time.time()
            result = classify_text(text)
            end = time.time()

            if not result["success"]:
                st.error(f"Something went wrong while analyzing this text: {result['error']}")
            else:
                result_class = label_css_class(result["label"])
                display_label = label_display(result["label"])

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Prediction</div>
                        <div class="result-value {result_class}">{display_label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(min(result["confidence"] / 100, 1.0))

                c1, c2 = st.columns(2)

                with c1:
                    st.metric("Confidence", f"{result['confidence']:.2f}%")

                with c2:
                    st.metric("Inference Time", f"{end - start:.3f} sec")

                if result["truncated"]:
                    st.warning(
                        f"This text is longer than the model's {MODEL_MAX_TOKENS}-token limit. "
                        "Only the first portion was actually scored."
                    )

                add_to_history(text, result)

                st.divider()
                st.subheader("Model")
                st.write("DistilBERT (Hugging Face Transformers)")

        else:
            # Sentence-by-sentence breakdown
            sentences = split_sentences(text)

            if len(sentences) == 0:
                st.warning("Couldn't find any sentences to analyze.")
            else:
                rows = []
                any_truncated = False
                start = time.time()

                for s in sentences:
                    result = classify_text(s)
                    if not result["success"]:
                        rows.append({
                            "Sentence": s,
                            "Prediction": "Error",
                            "Confidence": "-"
                        })
                        continue
                    if result["truncated"]:
                        any_truncated = True
                    rows.append({
                        "Sentence": s,
                        "Prediction": label_display(result["label"]),
                        "Confidence": f"{result['confidence']:.2f}%"
                    })
                    add_to_history(s, result)

                end = time.time()

                st.divider()
                st.subheader("Sentence Breakdown")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption(f"Processed {len(sentences)} sentence(s) in {end - start:.3f} sec")

                if any_truncated:
                    st.warning("One or more sentences exceeded the model's token limit and were truncated.")

# ----------------------------------------------------
# TAB 2: BATCH UPLOAD
# ----------------------------------------------------

with tab_batch:

    st.write(
        "Upload a CSV file with a column of text to analyze every row at once. "
        f"Files are limited to {MAX_BATCH_ROWS} rows per run."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read this file as a CSV: {e}")
            df = None

        if df is not None:

            if df.empty:
                st.warning("This file doesn't contain any rows.")
            else:
                default_col = guess_text_column(df)
                default_index = list(df.columns).index(default_col)

                text_column = st.selectbox(
                    "Select the column that contains text",
                    df.columns,
                    index=default_index
                )

                st.caption("Preview of the selected column (double-check this is actual text, not an ID or number):")
                st.dataframe(df[[text_column]].head(3), use_container_width=True, hide_index=True)

                if len(df) > MAX_BATCH_ROWS:
                    st.warning(
                        f"This file has {len(df)} rows. Only the first {MAX_BATCH_ROWS} will be processed."
                    )
                    df = df.head(MAX_BATCH_ROWS)

                if st.button("Run Batch Analysis"):

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    predictions = []
                    confidences = []
                    errors = []

                    start = time.time()
                    total = len(df)

                    for i, raw_val in enumerate(df[text_column].tolist()):

                        row_text = "" if pd.isna(raw_val) else str(raw_val)

                        if row_text.strip() == "" or not has_meaningful_content(row_text):
                            predictions.append("Invalid")
                            confidences.append(None)
                            errors.append(None)
                            progress_bar.progress((i + 1) / total)
                            status_text.caption(f"Processed {i + 1} / {total} rows")
                            continue

                        result = classify_text(row_text)

                        if not result["success"]:
                            predictions.append("Error")
                            confidences.append(None)
                            errors.append(result["error"])
                        else:
                            predictions.append(label_display(result["label"]))
                            confidences.append(round(result["confidence"], 2))
                            errors.append(None)
                            add_to_history(row_text, result)

                        progress_bar.progress((i + 1) / total)
                        status_text.caption(f"Processed {i + 1} / {total} rows")

                    end = time.time()

                    df["Prediction"] = predictions
                    df["Confidence (%)"] = confidences

                    error_count = sum(1 for e in errors if e is not None)

                    status_text.empty()
                    progress_bar.empty()

                    st.success(f"Finished {total} rows in {end - start:.2f} sec")

                    if error_count > 0:
                        st.warning(f"{error_count} row(s) failed to process and are marked 'Error'.")

                    st.dataframe(df, use_container_width=True, hide_index=True)

                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)

                    st.download_button(
                        "Download Results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name="sentiment_results.csv",
                        mime="text/csv"
                    )

# ----------------------------------------------------
# TAB 3: HISTORY
# ----------------------------------------------------

with tab_history:

    st.write("Everything analyzed so far in this session.")

    if len(st.session_state.history) == 0:
        st.write("No history yet. Run an analysis in the Single Text or Batch Upload tab.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        csv_buffer = io.StringIO()
        history_df.to_csv(csv_buffer, index=False)

        c1, c2 = st.columns(2)

        with c1:
            st.download_button(
                "Download History as CSV",
                data=csv_buffer.getvalue(),
                file_name="sentiment_history.csv",
                mime="text/csv"
            )

        with c2:
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()

# ----------------------------------------------------
# EXAMPLES
# ----------------------------------------------------

st.divider()

st.subheader("Example Inputs")

st.code("I absolutely loved this movie.")

st.code("The food was terrible.")

st.code("The weather is pleasant today.")

st.code("I regret buying this laptop.")

st.code("The meeting was productive.")

# ----------------------------------------------------
# ABOUT
# ----------------------------------------------------

st.divider()

st.subheader("About")

st.write(f"""
This project demonstrates sentiment analysis
using a pretrained DistilBERT transformer model.

The application predicts whether text expresses
positive or negative sentiment, reports the model's
confidence, and labels low-confidence predictions
as Uncertain rather than forcing them into a category.

It supports single-sentence analysis, sentence-by-sentence
breakdown of longer text, and batch processing of CSV files,
with results downloadable at every stage.

The objective of this project is to understand how
pretrained transformer models can be applied to real
NLP workflows, including their edge cases and limits,
not just their happy-path predictions.
""")