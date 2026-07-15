# AI Sentiment Analyzer

A Streamlit app that classifies text as Positive or Negative using a
pretrained DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`),
with confidence scoring, sentence-level breakdown, and CSV batch processing.

This project went through a manual QA pass (40 test cases across Single Text,
Batch Upload, and History): see the [Tested Behavior](#tested-behavior)
section below for what was actually verified, what broke and got fixed, and
what's still unverified.

## Features

- **Single text analysis** : classify one piece of text and see the
  prediction, confidence score, and inference time.
- **Sentence-by-sentence breakdown** : split a paragraph into individual
  sentences and see how sentiment shifts across them, instead of one
  verdict for the whole block.
- **Uncertain band** : predictions below 60% confidence are labeled
  "Uncertain" instead of being forced into Positive or Negative.
- **Invalid-input detection** : text with no letters or numbers (blank
  input, a stray comma, "???") is caught and labeled "Invalid" instead of
  producing a confident-sounding but meaningless prediction.
- **Batch upload** : upload a CSV, auto-detects the likely text column
  (instead of defaulting to column 1, which is often an ID column), shows
  a preview of the selected column so a wrong pick is obvious before you
  run, then processes every row with results downloadable as a new CSV.
- **Session history** : every analysis run in the current session
  (single text, sentence breakdown, or batch) is logged and can be
  downloaded or cleared.
- **Truncation warnings** : DistilBERT has a 512-token limit; if text
  exceeds that, the app tells you instead of silently scoring a
  partial input.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure the `.streamlit/config.toml` file sits in the same directory
as `app.py` — it forces a light theme so the styling doesn't break
under a visitor's dark mode setting.

## Project structure

```
.
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── README.md
```

## Model limitations

- Fine-tuned on movie reviews (SST-2), so it can misread sarcasm,
  mixed sentiment, or text from very different domains (e.g.
  financial or medical writing).
- Only outputs Positive/Negative — no native Neutral class. The
  "Uncertain" label in this app is a confidence-threshold heuristic,
  not something the model itself predicts. In practice, the model tends
  to be highly confident even on genuinely ambiguous text (see
  [Tested Behavior](#tested-behavior)), so the Uncertain band triggers
  less often than you'd expect.
- Input longer than 512 tokens is truncated before scoring.

## Tested Behavior

Manually tested against a 40-case checklist. Results below reflect what
actually happened, not what was expected going in.

### Confirmed working

- **Single text analysis** : correct Positive/Negative predictions with
  confidence scores across plain, emoji-containing, and non-English
  inputs. Empty/whitespace input is correctly blocked with a warning.
- **Sentence-by-sentence breakdown** : correctly splits mixed-sentiment
  paragraphs and scores each sentence independently; each sentence is
  logged to History individually. Text without sentence-ending
  punctuation is handled as a single block rather than crashing.
- **Batch upload core flow** : CSV upload, column selection, per-row
  scoring, results table, and CSV download all work end to end.
- **Batch edge cases** : row-count limit warning (>500 rows), empty CSV
  (headers only), non-CSV file rejection, and malformed CSV all show
  clear error/warning messages instead of crashing.
- **History tab** : empty-state message, and entries logged from single
  text and sentence breakdown, confirmed working.
- **General/UI** : sidebar limitations text, example inputs, About
  section, and the light theme holding under system dark mode were all
  confirmed visually.

### Bugs found during testing and fixed

1. **Batch column selector defaulted to the wrong column.** The dropdown
   defaulted to the first column in the CSV (often an ID column), which
   silently produced meaningless results — e.g. every row scored
   "Positive" at ~98% because it was classifying row numbers, not review
   text. **Fixed**: the app now guesses the likely text column (by name
   match or by longest average string length) and shows a preview of the
   selected column's values so a wrong pick is obvious before running.

2. **Blank CSV cells crashed the batch run.** A row with an empty cell
   caused `AttributeError: 'float' object has no attribute 'strip'`,
   because `pandas` doesn't always convert `NaN` to a string via
   `.astype(str)` — depending on the column's dtype, a missing value can
   stay a `float` and crash on the first string operation. **Fixed**:
   each value is now checked with `pd.isna()` before conversion, so blank
   cells are reliably caught and marked "Invalid" instead of crashing the
   whole batch. Verified against the actual CSV that triggered the
   original crash.

3. **Batch rows weren't logged to History.** Only single-text and
   sentence-breakdown runs were being added to session history at first.
   **Fixed**: batch rows are now logged the same way.

4. **Punctuation-only input returned a confident, meaningless prediction.**
   A single comma, or input like "???", isn't real text but wasn't
   caught by the empty-string check, so it got scored anyway (e.g. a
   lone comma returned "Positive"). **Fixed**: added a check requiring at
   least one letter or digit; anything else is now labeled "Invalid"
   (both in Single Text and Batch Upload) instead of being scored.

5. **Truncation warning was untestable at the original character limit.**
   The single-text box was capped at 2000 characters, which isn't
   enough to reliably reach the model's 512-token limit with normal
   English text. **Fixed**: raised the cap to 3000 characters so the
   truncation warning is actually reachable.

### Known behavior (not bugs, but worth knowing)

- **The Uncertain band rarely triggers.** Testing a deliberately flat
  sentence ("It was okay I guess, nothing special either way.") still
  returned a confident 99.05% Negative, not Uncertain. The model appears
  to commit strongly even on ambiguous phrasing, so don't expect to see
  "Uncertain" often; it mainly protects against edge cases, not general
  ambiguity.
- **Small batches finish before the progress bar is really visible.** For
  a 20-row file, processing completes in under a second, so the progress
  bar can flash by unnoticed. This is expected — it becomes visible on
  larger batches.
- **Non-English input doesn't crash but isn't reliable.** Spanish input
  returned a plausible-looking Negative prediction, but the model is
  English-trained (SST-2), so treat any non-English result as
  unverified rather than accurate.

### Not yet verified

A few checklist items weren't exercised during this round of testing and
should be treated as open, not passing:

- History tab: downloading history as CSV, the Clear History button,
  and history persisting across tab switches (HI-02 through HI-04).
- History resetting on a full app restart (HI-05) — expected behavior,
  but not explicitly confirmed.
- Re-confirming batch-to-History logging (fix #3 above) against the
  current code, since the original test predates the fix.
- Re-testing the punctuation-only input case (fix #4) and the
  long-passage truncation cases (fix #5) against the current code, since
  both were tested before those fixes went in.

## Possible future additions

- Word-level attribution (highlighting which words pushed a
  prediction toward positive/negative) using attention or gradient-based
  methods.
- Support for additional models/languages via a dropdown.
- Persistent history across sessions (currently in-memory only, via
  `st.session_state`, and resets when the app restarts).