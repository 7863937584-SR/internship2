# SupportIQ

SupportIQ is a Streamlit application for customer support ticket analytics and machine-learning predictions. It predicts ticket category, priority, and sentiment from ticket text.

## Features

- Dashboard with ticket distribution charts and model metrics
- Single-ticket prediction form
- Batch CSV upload and prediction download
- Interactive data explorer
- In-app model training from the bundled dataset or a labeled CSV
- Local scikit-learn pipelines saved with Joblib

## Project structure

```text
SupportIQ/
├── app.py
├── pages/
├── src/
├── training/
├── models/
├── data/
├── requirements.txt
└── README.md
```

## Local setup

```bash
cd SupportIQ
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Train or retrain the model

The repository includes a bundled synthetic dataset and a training command:

```bash
python training/train_model.py
```

You can also train from the **Train Model** page in the app. Uploading a custom training CSV requires the labels `category`, `priority`, and `sentiment`, plus at least one text field such as `subject`/`message`, `ticket_text`, `description`, `body`, or `content`.

## CSV prediction

The batch prediction page accepts CSV files with common column aliases. It detects common delimiters, validates that text exists, predicts up to 10,000 rows, and provides a downloadable predictions CSV.

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, select the repository.
3. Set the main file path to `app.py`.
4. Deploy.

The app includes `requirements.txt`, a bundled sample dataset, and a trained model artifact under `models/`, so it can run without external services.
