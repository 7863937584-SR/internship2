from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
TRAINING_DIR = ROOT_DIR / "training"

SAMPLE_DATA_PATH = DATA_DIR / "sample_tickets.csv"
MODEL_PATH = MODELS_DIR / "support_model.joblib"

MAX_UPLOAD_ROWS = 10_000
MAX_TEXT_CHARS = 20_000
TARGET_COLUMNS = ("category", "priority", "sentiment")

SUBJECT_ALIASES = ("subject", "title", "summary", "ticket_subject")
BODY_ALIASES = (
    "message",
    "body",
    "description",
    "ticket_text",
    "content",
    "text",
    "issue",
    "details",
)
CHANNEL_ALIASES = ("channel", "source", "origin", "contact_channel", "medium")
PRODUCT_ALIASES = (
    "product_area",
    "product",
    "product_line",
    "service",
    "department",
    "area",
)
TENURE_ALIASES = (
    "customer_tenure_days",
    "tenure_days",
    "account_age_days",
    "customer_age_days",
)
CREATED_AT_ALIASES = ("created_at", "timestamp", "date", "created_on")
LABEL_ALIASES = {
    "category": ("category", "issue_category", "ticket_category", "ticket_type", "case_category"),
    "priority": ("priority", "urgency", "severity"),
    "sentiment": ("sentiment", "tone", "customer_sentiment"),
}
