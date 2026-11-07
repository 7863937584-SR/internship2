from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import (
    BODY_ALIASES,
    CHANNEL_ALIASES,
    CREATED_AT_ALIASES,
    LABEL_ALIASES,
    MAX_TEXT_CHARS,
    PRODUCT_ALIASES,
    SAMPLE_DATA_PATH,
    SUBJECT_ALIASES,
    TARGET_COLUMNS,
    TENURE_ALIASES,
)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = [str(column).strip().lower() for column in clean.columns]
    return clean


def _matching_series(df: pd.DataFrame, aliases: Iterable[str]) -> list[pd.Series]:
    alias_set = set(aliases)
    series = []
    for position, column in enumerate(df.columns):
        if column in alias_set:
            values = df.iloc[:, position]
            series.append(values.fillna("").astype(str).str.strip())
    return series


def _combine_series(df: pd.DataFrame, aliases: Iterable[str]) -> pd.Series:
    parts = _matching_series(df, aliases)
    if not parts:
        return pd.Series([""] * len(df), index=df.index, dtype="object")

    combined = parts[0].copy()
    for part in parts[1:]:
        combined = combined + " " + part
    return combined.str.strip()


def _first_non_empty(df: pd.DataFrame, aliases: Iterable[str]) -> pd.Series:
    parts = _matching_series(df, aliases)
    result = pd.Series([""] * len(df), index=df.index, dtype="object")
    filled = pd.Series(False, index=df.index)

    for part in parts:
        mask = part.ne("") & ~filled
        result = result.mask(mask, part)
        filled = filled | mask

    return result


def _truncate_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.slice(0, MAX_TEXT_CHARS)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical frame with text features and target labels."""
    if df.empty:
        raise ValueError("The input data is empty.")

    clean = _clean_columns(df)
    normalized = pd.DataFrame(index=clean.index)

    normalized["subject"] = _combine_series(clean, SUBJECT_ALIASES)
    normalized["body"] = _combine_series(clean, BODY_ALIASES)
    normalized["channel"] = _first_non_empty(clean, CHANNEL_ALIASES)
    normalized["product_area"] = _first_non_empty(clean, PRODUCT_ALIASES)
    normalized["customer_tenure_days"] = pd.to_numeric(
        _first_non_empty(clean, TENURE_ALIASES), errors="coerce"
    )
    normalized["created_at"] = _first_non_empty(clean, CREATED_AT_ALIASES)

    base_text = (
        normalized["subject"].fillna("").astype(str)
        + " "
        + normalized["body"].fillna("").astype(str)
    )
    has_content = base_text.str.strip().ne("")

    combined_text = base_text.copy()
    for name, values in (
        ("channel", normalized["channel"]),
        ("product", normalized["product_area"]),
    ):
        cleaned = values.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
        token = cleaned.where(cleaned.ne(""), "")
        metadata = token.map(lambda value: f" {name}:{value}" if value else "")
        combined_text = combined_text + metadata.fillna("").astype(str)

    normalized["text"] = _truncate_text(combined_text.where(has_content, "")).str.strip()

    for target in TARGET_COLUMNS:
        normalized[target] = _first_non_empty(clean, LABEL_ALIASES[target])

    return normalized


def ensure_text_present(df: pd.DataFrame, frame_name: str = "data") -> None:
    if {"subject", "body"}.issubset(df.columns):
        content = (
            df["subject"].fillna("").astype(str)
            + " "
            + df["body"].fillna("").astype(str)
        )
    elif "text" in df.columns:
        content = df["text"].fillna("").astype(str)
    else:
        content = pd.Series([""] * len(df), index=df.index)

    if content.str.strip().eq("").all():
        raise ValueError(
            f"{frame_name} must contain at least one text column such as "
            "subject/message, ticket_text, description, body, or content."
        )


def ensure_training_labels(df: pd.DataFrame) -> None:
    missing = [
        target
        for target in TARGET_COLUMNS
        if target not in df.columns or df[target].fillna("").astype(str).str.strip().eq("").all()
    ]
    if missing:
        raise ValueError(
            "Training data is missing required label columns: "
            + ", ".join(missing)
            + ". Expected category, priority, and sentiment."
        )


TEMPLATES = [
    ("Billing", "Critical", "Negative", "Duplicate charge on my card", "I was charged twice for the same invoice and my account is now overdrawn. Please refund the duplicate payment immediately."),
    ("Billing", "High", "Negative", "Unexpected subscription renewal", "My subscription renewed without warning and I did not approve the charge. I need a refund and cancellation confirmation."),
    ("Billing", "Medium", "Neutral", "Invoice amount question", "Can you explain the line items on my latest invoice? The total is higher than I expected."),
    ("Billing", "Low", "Positive", "Receipt copy request", "Could you send another copy of my payment receipt for expense reporting?"),
    ("Billing", "High", "Negative", "Refund still not received", "I requested a refund two weeks ago and the money has not appeared in my bank account. This is causing financial stress."),
    ("Billing", "Medium", "Neutral", "Payment method update failed", "I tried to update my card details but the payment page keeps rejecting the new expiration date."),
    ("Billing", "Low", "Positive", "Billing contact change", "Please update the billing email on my account to our finance team address."),
    ("Billing", "Critical", "Negative", "Fraudulent charge reported", "I see an unauthorized charge from your company on my statement. I did not make this purchase and need immediate help."),

    ("Technical Support", "Critical", "Negative", "Application outage for all users", "The dashboard is completely down and our team cannot access customer records. This is blocking production work."),
    ("Technical Support", "High", "Negative", "Login loop after update", "Since the latest release, the app signs me out every few minutes and I cannot save any work."),
    ("Technical Support", "Medium", "Neutral", "Export fails with large files", "When I export more than a few thousand rows, the download stops halfway and shows a generic error."),
    ("Technical Support", "Low", "Positive", "Keyboard shortcut request", "Is there a keyboard shortcut for creating a new ticket? I would like to speed up data entry."),
    ("Technical Support", "High", "Negative", "Integration webhook not firing", "Our webhook endpoint stopped receiving events yesterday. Retries did not fix the issue."),
    ("Technical Support", "Medium", "Neutral", "Mobile app crashes on launch", "The mobile app closes immediately after the splash screen on my Android phone."),
    ("Technical Support", "Low", "Neutral", "Browser compatibility question", "Does the console support the latest Safari version? I see a warning banner."),
    ("Technical Support", "Critical", "Negative", "Data sync corruption", "Records are duplicating every time the app syncs. We are worried about data loss and corruption."),

    ("Account Access", "High", "Negative", "Cannot reset password", "I requested a password reset three times but never receive the email. I am locked out of my account."),
    ("Account Access", "Medium", "Neutral", "Two-factor authentication setup", "I enabled two-factor authentication but lost my backup codes. How can I regain access safely?"),
    ("Account Access", "Low", "Positive", "Update profile information", "Please help me change my job title and department on the account profile."),
    ("Account Access", "Critical", "Negative", "Account takeover concern", "I received a login alert from an unknown device and I think someone else accessed my account."),
    ("Account Access", "High", "Negative", "SSO login not working", "Our team cannot log in through SSO after the identity provider certificate changed."),
    ("Account Access", "Medium", "Neutral", "Permissions for new employee", "A new employee needs access to the support queue but cannot see any tickets."),
    ("Account Access", "Low", "Positive", "Change account email", "I need to update the account email after our company migrated domains."),
    ("Account Access", "High", "Negative", "Locked out after failed attempts", "My account is locked after several failed login attempts and support has not responded."),

    ("Shipping & Delivery", "High", "Negative", "Package marked delivered but missing", "Tracking says the package was delivered, but nothing arrived at our office. We need a replacement urgently."),
    ("Shipping & Delivery", "Medium", "Neutral", "Delivery date changed", "The estimated delivery date moved twice and there is no explanation in the tracking page."),
    ("Shipping & Delivery", "Low", "Positive", "Shipping address confirmation", "Please confirm the shipping address on order 4492 before it ships tomorrow."),
    ("Shipping & Delivery", "Critical", "Negative", "Perishable item arrived spoiled", "The refrigerated shipment arrived warm and the product is unusable. We need a refund today."),
    ("Shipping & Delivery", "Medium", "Neutral", "Tracking number not updating", "The carrier tracking link has not updated for five days and shows no new scans."),
    ("Shipping & Delivery", "Low", "Neutral", "Shipping method options", "Can I change from standard shipping to express for an order that has not shipped yet?"),
    ("Shipping & Delivery", "High", "Negative", "Wrong item delivered", "We received a different product than the one ordered. The box contains the wrong size and color."),
    ("Shipping & Delivery", "Medium", "Negative", "Delayed customs clearance", "Our international shipment is stuck in customs and the delivery guarantee has passed."),

    ("Product Question", "Low", "Neutral", "Feature availability question", "Does the enterprise plan include advanced audit logs and custom roles?"),
    ("Product Question", "Medium", "Neutral", "API rate limit details", "What is the API rate limit for the growth plan, and can it be increased for batch jobs?"),
    ("Product Question", "Low", "Positive", "Documentation request", "Could you share the latest API documentation and sample payloads for the reporting endpoint?"),
    ("Product Question", "Medium", "Neutral", "Integration compatibility", "Will the connector work with our existing Salesforce sandbox environment?"),
    ("Product Question", "Low", "Neutral", "Pricing tier comparison", "What is the difference between the team and business plans for support ticket volume?"),
    ("Product Question", "Medium", "Positive", "Trial extension request", "We are evaluating the product and need a few more days to finish testing. Can the trial be extended?"),
    ("Product Question", "Low", "Neutral", "Data export format", "Which file formats are available when exporting reports from the dashboard?"),
    ("Product Question", "Medium", "Neutral", "Webhook signature verification", "How should we verify webhook signatures in our integration? The docs example does not match the payload."),

    ("Complaint", "High", "Negative", "Repeated service failures", "This is the third time this month that your service failed during our busiest hours. We are very frustrated."),
    ("Complaint", "Medium", "Negative", "Unhelpful support response", "The previous agent gave a copy-paste answer that did not solve the problem and closed the ticket too early."),
    ("Complaint", "Low", "Neutral", "Feedback on onboarding", "The onboarding checklist was confusing because several steps were already completed automatically."),
    ("Complaint", "Critical", "Negative", "Escalation required for contract issue", "Our account manager promised features that are not included and now procurement is questioning the contract."),
    ("Complaint", "High", "Negative", "SLA breach notification", "We missed our SLA because the incident was not escalated. Please explain the resolution plan."),
    ("Complaint", "Medium", "Negative", "Billing dispute escalation", "I have disputed the same charge twice and keep receiving automated replies. This needs a supervisor."),
    ("Complaint", "Low", "Positive", "Compliment for support team", "The support engineer resolved our migration issue quickly and explained every step clearly. Thank you."),
    ("Complaint", "Medium", "Neutral", "Product feedback summary", "Several users find the search filters confusing. We suggest adding saved filter presets."),

    ("Refund & Cancellation", "High", "Negative", "Cancel service immediately", "Please cancel our subscription today and confirm there will be no further charges."),
    ("Refund & Cancellation", "Medium", "Neutral", "Refund status inquiry", "I approved the refund last week. Can you confirm when the payment will be reversed?"),
    ("Refund & Cancellation", "Low", "Positive", "Partial refund accepted", "The partial refund is acceptable. Please process it to the original payment method."),
    ("Refund & Cancellation", "Critical", "Negative", "Unauthorized renewal refund", "You charged us after the cancellation date. I need a full refund and written confirmation."),
    ("Refund & Cancellation", "Medium", "Negative", "Return label request", "I need a prepaid return label for the defective unit. The portal is not generating one."),
    ("Refund & Cancellation", "Low", "Neutral", "Cancellation effective date", "If I cancel now, will the service remain active until the end of the billing period?"),
    ("Refund & Cancellation", "High", "Negative", "Restocking fee dispute", "The restocking fee was not disclosed before purchase. Please waive it or explain the policy."),
    ("Refund & Cancellation", "Medium", "Neutral", "Refund to different card", "Can the refund be sent to a different card because the original one was closed?"),

    ("Other", "Medium", "Neutral", "General inquiry", "I have a question that does not fit the usual categories. Could someone point me to the right team?"),
    ("Other", "Low", "Positive", "Thank you note", "Just wanted to thank the team for the quick response to our onboarding question."),
    ("Other", "Medium", "Neutral", "Partnership inquiry", "We are interested in a reseller partnership and would like to schedule an introductory call."),
    ("Other", "Low", "Neutral", "Office location question", "Where is your support office located, and what are the business hours for phone support?"),
    ("Other", "High", "Negative", "Urgent vendor issue", "A vendor we rely on is down and we need guidance on failover options immediately."),
    ("Other", "Medium", "Neutral", "Event sponsorship request", "We would like to discuss sponsorship opportunities for an industry webinar next quarter."),
    ("Other", "Low", "Positive", "Feature suggestion", "It would be great if reports could be scheduled automatically and emailed weekly."),
    ("Other", "Medium", "Neutral", "Account migration question", "We are migrating from another provider and need a checklist for exporting historical tickets."),
]

PRODUCTS = [
    "payments",
    "invoicing",
    "subscriptions",
    "console",
    "api",
    "mobile_app",
    "integrations",
    "identity",
    "sso",
    "permissions",
    "logistics",
    "orders",
    "warehouse",
    "reports",
    "support",
    "sla",
    "account_management",
    "returns",
    "general",
    "partnerships",
    "events",
]

CHANNELS = ["email", "chat", "phone", "social", "portal"]


def synthetic_tickets(n_rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")

    rng = random.Random(seed)
    rows = []
    base_date = pd.Timestamp("2025-01-01")

    for index in range(n_rows):
        category, priority, sentiment, subject, message = TEMPLATES[rng.randrange(len(TEMPLATES))]
        ticket_id = f"TCK-{100000 + index:06d}"
        varied_message = f"{message} Reference {ticket_id}."
        created_at = base_date + pd.Timedelta(days=rng.randint(0, 540), hours=rng.randint(0, 23))

        rows.append(
            {
                "ticket_id": ticket_id,
                "subject": subject,
                "message": varied_message,
                "channel": rng.choice(CHANNELS),
                "product_area": rng.choice(PRODUCTS),
                "customer_tenure_days": rng.randint(1, 2400),
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "category": category,
                "priority": priority,
                "sentiment": sentiment,
            }
        )

    return pd.DataFrame(rows)


def ensure_sample_data(n_rows: int = 1200, path: Path = SAMPLE_DATA_PATH) -> Path:
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_tickets(n_rows=n_rows).to_csv(path, index=False)
    return path


def load_sample_tickets(path: Path = SAMPLE_DATA_PATH, n_rows: int = 1200) -> pd.DataFrame:
    ensure_sample_data(n_rows=n_rows, path=path)
    raw = pd.read_csv(path)
    return normalize_dataframe(raw)
