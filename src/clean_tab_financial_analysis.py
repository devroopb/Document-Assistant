
import hashlib
import json
import re

import pandas as pd


# --------------------------------------------------
# 1. STANDARDISATION REFERENCE DATA
# --------------------------------------------------

MISSING_VALUES = {
    "",
    "not specified",
    "unknown",
    "none",
    "null",
    "nan"
}

SCALE_MULTIPLIERS = {
    "unit": 1,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}

CURRENCY_ALIASES = {
    "US$": "USD",
    "U.S. DOLLAR": "USD",
    "U.S. DOLLARS": "USD",
    "US DOLLAR": "USD",
    "US DOLLARS": "USD",
    "$": "USD",
    "CA$": "CAD",
    "C$": "CAD",
    "CANADIAN DOLLAR": "CAD",
    "CANADIAN DOLLARS": "CAD",
    "£": "GBP",
    "€": "EUR",
    "₹": "INR",
}

METRIC_ALIASES = {
    "net sales": "Revenue",
    "total sales": "Revenue",
    "total revenue": "Revenue",
    "revenue": "Revenue",
    "sales revenue": "Revenue",

    "cost of sales": "Cost of Revenue",
    "cost of goods sold": "Cost of Revenue",
    "cost of revenue": "Cost of Revenue",

    "gross profit": "Gross Profit",
    "operating income": "Operating Income",
    "operating profit": "Operating Income",
    "net income": "Net Income",
    "net profit": "Net Income",

    "total assets": "Total Assets",
    "total liabilities": "Total Liabilities",
    "total stockholders' equity": "Total Equity",
    "total shareholders' equity": "Total Equity",
    "total equity": "Total Equity",

    "cash and cash equivalents": "Cash and Cash Equivalents",
}


# --------------------------------------------------
# 2. GENERAL NORMALISATION HELPERS
# --------------------------------------------------

def is_missing(value):
    """Check whether a value is missing or unspecified."""

    if value is None:
        return True

    return str(value).strip().lower() in MISSING_VALUES


def normalise_currency(value):
    """Standardise recognised currency names and symbols."""

    if is_missing(value):
        return "NOT SPECIFIED"

    original = str(value).strip()
    upper_value = original.upper()

    for alias, currency_code in CURRENCY_ALIASES.items():
        if upper_value == alias.upper():
            return currency_code

    return upper_value


def normalise_scale(value):
    """Standardise numerical scale descriptions."""

    if is_missing(value):
        return "NOT SPECIFIED"

    scale = str(value).strip().lower()

    aliases = {
        "ones": "unit",
        "units": "unit",
        "thousands": "thousand",
        "000s": "thousand",
        "millions": "million",
        "mm": "million",
        "billions": "billion",
        "bn": "billion",
    }

    return aliases.get(scale, scale)


def normalise_unit(value):
    """Standardise common financial measurement units."""

    if is_missing(value):
        return "NOT SPECIFIED"

    unit = str(value).strip().lower()

    if "%" in unit or "percent" in unit or "percentage" in unit:
        return "percentage"

    if unit in ["share", "shares"]:
        return "shares"

    if "per share" in unit:
        return "per share"

    if unit in ["currency", "monetary", "money"]:
        return "currency"

    return unit


def normalise_metric_name(value):
    """Standardise common financial metric names."""

    if is_missing(value):
        return "NOT SPECIFIED"

    metric = " ".join(str(value).strip().lower().split())

    if metric in METRIC_ALIASES:
        return METRIC_ALIASES[metric]

    return metric.title()


def normalise_fiscal_year(value, metric_name=None):
    """
    Extract a reporting fiscal year.

    Years associated with maturity schedules are not treated
    as reporting fiscal years.
    """

    if is_missing(value):
        return "NOT SPECIFIED"

    year_text = str(value).strip()

    # Maturity-related metrics use a scheduled year, not a fiscal year.
    if metric_name:
        metric_text = str(metric_name).lower()

        maturity_terms = [
            "maturities",
            "maturity",
            "principal due",
            "scheduled repayment"
        ]

        if any(term in metric_text for term in maturity_terms):
            return "NOT SPECIFIED"

    match = re.search(r"\b(19|20)\d{2}\b", year_text)

    if match:
        return match.group(0)

    return year_text

def normalise_scheduled_year(value, metric_name=None):
    """
    Extract a scheduled year from maturity-related financial metrics.

    This keeps debt maturity years separate from reporting fiscal years.
    """

    if is_missing(value) or not metric_name:
        return "NOT SPECIFIED"

    metric_text = str(metric_name).lower()

    maturity_terms = [
        "maturities",
        "maturity",
        "principal due",
        "scheduled repayment"
    ]

    if not any(term in metric_text for term in maturity_terms):
        return "NOT SPECIFIED"

    match = re.search(r"\b(19|20)\d{2}\b", str(value))

    if match:
        return match.group(0)

    # Some maturity years may appear in the metric name itself.
    match = re.search(r"\b(19|20)\d{2}\b", metric_text)

    if match:
        return match.group(0)

    return "NOT SPECIFIED"

def classify_statement(statement_type):
    """Identify the three core financial statement categories."""

    if is_missing(statement_type):
        return None

    statement = str(statement_type).lower()

    if "balance sheet" in statement:
        return "Balance Sheet"

    if "cash flow" in statement:
        return "Cash Flow"

    if (
        "operations" in statement
        or "income statement" in statement
        or "statements of income" in statement
    ):
        return "Income Statement"

    return None


# --------------------------------------------------
# 3. NUMERICAL CLEANING
# --------------------------------------------------

def convert_numeric_value(value):
    """Convert an extracted value into a numeric value safely."""

    if is_missing(value):
        return None

    try:
        numeric_value = pd.to_numeric(value, errors="coerce")

        if pd.isna(numeric_value):
            return None

        return float(numeric_value)

    except (TypeError, ValueError):
        return None


def calculate_normalised_value(row):
    """
    Convert a reported value into base units where its scale
    is known.

    The original reported value remains unchanged.
    Currency conversion is not performed.
    """

    value = row["cleaned_value"]
    scale = row["standardized_scale"]
    unit = row["standardized_unit"]

    if value is None:
        return None

    if unit == "percentage":
        return value

    multiplier = SCALE_MULTIPLIERS.get(scale)

    if multiplier is None:
        return None

    return value * multiplier


# --------------------------------------------------
# 4. MEASUREMENT DESCRIPTION
# --------------------------------------------------

def create_measurement(row):
    """Create a readable description of a financial measurement."""

    currency = row["standardized_currency"]
    scale = row["standardized_scale"]
    unit = row["standardized_unit"]

    if unit == "percentage":
        return "Percentage"

    if unit == "shares":
        if scale == "NOT SPECIFIED":
            return "Shares"

        return f"{scale.title()} of shares"

    if unit == "per share":
        if currency != "NOT SPECIFIED":
            return f"{currency} per share"

        return "Per share"

    if unit == "currency":
        parts = []

        if currency != "NOT SPECIFIED":
            parts.append(currency)

        if scale != "NOT SPECIFIED":
            parts.append(scale)

        return " ".join(parts) if parts else "Currency"

    return str(row["unit"]).strip()


# --------------------------------------------------
# 5. RECORD IDENTIFICATION
# --------------------------------------------------

def create_record_id(record, document_id, document_name, index):
    """
    Generate a repeatable record identifier.

    The document identity and record position help distinguish
    separate records with otherwise identical values.
    """

    identity = {
        "document_id": document_id or document_name or "unknown-document",
        "index": index,
        "metric_name": record.get("metric_name"),
        "fiscal_year": record.get("fiscal_year"),
        "value": record.get("value"),
        "page": record.get("page"),
        "statement_type": record.get("statement_type"),
    }

    identity_text = json.dumps(
        identity,
        sort_keys=True,
        default=str
    )

    return hashlib.sha256(
        identity_text.encode("utf-8")
    ).hexdigest()[:16]


# --------------------------------------------------
# 6. DATA QUALITY VALIDATION
# --------------------------------------------------

def identify_quality_issues(row):
    """Flag questionable or incomplete financial records."""

    flags = []

    if row["standardized_metric"] == "NOT SPECIFIED":
        flags.append("Metric not specified")

    if row["standardized_fiscal_year"] == "NOT SPECIFIED":
        flags.append("Fiscal year not specified")

    if row["cleaned_value"] is None:
        flags.append("Invalid or missing numerical value")

    if row["standardized_unit"] == "NOT SPECIFIED":
        flags.append("Unit not specified")

    if (
        row["standardized_unit"] == "currency"
        and row["standardized_currency"] == "NOT SPECIFIED"
    ):
        flags.append("Currency not specified")

    if (
        row["standardized_unit"] in ["currency", "shares"]
        and row["standardized_scale"] == "NOT SPECIFIED"
    ):
        flags.append("Scale not specified")

    if row["category"] is None:
        flags.append("Statement type not recognised")

    # Flag unusually large share counts reported in thousands.
    value = row["cleaned_value"]

    if (
        value is not None
        and row["standardized_unit"] == "shares"
        and row["standardized_scale"] == "thousand"
        and abs(value) > 1_000_000
    ):
        flags.append("Verify share count and scale")

    return flags


# --------------------------------------------------
# 7. MAIN FINANCIAL DATA CLEANING FUNCTION
# --------------------------------------------------

def create_financial_analysis_dataframe(
    figures,
    company=None,
    document_id=None,
    document_name=None
):
    """
    Create a cleaned financial analysis DataFrame.

    Preserves original extracted fields and adds standardised
    fields, record identifiers, provenance and quality flags.

    Duplicate records are deliberately retained.
    """

    raw_df = pd.DataFrame(figures)

    if raw_df.empty:
        raise ValueError("No financial figures are available.")

    required_columns = [
        "metric_name",
        "fiscal_year",
        "value",
        "currency",
        "scale",
        "unit",
        "page",
        "statement_type"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in raw_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required fields: {missing_columns}"
        )

    # Preserve all original extracted fields.
    analysis_df = raw_df.copy()

    # Attach document-level metadata to every record.
    analysis_df["company"] = company or "NOT SPECIFIED"
    analysis_df["document_id"] = document_id or "NOT SPECIFIED"
    analysis_df["source_document"] = document_name or "NOT SPECIFIED"

    # Preserve original values explicitly.
    analysis_df["original_value"] = analysis_df["value"]
    analysis_df["original_metric_name"] = analysis_df["metric_name"]
    analysis_df["original_currency"] = analysis_df["currency"]
    analysis_df["original_scale"] = analysis_df["scale"]
    analysis_df["original_unit"] = analysis_df["unit"]

    # Generate stable record identifiers.
    analysis_df["record_id"] = [
        create_record_id(
            row.to_dict(),
            document_id,
            document_name,
            index
        )
        for index, (_, row) in enumerate(analysis_df.iterrows())
    ]

    # Standardise financial fields.
    analysis_df["standardized_metric"] = (
        analysis_df["metric_name"].apply(normalise_metric_name)
    )

    analysis_df["standardized_fiscal_year"] = analysis_df.apply(
        lambda row: normalise_fiscal_year(
            row["fiscal_year"],
            row["metric_name"]
        ),
        axis=1
    )

    analysis_df["standardized_scheduled_year"] = analysis_df.apply(
        lambda row: normalise_scheduled_year(
            row["fiscal_year"],
            row["metric_name"]
        ),
        axis=1
    )

    analysis_df["standardized_currency"] = (
        analysis_df["currency"].apply(normalise_currency)
    )

    analysis_df["standardized_scale"] = (
        analysis_df["scale"].apply(normalise_scale)
    )

    analysis_df["standardized_unit"] = (
        analysis_df["unit"].apply(normalise_unit)
    )

    # Standardise statement classification.
    analysis_df["category"] = (
        analysis_df["statement_type"].apply(classify_statement)
    )

    # Convert values without overwriting the original extraction.
    analysis_df["cleaned_value"] = (
        analysis_df["value"].apply(convert_numeric_value)
    )

    # Calculate equivalent base-unit values where possible.
    analysis_df["normalised_value"] = analysis_df.apply(
        calculate_normalised_value,
        axis=1
    )

    # Create readable measurement descriptions.
    analysis_df["measurement"] = analysis_df.apply(
        create_measurement,
        axis=1
    )

    # Generate data quality flags.
    analysis_df["review_flags"] = analysis_df.apply(
        identify_quality_issues,
        axis=1
    )

    # Keep the original app's review_flag field for compatibility.
    analysis_df["review_flag"] = analysis_df["review_flags"].apply(
        lambda flags: "; ".join(flags)
    )

    # Duplicate detection will be handled by a separate module.
    analysis_df["duplicate_status"] = "Not assessed"
    analysis_df["calculation_status"] = "Pending duplicate assessment"

    # Retain all records, including duplicates and supporting disclosures.
    analysis_df = analysis_df.sort_values(
        by=[
            "standardized_fiscal_year",
            "category",
            "standardized_metric"
        ],
        ascending=[False, True, True],
        na_position="last"
    ).reset_index(drop=True)

    return analysis_df
