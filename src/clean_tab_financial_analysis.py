
import pandas as pd


def classify_statement(statement_type):
    """Identify the three core financial statement categories."""

    statement = str(statement_type).lower()

    if "balance sheet" in statement or "balance sheets" in statement:
        return "Balance Sheet"

    if "cash flow" in statement:
        return "Cash Flow"

    if "operations" in statement or "income statement" in statement:
        return "Income Statement"

    if "statements of income" in statement:
        return "Income Statement"

    return None


def create_measurement(row):
    """
    Combine currency, scale and unit into one readable measurement.
    Do not modify the original numerical value.
    """

    currency = str(row["currency"]).strip().upper()
    scale = str(row["scale"]).strip().lower()
    unit = str(row["unit"]).strip().lower()

    # Identify percentage figures.
    if "percent" in unit or "%" in unit:
        return "Percentage"

    # Identify per-share figures.
    if "per share" in unit:
        if currency not in ["", "NOT SPECIFIED", "UNKNOWN"]:
            return f"{currency} per share"
        return "Per share"

    # Identify share counts.
    if unit in ["share", "shares"]:
        if scale in ["thousand", "thousands"]:
            return "Thousands of shares"
        if scale in ["million", "millions"]:
            return "Millions of shares"
        return "Shares"

    # Identify monetary figures.
    monetary_units = [
        "currency", "monetary", "usd", "dollars",
        "usd millions", "usd millions"
    ]

    if (
        unit in monetary_units
        or currency not in ["", "NOT SPECIFIED", "UNKNOWN"]
    ):
        parts = []

        if currency not in ["", "NOT SPECIFIED", "UNKNOWN"]:
            parts.append(currency)

        if scale in ["million", "millions"]:
            parts.append("millions")
        elif scale in ["billion", "billions"]:
            parts.append("billions")
        elif scale in ["thousand", "thousands"]:
            parts.append("thousands")

        return " ".join(parts) if parts else "Currency"

    return str(row["unit"]).strip()


def identify_quality_issues(row):
    """Flag records whose measurement metadata needs checking."""

    flags = []

    currency = str(row["currency"]).strip().lower()
    scale = str(row["scale"]).strip().lower()
    unit = str(row["unit"]).strip().lower()

    # Flag missing measurement information.
    if unit in ["", "not specified", "unknown"]:
        flags.append("Unit not specified")

    # Flag monetary figures without a currency.
    if (
        unit in ["currency", "monetary"]
        and currency in ["", "not specified", "unknown"]
    ):
        flags.append("Currency not specified")

    # Flag monetary figures without a scale.
    if (
        unit in ["currency", "monetary"]
        and scale in ["", "not specified", "unknown"]
    ):
        flags.append("Scale not specified")

    # Flag unusually large share counts reported in thousands.
    if (
        unit in ["share", "shares"]
        and scale in ["thousand", "thousands"]
        and abs(row["value"]) > 1_000_000
    ):
        flags.append("Verify share count and scale")

    return "; ".join(flags)


def create_financial_analysis_dataframe(figures):
    """
    Create a simplified financial analysis DataFrame.

    The original extracted records remain untouched.
    Only figures from the three core financial statements
    are included in this analysis table.
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

    # Determine which core financial statement each record belongs to.
    raw_df["category"] = raw_df["statement_type"].apply(
        classify_statement
    )

    # Exclude supporting notes and other non-core disclosures.
    analysis_df = raw_df[
        raw_df["category"].notna()
    ].copy()

    if analysis_df.empty:
        raise ValueError(
            "No figures from the three core financial statements "
            "were identified."
        )

    # Remove only exact duplicates within the same statement,
    # fiscal year and source page.
    analysis_df = analysis_df.drop_duplicates(
        subset=[
            "metric_name",
            "fiscal_year",
            "value",
            "page",
            "statement_type"
        ]
    )

    # Create one readable measurement field.
    analysis_df["measurement"] = analysis_df.apply(
        create_measurement,
        axis=1
    )

    # Flag questionable metadata without changing the figures.
    analysis_df["review_flag"] = analysis_df.apply(
        identify_quality_issues,
        axis=1
    )

    # Keep only the columns needed for financial analysis.
    analysis_df = analysis_df[
        [
            "metric_name",
            "category",
            "fiscal_year",
            "value",
            "measurement",
            "page",
            "review_flag"
        ]
    ]

    # Sort by reporting year and financial metric.
    analysis_df = analysis_df.sort_values(
        by=["fiscal_year", "category", "metric_name"],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    return analysis_df