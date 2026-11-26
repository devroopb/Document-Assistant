
import hashlib
import json
import re
import pandas as pd


def is_missing(value):
    """Check whether a value is missing or unspecified."""

    if pd.isna(value):
        return True

    return str(value).strip().upper() in [
        "",
        "NOT SPECIFIED",
        "NONE",
        "NAN"
    ]


def create_duplicate_group_id(key):
    """Generate a repeatable identifier for a duplicate group."""

    key_string = json.dumps(
        key,
        sort_keys=True,
        default=str
    )

    return hashlib.sha256(
        key_string.encode("utf-8")
    ).hexdigest()[:16]


def detect_duplicates(financial_df):
    """
    Identify duplicate documents and financial records.

    Detects:
    1. Exact PDF copies using document IDs.
    2. Confirmed duplicate financial records.
    3. Potentially conflicting financial records.

    All original records are retained.
    """

    if financial_df.empty:
        return financial_df.copy()

    df = financial_df.copy()

    required_columns = [
        "record_id",
        "document_id",
        "source_document",
        "company",
        "standardized_metric",
        "standardized_fiscal_year",
        "standardized_currency",
        "standardized_scale",
        "standardized_unit",
        "category",
        "statement_type",
        "cleaned_value"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns for duplicate detection: {missing_columns}"
        )

    # Initialise duplicate-detection results.
    df["document_duplicate_status"] = "Unique document"
    df["duplicate_status"] = "Not assessed"
    df["duplicate_group_id"] = ""
    df["duplicate_of_record_id"] = ""
    df["calculation_status"] = "Pending quality validation"

    # --------------------------------------------------
    # 1. IDENTIFY EXACT DUPLICATE PDF DOCUMENTS
    # --------------------------------------------------

    valid_documents = df[
        ~df["document_id"].apply(is_missing)
        & ~df["source_document"].apply(is_missing)
    ]

    for document_id, group in valid_documents.groupby("document_id"):

        filenames = group["source_document"].drop_duplicates().tolist()

        if len(filenames) <= 1:
            continue

        # Prefer filenames without numbered copy suffixes.
        # Use alphabetical order as a consistent tie-breaker.

        representative_filename = sorted(
            filenames,
            key=lambda name: (
                bool(re.search(r"\s\(\d+\)(?=\.[^.]+$|$)", name)),
                name.lower()
            )
        )[0]

        for filename in filenames:

            matching_rows = (
                (df["document_id"] == document_id)
                & (df["source_document"] == filename)
            )

            if filename == representative_filename:

                df.loc[
                    matching_rows,
                    "document_duplicate_status"
                ] = "Original document"

            else:

                df.loc[
                    matching_rows,
                    "document_duplicate_status"
                ] = "Duplicate document copy"

                df.loc[
                    matching_rows,
                    "duplicate_status"
                ] = "Duplicate document copy"

                df.loc[
                    matching_rows,
                    "calculation_status"
                ] = "Exclude: duplicate document copy"

    # --------------------------------------------------
    # 2. IDENTIFY DUPLICATE FINANCIAL RECORDS
    # --------------------------------------------------

    comparison_columns = [
        "company",
        "standardized_metric",
        "standardized_fiscal_year",
        "standardized_scheduled_year",
        "category",
        "statement_type",
        "standardized_currency",
        "standardized_scale",
        "standardized_unit"
    ]

    # Include segment when available to avoid comparing
    # records belonging to different business segments.
    if "segment" in df.columns:
        comparison_columns.append("segment")

    # Scheduled year is optional in older datasets.
    if "standardized_scheduled_year" not in df.columns:
        df["standardized_scheduled_year"] = "NOT SPECIFIED"

    eligible_rows = df[
        df["document_duplicate_status"] != "Duplicate document copy"
    ]

    for index, row in eligible_rows.iterrows():

        # Do not compare records without sufficient context.
        required_context = [
            "company",
            "standardized_metric",
            "standardized_unit",
            "standardized_currency"
        ]

        if any(is_missing(row[column]) for column in required_context):
            df.at[index, "duplicate_status"] = "Insufficient context"
            df.at[index, "calculation_status"] = "Review required"
            continue

        if (
            is_missing(row["standardized_fiscal_year"])
            and is_missing(row["standardized_scheduled_year"])
        ):
            df.at[index, "duplicate_status"] = "Insufficient context"
            df.at[index, "calculation_status"] = "Review required"

    # Build groups using the reporting context.
    comparison_data = df[
        (df["document_duplicate_status"] != "Duplicate document copy")
        & (df["duplicate_status"] == "Not assessed")
    ]

    grouped = comparison_data.groupby(
        comparison_columns,
        dropna=False,
        sort=False
    )

    for key, group in grouped:

        if len(group) < 2:
            continue

        # Do not compare records with unspecified reporting periods.
        fiscal_year = key[2]
        scheduled_year = key[3]

        if (
            is_missing(fiscal_year)
            and is_missing(scheduled_year)
        ):
            continue

        group_id = create_duplicate_group_id(key)

        # Sort consistently to select a representative record.
        sort_columns = [
            column for column in [
                "source_document",
                "page",
                "record_id"
            ]
            if column in group.columns
        ]

        group = group.sort_values(
            by=sort_columns
        )

        values = pd.to_numeric(
            group["cleaned_value"],
            errors="coerce"
        )

        # Missing values cannot establish a confirmed duplicate.
        if values.isna().any():
            df.loc[
                group.index,
                "duplicate_status"
            ] = "Potential duplicate - missing value"

            df.loc[
                group.index,
                "duplicate_group_id"
            ] = group_id

            df.loc[
                group.index,
                "calculation_status"
            ] = "Exclude: requires review"

            continue

        # Same reporting context and same numerical value.
        if values.round(8).nunique() == 1:

            representative_index = group.index[0]

            representative_record_id = df.at[
                representative_index,
                "record_id"
            ]

            df.loc[
                group.index,
                "duplicate_group_id"
            ] = group_id

            df.at[
                representative_index,
                "duplicate_status"
            ] = "Confirmed duplicate - representative"

            df.at[
                representative_index,
                "calculation_status"
            ] = "Pending quality validation"

            for duplicate_index in group.index[1:]:

                df.at[
                    duplicate_index,
                    "duplicate_status"
                ] = "Confirmed duplicate"

                df.at[
                    duplicate_index,
                    "duplicate_of_record_id"
                ] = representative_record_id

                df.at[
                    duplicate_index,
                    "calculation_status"
                ] = "Exclude: confirmed duplicate"

        # Same reporting context but different values.
        else:

            df.loc[
                group.index,
                "duplicate_status"
            ] = "Potential duplicate conflict"

            df.loc[
                group.index,
                "duplicate_group_id"
            ] = group_id

            df.loc[
                group.index,
                "calculation_status"
            ] = "Exclude: conflicting records"

    # --------------------------------------------------
    # 3. MARK RECORDS WITHOUT IDENTIFIED DUPLICATES
    # --------------------------------------------------

    df.loc[
        df["duplicate_status"] == "Not assessed",
        "duplicate_status"
    ] = "No duplicate detected"

    return df
