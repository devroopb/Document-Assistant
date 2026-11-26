
import sys
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MAX_UNIQUE_VALUES = 20
MAX_SAMPLE_ROWS = 15

KEY_COLUMNS = [
    "company",
    "document_id",
    "source_document",
    "metric_name",
    "standardized_metric",
    "fiscal_year",
    "standardized_fiscal_year",
    "scheduled_year",
    "currency",
    "standardized_currency",
    "scale",
    "standardized_scale",
    "unit",
    "standardized_unit",
    "statement_type",
    "category",
    "duplicate_status",
    "document_duplicate_status",
    "calculation_status",
    "duplicate_group_id",
    "duplicate_of_record_id",
    "review_flag",
    "review_flags",
]

SAMPLE_COLUMNS = [
    "company",
    "source_document",
    "document_id",
    "record_id",
    "metric_name",
    "fiscal_year",
    "value",
    "currency",
    "scale",
    "unit",
    "page",
    "statement_type",
    "standardized_metric",
    "standardized_fiscal_year",
    "standardized_unit",
    "document_duplicate_status",
    "duplicate_status",
    "calculation_status",
]


# --------------------------------------------------
# REPORT HELPERS
# --------------------------------------------------

def section(title):
    return f"\n{'=' * 75}\n{title}\n{'=' * 75}\n"


def format_distribution(df, column):
    if column not in df.columns:
        return f"{column}: COLUMN NOT FOUND\n"

    values = (
        df[column]
        .fillna("[EMPTY]")
        .astype(str)
        .str.strip()
        .replace("", "[BLANK]")
    )

    counts = values.value_counts(dropna=False)

    output = [f"\n{column} ({len(counts)} distinct values):"]

    for value, count in counts.head(MAX_UNIQUE_VALUES).items():
        output.append(f"  {value}: {count}")

    if len(counts) > MAX_UNIQUE_VALUES:
        output.append(
            f"  ... {len(counts) - MAX_UNIQUE_VALUES} more values"
        )

    return "\n".join(output) + "\n"


def format_sample(df, title, columns, max_rows=MAX_SAMPLE_ROWS):
    output = [section(title)]

    available_columns = [
        column for column in columns if column in df.columns
    ]

    if df.empty:
        output.append("No matching records found.\n")
        return "".join(output)

    if not available_columns:
        output.append("None of the requested columns exist.\n")
        return "".join(output)

    sample = df[available_columns].head(max_rows)

    output.append(
        sample.to_string(index=False, max_colwidth=45)
    )

    if len(df) > max_rows:
        output.append(
            f"\nShowing {max_rows} of {len(df)} matching records."
        )

    output.append("\n")
    return "".join(output)


# --------------------------------------------------
# DUPLICATE DOCUMENT DIAGNOSTICS
# --------------------------------------------------

def duplicate_document_diagnostics(df):

    output = [section("5. DUPLICATE DOCUMENT DIAGNOSTICS")]

    required = [
        "source_document",
        "document_id",
        "document_duplicate_status",
        "duplicate_status",
        "calculation_status",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        output.append(
            "Cannot complete duplicate diagnostics. Missing columns:\n"
        )
        output.append(", ".join(missing) + "\n")
        return "".join(output)

    # --------------------------------------------------
    # A. RECORD COUNTS BY SOURCE FILE
    # --------------------------------------------------

    output.append("\nA. RECORD COUNTS BY SOURCE FILE\n")

    file_counts = (
        df.groupby(
            ["company", "source_document", "document_id"],
            dropna=False
        )
        .size()
        .reset_index(name="record_count")
    )

    output.append(
        file_counts.to_string(index=False, max_colwidth=60)
    )
    output.append("\n")

    # --------------------------------------------------
    # B. DOCUMENT CLASSIFICATION BY SOURCE FILE
    # --------------------------------------------------

    output.append("\nB. DOCUMENT CLASSIFICATION BY SOURCE FILE\n")

    document_status = (
        df.groupby(
            [
                "source_document",
                "document_duplicate_status"
            ],
            dropna=False
        )
        .size()
        .reset_index(name="record_count")
    )

    output.append(
        document_status.to_string(index=False, max_colwidth=60)
    )
    output.append("\n")

    # --------------------------------------------------
    # C. DUPLICATE STATUS BY SOURCE FILE
    # --------------------------------------------------

    output.append("\nC. DUPLICATE STATUS BY SOURCE FILE\n")

    duplicate_status = (
        df.groupby(
            ["source_document", "duplicate_status"],
            dropna=False
        )
        .size()
        .reset_index(name="record_count")
    )

    output.append(
        duplicate_status.to_string(index=False, max_colwidth=60)
    )
    output.append("\n")

    # --------------------------------------------------
    # D. CALCULATION STATUS BY SOURCE FILE
    # --------------------------------------------------

    output.append("\nD. CALCULATION STATUS BY SOURCE FILE\n")

    calculation_status = (
        df.groupby(
            ["source_document", "calculation_status"],
            dropna=False
        )
        .size()
        .reset_index(name="record_count")
    )

    output.append(
        calculation_status.to_string(index=False, max_colwidth=60)
    )
    output.append("\n")

    # --------------------------------------------------
    # E. DUPLICATE GROUP AND REFERENCE FIELDS
    # --------------------------------------------------

    output.append("\nE. DUPLICATE GROUP AND REFERENCE FIELDS\n")

    for column in ["duplicate_group_id", "duplicate_of_record_id"]:

        if column not in df.columns:
            output.append(f"{column}: COLUMN NOT FOUND\n")
            continue

        populated = df[column].notna().sum()

        output.append(
            f"{column}: {populated} populated, "
            f"{len(df) - populated} missing\n"
        )

    # --------------------------------------------------
    # F. SAMPLE OF EACH DOCUMENT CLASSIFICATION
    # --------------------------------------------------

    output.append(
        format_sample(
            df,
            "F. SAMPLE OF DOCUMENT DUPLICATE CLASSIFICATIONS",
            SAMPLE_COLUMNS
        )
    )

    return "".join(output)


# --------------------------------------------------
# MAIN DIAGNOSTIC
# --------------------------------------------------

def inspect_csv(csv_path):

    csv_path = Path(csv_path).expanduser()

    if not csv_path.exists():
        print(f"\nERROR: File not found: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as error:
        print(f"\nUnable to read CSV: {error}")
        return

    report = []

    report.append(section("FINANCIAL DATA DIAGNOSTIC REPORT"))
    report.append(f"File: {csv_path.name}\n")
    report.append(f"Full path: {csv_path}\n")
    report.append(f"Rows: {len(df)}\n")
    report.append(f"Columns: {len(df.columns)}\n")

    # --------------------------------------------------
    # 1. COLUMN STRUCTURE
    # --------------------------------------------------

    report.append(section("1. COLUMN STRUCTURE"))

    for column in df.columns:
        report.append(
            f"{column} | dtype: {df[column].dtype}\n"
        )

    # --------------------------------------------------
    # 2. MISSING AND BLANK VALUES
    # --------------------------------------------------

    report.append(section("2. MISSING AND BLANK VALUES"))

    for column in df.columns:

        missing = int(df[column].isna().sum())

        blank = int(
            df[column]
            .astype("string")
            .str.strip()
            .eq("")
            .fillna(False)
            .sum()
        )

        report.append(
            f"{column}: missing={missing}, blank={blank}\n"
        )

    # --------------------------------------------------
    # 3. EXPECTED COLUMNS
    # --------------------------------------------------

    report.append(section("3. EXPECTED COLUMN CHECK"))

    for column in KEY_COLUMNS:
        status = (
            "PRESENT" if column in df.columns else "MISSING"
        )
        report.append(f"{column}: {status}\n")

    # --------------------------------------------------
    # 4. IMPORTANT VALUE DISTRIBUTIONS
    # --------------------------------------------------

    report.append(section("4. VALUE DISTRIBUTIONS"))

    distribution_columns = [
        "company",
        "document_id",
        "source_document",
        "fiscal_year",
        "standardized_fiscal_year",
        "scheduled_year",
        "currency",
        "standardized_currency",
        "scale",
        "standardized_scale",
        "unit",
        "standardized_unit",
        "statement_type",
        "category",
        "document_duplicate_status",
        "duplicate_status",
        "calculation_status",
        "review_flag",
        "review_flags",
    ]

    for column in distribution_columns:
        report.append(format_distribution(df, column))

    # --------------------------------------------------
    # 5. DUPLICATE DOCUMENT DIAGNOSTICS
    # --------------------------------------------------

    report.append(duplicate_document_diagnostics(df))

    # --------------------------------------------------
    # 6. EXACT DUPLICATE ROWS
    # --------------------------------------------------

    report.append(section("6. EXACT DUPLICATE ROWS"))

    exact_duplicates = int(df.duplicated().sum())

    report.append(
        f"Completely identical rows: {exact_duplicates}\n"
    )

    if "record_id" in df.columns:
        duplicated_ids = df[
            df["record_id"].duplicated(keep=False)
        ]

        report.append(
            f"Records with repeated record_id: "
            f"{len(duplicated_ids)}\n"
        )

    # --------------------------------------------------
    # 7. NUMERIC VALUE CHECKS
    # --------------------------------------------------

    report.append(section("7. NUMERIC VALUE CHECKS"))

    if "value" in df.columns:

        numeric_values = pd.to_numeric(
            df["value"], errors="coerce"
        )

        report.append(
            f"Numeric values: {numeric_values.notna().sum()}\n"
        )

        report.append(
            f"Non-numeric or missing values: "
            f"{numeric_values.isna().sum()}\n"
        )

        report.append(
            f"Zero values: {(numeric_values == 0).sum()}\n"
        )

        report.append(
            f"Negative values: {(numeric_values < 0).sum()}\n"
        )

        report.append(
            f"Positive values: {(numeric_values > 0).sum()}\n"
        )

        report.append(
            f"Minimum: {numeric_values.min()}\n"
        )

        report.append(
            f"Maximum: {numeric_values.max()}\n"
        )

    # --------------------------------------------------
    # 8. REVIEW FLAGS
    # --------------------------------------------------

    report.append(section("8. RECORDS REQUIRING REVIEW"))

    if "review_flag" in df.columns:

        flagged = df[
            df["review_flag"].notna()
            & df["review_flag"].astype(str).str.strip().ne("")
        ]

        report.append(
            f"Records with review_flag: {len(flagged)}\n"
        )

        report.append(
            format_sample(
                flagged,
                "FLAGGED RECORDS",
                SAMPLE_COLUMNS
            )
        )

    if "review_flags" in df.columns:

        flagged_list = df[
            df["review_flags"].notna()
            & df["review_flags"].astype(str).str.strip().ne("")
        ]

        report.append(
            f"Records with review_flags: {len(flagged_list)}\n"
        )

    # --------------------------------------------------
    # 9. MISSING PROVENANCE
    # --------------------------------------------------

    report.append(section("9. MISSING PROVENANCE"))

    provenance_columns = [
        "company",
        "document_id",
        "source_document",
        "page",
        "statement_type",
    ]

    for column in provenance_columns:

        if column not in df.columns:
            report.append(f"{column}: COLUMN MISSING\n")
            continue

        values = (
            df[column]
            .astype("string")
            .str.strip()
        )

        missing = (
            values.isna()
            | values.eq("")
            | values.str.upper().isin(
                ["NOT SPECIFIED", "UNKNOWN", "NONE", "N/A"]
            )
        )

        report.append(
            f"{column}: {int(missing.sum())} "
            f"records missing or unspecified\n"
        )

    # --------------------------------------------------
    # 10. POSSIBLY AMBIGUOUS FINANCIAL PERIODS
    # --------------------------------------------------

    report.append(
        section("10. POSSIBLY AMBIGUOUS FINANCIAL PERIODS")
    )

    if "fiscal_year" in df.columns:

        years = (
            df["fiscal_year"]
            .astype("string")
            .str.extract(r"(\d{4})", expand=False)
        )

        year_numbers = pd.to_numeric(
            years, errors="coerce"
        )

        unusual_years = df[
            year_numbers.notna()
            & (
                (year_numbers < 1990)
                | (year_numbers > 2026)
            )
        ]

        report.append(
            f"Years outside 1990–2026: "
            f"{len(unusual_years)}\n"
        )

        report.append(
            format_sample(
                unusual_years,
                "YEARS OUTSIDE EXPECTED RANGE",
                SAMPLE_COLUMNS
            )
        )

    # --------------------------------------------------
    # 11. UNUSUAL TEXT AND UNITS
    # --------------------------------------------------

    report.append(section("11. NONSTANDARD TEXT AND UNITS"))

    text_columns = [
        "currency",
        "standardized_currency",
        "unit",
        "standardized_unit",
        "scale",
        "standardized_scale",
        "statement_type",
    ]

    for column in text_columns:

        if column not in df.columns:
            continue

        values = df[column].dropna().astype(str)

        non_ascii = values[
            values.apply(
                lambda value: any(
                    ord(character) > 127
                    for character in value
                )
            )
        ]

        report.append(
            f"{column}: {len(non_ascii)} values containing "
            f"non-ASCII characters\n"
        )

        if not non_ascii.empty:
            report.append(
                "  Examples: "
                + ", ".join(
                    non_ascii.unique()[:10]
                )
                + "\n"
            )

    # --------------------------------------------------
    # 12. SAMPLE OF ALL RECORDS
    # --------------------------------------------------

    report.append(
        format_sample(
            df,
            "12. SAMPLE OF EXTRACTED AND CLEANED RECORDS",
            SAMPLE_COLUMNS
        )
    )

    # --------------------------------------------------
    # 13. FINAL REPORT
    # --------------------------------------------------

    report_text = "".join(report)

    output_path = csv_path.with_name(
        csv_path.stem + ".txt"
    )

    try:
        output_path.write_text(
            report_text,
            encoding="utf-8"
        )
    except Exception as error:
        print(f"\nCould not save report: {error}")

    print(report_text)

    print("\n" + "=" * 75)
    print(f"REPORT SAVED TO: {output_path}")
    print("=" * 75)


# --------------------------------------------------
# SCRIPT ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) > 1:
        file_path = " ".join(sys.argv[1:])
    else:
        file_path = input(
            "Enter the full path to your cleaned CSV file: "
        ).strip().strip('"')

    if file_path:
        inspect_csv(file_path)
    else:
        print("No file path provided.")
