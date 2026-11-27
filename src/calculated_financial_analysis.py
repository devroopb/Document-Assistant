
import os
import json
import re
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.embedding_search import search_documents


load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"


FINANCIAL_COMPONENTS = {
    "Revenue": "What was the total revenue or net sales?",
    "Cost of Goods Sold": "What was the cost of goods sold, cost of sales, or cost of revenue?",
    "Operating Income": "What was the operating income or income from operations?",
    "Net Income": "What was the net income?",
    "Shareholders' Equity": "What was total shareholders' equity or total stockholders' equity?",
    "Current Assets": "What were total current assets?",
    "Current Liabilities": "What were total current liabilities?",
    "Inventory": "What was the total inventory?",
    "Total Liabilities": "What were total liabilities?",
    "Interest Expense": "What was the interest expense?",
    "Operating Cash Flow": "What was net cash provided by or generated from operating activities?",
    "Capital Expenditures": "What were capital expenditures or purchases of property, plant and equipment?",
    "Total Assets": "What were total assets?",
}


STATEMENT_TYPES = {
    "Revenue": "Income Statement",
    "Cost of Goods Sold": "Income Statement",
    "Operating Income": "Income Statement",
    "Net Income": "Income Statement",
    "Shareholders' Equity": "Balance Sheet",
    "Current Assets": "Balance Sheet",
    "Current Liabilities": "Balance Sheet",
    "Inventory": "Balance Sheet",
    "Total Liabilities": "Balance Sheet",
    "Interest Expense": "Income Statement",
    "Operating Cash Flow": "Cash Flow Statement",
    "Capital Expenditures": "Cash Flow Statement",
    "Total Assets": "Balance Sheet",
}


SCALE_MULTIPLIERS = {
    "unit": 1,
    "units": 1,
    "ones": 1,
    "thousand": 1_000,
    "thousands": 1_000,
    "million": 1_000_000,
    "millions": 1_000_000,
    "billion": 1_000_000_000,
    "billions": 1_000_000_000,
}


# --------------------------------------------------
# COMPANY NAME NORMALISATION
# --------------------------------------------------

def _normalise_company(company):
    """
    Standardise company names by removing common legal suffixes.

    Examples:
        Tesla, Inc.       -> Tesla
        TESLA INC         -> Tesla
        Tesla Incorporated -> Tesla
    """

    if not company:
        return "Unknown Company"

    name = str(company).strip()

    # Remove punctuation and standardise whitespace.
    name = re.sub(r"[.,]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    suffixes = {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "ltd",
        "limited",
        "llc",
        "plc",
        "lp",
        "llp",
    }

    words = name.split()

    # Remove legal suffixes from the end of the company name.
    while words and words[-1].lower() in suffixes:
        words.pop()

    name = " ".join(words).strip()

    if not name:
        return "Unknown Company"

    # Apply consistent display formatting.
    if name.lower() == "tesla":
        return "Tesla"

    return name.title()


# --------------------------------------------------
# VALUE NORMALISATION
# --------------------------------------------------

def _normalise_scale(scale):
    """Convert scale descriptions into a standard multiplier."""

    scale = str(scale or "").strip().lower()
    scale = re.sub(r"[^a-z ]", " ", scale)
    words = set(scale.split())

    for name, multiplier in SCALE_MULTIPLIERS.items():
        if name in words:
            return multiplier

    return None


def _normalise_number(value):
    """Convert a financial value into a Python float."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()

    if not value or value.lower() in {
        "not specified",
        "unknown",
        "n/a",
        "none",
    }:
        return None

    negative = value.startswith("(") and value.endswith(")")

    value = (
        value.strip("()")
        .replace(",", "")
        .replace("$", "")
        .strip()
    )

    try:
        number = float(value)
        return -number if negative else number

    except ValueError:
        return None


# --------------------------------------------------
# GEMINI RATE-LIMIT HANDLING
# --------------------------------------------------

def _generate_with_retry(client, prompt):
    """
    Send a Gemini request with limited retry handling.

    A token-per-minute quota error may require waiting for the
    next quota window before retrying.
    """

    max_attempts = 3

    for attempt in range(max_attempts):

        try:
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=3000,
                ),
            )

        except Exception as error:

            error_text = str(error)

            is_rate_limit = (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            )

            if not is_rate_limit or attempt == max_attempts - 1:
                raise

            # Wait for a full minute to allow the token-per-minute
            # quota window to reset.
            wait_seconds = 60

            print(
                f"Gemini rate limit reached. "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    return None


# --------------------------------------------------
# RAG RETRIEVAL AND EXTRACTION
# --------------------------------------------------

def _retrieve_component(component, question, model, index, chunks, client):

    search_results = search_documents(
        question,
        model,
        index,
        chunks,
        top_k=20,
    )

    if not search_results:
        return []

    context_parts = []

    for result in search_results:

        context_parts.append(
            f"\nDocument: {result.get('document', 'Unknown')}\n"
            f"Page: {result.get('page', 'Unknown')}\n"
            f"{result.get('text', '')}\n"
        )

    context = "\n".join(context_parts)

    statement_type = STATEMENT_TYPES[component]

    prompt = f"""
You are extracting financial figures from business documents.

Extract the financial component: {component}

Expected financial statement: {statement_type}

Question:
{question}

Instructions:
- Use ONLY the supplied document passages.
- Extract every relevant fiscal year available in the passages.
- Return the actual reported numeric value, not a calculated value.
- Preserve the reported currency and scale.
- Do not combine figures from different companies.
- Do not confuse quarterly figures with annual figures.
- Do not confuse subtotals with the requested component.
- Do not invent missing figures.
- If a figure is not available, omit it.
- Prefer annual figures from the appropriate financial statement.
- For balance sheet components, use figures as of the fiscal year-end.
- For income statement and cash flow components, use figures for the full fiscal year.
- Do not use quarterly or year-to-date figures as annual figures.
- When comparative years are shown, extract each year's corresponding value.
- Do not mix figures from different reporting periods.
- If the document reports figures in thousands or millions, preserve that scale.
- Return valid JSON only.

Use this exact JSON structure:

{{
    "figures": [
        {{
            "company": "Company name",
            "fiscal_year": "2025",
            "value": 123.45,
            "currency": "USD",
            "scale": "million"
        }}
    ]
}}

DOCUMENT PASSAGES:
{context}
"""

    response = _generate_with_retry(client, prompt)

    if response is None or not response.text:
        return []

    try:
        extracted = json.loads(response.text)

    except (json.JSONDecodeError, TypeError):
        return []

    figures = []

    for figure in extracted.get("figures", []):

        value = _normalise_number(figure.get("value"))
        scale = _normalise_scale(figure.get("scale"))

        fiscal_year = str(figure.get("fiscal_year", "")).strip()
        currency = str(figure.get("currency", "")).strip().upper()
        company = _normalise_company(figure.get("company"))

        year_match = re.search(r"\b(19|20)\d{2}\b", fiscal_year)

        if (
            value is None
            or scale is None
            or not year_match
            or not currency
            or company == "Unknown Company"
        ):
            continue

        figures.append({
            "component": component,
            "company": company,
            "fiscal_year": year_match.group(0),
            "value": value * scale,
            "currency": currency,
        })

    return figures


# --------------------------------------------------
# FIGURE SELECTION
# --------------------------------------------------

def _get_value(records, component, company, year, currency):

    matches = [
        record["value"]
        for record in records
        if record["component"] == component
        and record["company"] == company
        and record["fiscal_year"] == year
        and record["currency"] == currency
    ]

    unique_values = set(matches)

    if len(unique_values) != 1:
        return None

    return unique_values.pop()


# --------------------------------------------------
# FINANCIAL CALCULATIONS
# --------------------------------------------------

def _calculate_for_year(records, company, year, currency):

    try:
        previous_year = str(int(year) - 1)

    except ValueError:
        previous_year = None

    def get(component, target_year=year):

        return _get_value(
            records,
            component,
            company,
            target_year,
            currency,
        )

    revenue = get("Revenue")
    cogs = get("Cost of Goods Sold")
    operating_income = get("Operating Income")
    net_income = get("Net Income")
    equity = get("Shareholders' Equity")
    current_assets = get("Current Assets")
    current_liabilities = get("Current Liabilities")
    inventory = get("Inventory")
    liabilities = get("Total Liabilities")
    interest_expense = get("Interest Expense")
    operating_cash_flow = get("Operating Cash Flow")
    capital_expenditures = get("Capital Expenditures")
    total_assets = get("Total Assets")

    previous_equity = (
        get("Shareholders' Equity", previous_year)
        if previous_year else None
    )

    previous_assets = (
        get("Total Assets", previous_year)
        if previous_year else None
    )

    results = []

    def add_result(name, value, unit):

        results.append({
            "Company": company,
            "Fiscal Year": year,
            "Metric": name,
            "Calculated Value": (
                round(value, 2) if value is not None else "Not available"
            ),
            "Unit": unit,
            "Currency": currency if unit == "Currency" else "",
        })

    # 1. Gross Profit Margin
    value = None

    if revenue is not None and cogs is not None and revenue != 0:
        value = ((revenue - cogs) / revenue) * 100

    add_result("Gross Profit Margin", value, "%")

    # 2. Operating Profit Margin
    value = None

    if (
        revenue is not None
        and operating_income is not None
        and revenue != 0
    ):
        value = (operating_income / revenue) * 100

    add_result("Operating Profit Margin", value, "%")

    # 3. Net Profit Margin
    value = None

    if revenue is not None and net_income is not None and revenue != 0:
        value = (net_income / revenue) * 100

    add_result("Net Profit Margin", value, "%")

    # 4. Return on Equity
    value = None

    if (
        net_income is not None
        and equity is not None
        and previous_equity is not None
    ):
        average_equity = (equity + previous_equity) / 2

        if average_equity != 0:
            value = (net_income / average_equity) * 100

    add_result("Return on Equity (ROE)", value, "%")

    # 5. Current Ratio
    value = None

    if current_assets is not None and current_liabilities not in (None, 0):
        value = current_assets / current_liabilities

    add_result("Current Ratio", value, "x")

    # 6. Quick Ratio
    value = None

    if (
        current_assets is not None
        and inventory is not None
        and current_liabilities not in (None, 0)
    ):
        value = (current_assets - inventory) / current_liabilities

    add_result("Quick Ratio", value, "x")

    # 7. Total Liabilities-to-Equity Ratio
    value = None

    if liabilities is not None and equity not in (None, 0):
        value = liabilities / equity

    add_result("Debt-to-Equity Ratio", value, "x")

    # 8. Interest Coverage Ratio
    value = None

    if operating_income is not None and interest_expense not in (None, 0):
        value = operating_income / abs(interest_expense)

    add_result("Interest Coverage Ratio", value, "x")

    # 9. Free Cash Flow
    value = None

    if operating_cash_flow is not None and capital_expenditures is not None:
        value = operating_cash_flow - abs(capital_expenditures)

    add_result("Free Cash Flow", value, "Currency")

    # 10. Asset Turnover Ratio
    value = None

    if (
        revenue is not None
        and total_assets is not None
        and previous_assets is not None
    ):
        average_assets = (total_assets + previous_assets) / 2

        if average_assets != 0:
            value = revenue / average_assets

    add_result("Asset Turnover Ratio", value, "x")

    return results


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def calculate_metrics_with_rag(model, index, chunks):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Check your .env file."
        )

    client = genai.Client(api_key=api_key)

    all_records = []

    for component, question in FINANCIAL_COMPONENTS.items():

        figures = _retrieve_component(
            component,
            question,
            model,
            index,
            chunks,
            client,
        )

        all_records.extend(figures)

    if not all_records:
        return pd.DataFrame()

    combinations = {
        (
            record["company"],
            record["fiscal_year"],
            record["currency"],
        )
        for record in all_records
    }

    calculated_results = []

    for company, year, currency in sorted(combinations):

        calculated_results.extend(
            _calculate_for_year(
                all_records,
                company,
                year,
                currency,
            )
        )

    return pd.DataFrame(calculated_results)
