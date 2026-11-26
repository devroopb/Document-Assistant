
import os
import json
import time
import random

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.5-flash-lite"


STATEMENT_HEADINGS = [
    "consolidated statements of operations",
    "consolidated statements of income",
    "consolidated income statements",
    "consolidated balance sheets",
    "consolidated statements of financial position",
    "consolidated statements of cash flows",
]

FINANCIAL_TERMS = [
    "revenue",
    "net sales",
    "cost of revenue",
    "cost of goods sold",
    "gross profit",
    "gross margin",
    "operating income",
    "income from operations",
    "net income",
    "current assets",
    "current liabilities",
    "total current assets",
    "total current liabilities",
    "total debt",
    "total borrowings",
    "total stockholders' equity",
    "total shareholders' equity",
    "total assets",
    "total liabilities",
    "cash and cash equivalents",
]


def select_financial_pages(pages, max_pages=12):
    """Identify pages likely to contain financial statements."""

    scored_pages = []

    for page in pages:
        text = " ".join(page["text"].lower().split())
        score = 0

        for heading in STATEMENT_HEADINGS:
            if heading in text:
                score += 5

        for term in FINANCIAL_TERMS:
            if term in text:
                score += 1

        if score > 0:
            scored_pages.append((score, page))

    scored_pages.sort(key=lambda item: item[0], reverse=True)

    selected_pages = [
        item[1] for item in scored_pages[:max_pages]
    ]

    selected_pages.sort(key=lambda page: page["page"])

    return selected_pages


def generate_with_retry(client, prompt, response_schema, max_attempts=5):
    """Retry Gemini requests when temporary service errors occur."""

    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    max_output_tokens=8000,
                    temperature=0.1,
                    automatic_continuation=False
                )
            )

        except Exception as error:
            error_code = getattr(error, "code", None)

            if error_code is None:
                error_code = getattr(error, "status_code", None)

            error_message = str(error).upper()

            retryable = (
                str(error_code) in {"429", "500", "502", "503", "504"}
                or any(code in error_message for code in [
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "UNAVAILABLE",
                    "RESOURCE_EXHAUSTED"
                ])
            )

            if not retryable or attempt == max_attempts - 1:
                raise

            delay = min(2 ** (attempt + 1), 30)
            delay += random.uniform(0, 1)

            time.sleep(delay)


def extract_financial_figures(pages):
    """Extract financial figures for Raw Extracted Data."""

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Check your .env file."
        )

    selected_pages = select_financial_pages(pages)

    if not selected_pages:
        raise ValueError(
            "No relevant financial statement pages were found."
        )

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "company": {"type": "STRING"},
            "currency": {"type": "STRING"},
            "scale": {"type": "STRING"},
            "statement_type": {"type": "STRING"},
            "figures": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "metric_name": {"type": "STRING"},
                        "fiscal_year": {"type": "STRING"},
                        "value": {"type": "NUMBER"},
                        "currency": {"type": "STRING"},
                        "scale": {"type": "STRING"},
                        "unit": {"type": "STRING"}
                    },
                    "required": [
                        "metric_name",
                        "fiscal_year",
                        "value",
                        "currency",
                        "scale",
                        "unit"
                    ]
                }
            }
        },
        "required": [
            "company",
            "currency",
            "scale",
            "statement_type",
            "figures"
        ]
    }

    client = genai.Client(api_key=api_key)

    all_figures = []
    company_name = "Unknown"

    for page in selected_pages:
        page_number = page["page"]

        prompt = f"""
        Extract the financial figures explicitly shown on this
        financial statement page.

        Instructions:
        - Extract every financial metric and fiscal year shown.
        - Preserve numerical values exactly as printed.
        - Identify the currency for each figure.
        - Identify the scale for each figure.
        - Identify the measurement unit for each figure.
        - Use USD millions for monetary amounts reported in millions.
        - Use USD per share for earnings per share.
        - Use shares for share counts.
        - Use percentages for percentage figures.
        - Do not assign monetary units to non-monetary figures.
        - Do not confuse gross margin amounts with percentages.
        - Preserve negative values.
        - Do not calculate, estimate, or invent missing figures.
        - Distinguish balance sheet balances from cash flow changes.
        - For cash flow changes, make the metric name explicit.
        - If information is unavailable, use "Not specified".
        - Return only figures supported by this page.

        DOCUMENT PAGE:
        {page["text"]}
        """

        response = generate_with_retry(
            client,
            prompt,
            response_schema
        )

        finish_reason = "Unknown"

        if response.candidates:
            finish_reason = response.candidates[0].finish_reason

        if not response.text:
            raise ValueError(
                f"Gemini returned an empty response for PDF page "
                f"{page_number}. Finish reason: {finish_reason}"
            )

        if "MAX_TOKENS" in str(finish_reason):
            raise ValueError(
                f"Gemini reached its output-token limit on PDF page "
                f"{page_number}."
            )

        page_data = json.loads(response.text)

        if page_data.get("company") not in (None, "", "Unknown"):
            company_name = page_data["company"]

        for figure in page_data.get("figures", []):
            figure["page"] = page_number
            figure["statement_type"] = page_data["statement_type"]
            all_figures.append(figure)

    if not all_figures:
        raise ValueError(
            "No financial figures could be extracted from the selected pages."
        )

    return {
        "company": company_name,
        "figures": all_figures
    }