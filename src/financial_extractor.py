
import os
import json
import time
import random

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load the Gemini API key from the project's .env file.
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Use the same Gemini model as the rest of the project.
MODEL_NAME = "gemini-3.5-flash-lite"


# Financial statement headings used to identify relevant PDF pages.
STATEMENT_HEADINGS = [
    "consolidated statements of operations",
    "consolidated statements of income",
    "consolidated balance sheets",
    "consolidated statements of cash flows",
]

# Common financial terms used to identify relevant pages.
FINANCIAL_TERMS = [
    "net sales",
    "total revenue",
    "net income",
    "operating income",
    "gross margin",
    "total assets",
    "total liabilities",
    "cash and cash equivalents",
]


def select_financial_pages(pages, max_pages=12):
    """
    Identify pages likely to contain financial statements.

    Pages are ranked using financial statement headings
    and commonly used financial terms.
    """

    scored_pages = []

    for page in pages:
        text = page["text"].lower()
        score = 0

        # Give more weight to pages containing statement headings.
        for heading in STATEMENT_HEADINGS:
            if heading in text:
                score += 5

        # Increase the score for pages containing financial terms.
        for term in FINANCIAL_TERMS:
            if term in text:
                score += 1

        if score > 0:
            scored_pages.append((score, page))

    # Select the highest-scoring pages.
    scored_pages.sort(key=lambda item: item[0], reverse=True)
    selected_pages = [
        item[1] for item in scored_pages[:max_pages]
    ]

    # Restore the original PDF page order.
    selected_pages.sort(key=lambda page: page["page"])

    return selected_pages

def generate_with_retry(client, prompt, response_schema, max_attempts=5):
    """
    Retry Gemini requests when temporary service errors occur.

    Uses exponential backoff to avoid repeatedly hitting
    an overloaded Gemini service.
    """

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

            # Only retry temporary server and rate-limit errors.
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

            # Immediately raise permanent errors.
            if not retryable or attempt == max_attempts - 1:
                raise

            # Wait progressively longer before retrying.
            delay = min(2 ** (attempt + 1), 30)
            delay += random.uniform(0, 1)

            time.sleep(delay)

def extract_financial_figures(pages):
    """
    Extract structured financial figures from a PDF.

    Each selected page is processed independently.
    Page-level information is stored once, while individual
    financial figures contain only their metric, year and value.
    """

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Check your .env file."
        )

    # Identify pages containing financial information.
    selected_pages = select_financial_pages(pages)

    if not selected_pages:
        raise ValueError(
            "No relevant financial statement pages were found."
        )

    # Define a compact response structure for Gemini.
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "company": {
                "type": "STRING"
            },
            "currency": {
                "type": "STRING"
            },
            "scale": {
                "type": "STRING"
            },
            "statement_type": {
                "type": "STRING"
            },
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

    # Create the Gemini client once for all page requests.
    client = genai.Client(api_key=api_key)

    all_figures = []
    company_name = "Unknown"

    # Process one page at a time.
    for page in selected_pages:

        page_number = page["page"]
        page_text = page["text"]

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
                - For cash flow changes, make the metric name explicit,
                for example "Change in Accounts Payable".
                - If information is unavailable, use "Not specified".
                - Return only figures supported by this page.

        DOCUMENT PAGE:
        {page_text}
        """

        response = generate_with_retry(
            client,
            prompt,
            response_schema
        )

        # Check whether Gemini completed its response.
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
                f"{page_number}. The page may contain too many figures."
            )

        # Convert the JSON response into a Python dictionary.
        try:
            page_data = json.loads(response.text)

        except json.JSONDecodeError as error:
            raise ValueError(
                f"Gemini returned invalid JSON for PDF page "
                f"{page_number}."
            ) from error

        # Save the company name when available.
        if page_data.get("company") not in (None, "", "Unknown"):
            company_name = page_data["company"]

        # Attach page-level information to each extracted figure.
        for figure in page_data.get("figures", []):

            # Attach the original PDF page and statement information.
            figure["page"] = page_number
            figure["statement_type"] = page_data["statement_type"]

            all_figures.append(figure)

    # Return the same overall structure expected by app.py.
    if not all_figures:
        raise ValueError(
            "No financial figures could be extracted from the selected pages."
        )

    return {
        "company": company_name,
        "figures": all_figures
    }