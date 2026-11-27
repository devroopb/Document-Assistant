# Business Document Assistant

An AI-powered Streamlit application for analysing PDF documents, answering questions using document content, and extracting structured financial data. The application combines retrieval-augmented generation (RAG), Gemini-powered language models, and Python-based financial calculations.

## Features

### Document Question Answering (RAG)

- PDF text extraction using PyPDF.
- Text chunking with overlap to preserve context.
- Local embeddings using Sentence Transformers.
- FAISS semantic search across uploaded documents.
- Gemini-powered answers grounded in retrieved document passages.
- Document names and page references for supporting passages.
- Collapsible **Relevant Document Passages** section beneath each answer.
- Insufficient-evidence handling when the retrieved content does not support an answer.
- Multilingual question answering, including questions about financial information in foreign-language documents.

### Duplicate PDF Detection

- SHA-256 content hashing identifies byte-identical PDF files, regardless of filename.
- The first uploaded copy is treated as the original.
- Subsequent exact copies are excluded from text chunking, embedding generation, RAG retrieval, financial extraction, and calculated financial metrics.
- The application identifies the duplicate filename and names the original file it matches.
- Duplicate financial-record detection is also available for extracted financial data.

> Exact-file duplicate detection is based on file contents. A PDF that has been re-saved or modified may have a different hash even when its pages appear identical.

### Financial Data Extraction

- AI-powered financial figure extraction using Gemini.
- Extraction of financial metrics, fiscal years, values, currencies, scales, and units.
- Raw extracted financial data displayed separately from cleaned financial data.
- Financial statement classification.
- Data cleaning, standardisation, and quality checks using Pandas.
- Duplicate and potentially conflicting financial-record identification.
- Primarily supports common US and Canadian financial reporting terminology and conventions.

### Calculated Financial Metrics

- Retrieves financial components from the shared RAG document index.
- Uses Python to calculate financial ratios and measures from retrieved figures.
- Supports calculations such as profitability, liquidity, leverage, interest coverage, free cash flow, and asset turnover metrics.
- Groups calculated results by company and fiscal year.
- Displays the number of companies, fiscal years, and metrics calculated.
- Keeps the retrieved calculation components internal to the calculation workflow; the interface displays the calculated results rather than a separate component table.
- Includes a 60-second cooldown after the financial extraction action before the calculation action becomes available, to help separate Gemini API requests.

Calculated results should be independently cross-checked against the AI-generated answers and the source document passages. Retrieval and calculation can use different passages, so an insufficient-evidence answer and an available calculated result should be investigated rather than assumed to agree.

## Technology Stack

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Google Gemini API
- Pandas
- NumPy

## Project Structure

```text
Document Assistant/
├── app.py
├── src/
│   ├── document_processor.py
│   ├── text_chunker.py
│   ├── embedding_search.py
│   ├── answer_generator.py
│   ├── financial_extractor.py
│   ├── clean_tab_financial_analysis.py
│   ├── duplicate_detector.py
│   └── calculated_financial_analysis.py
├── test_files/
├── .streamlit/
├── .env
├── .gitignore
└── requirements.txt
```

## Installation

The following commands use Windows PowerShell.

```powershell
git clone https://github.com/devroopb/Document-Assitant.git
cd Document-Assitant

python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Create a `.env` file in the project root and add your Gemini API key:

```text
GEMINI_API_KEY=your_api_key_here
```

Keep API keys private. Do not commit `.env` or other files containing credentials to GitHub.

Run the application:

```powershell
streamlit run app.py
```

## Using the Application

1. Upload one or more PDF documents.
2. The application calculates a content hash for each PDF and excludes exact duplicate copies from analysis.
3. Ask questions about the unique uploaded documents in the Document Question Answering section.
4. Expand **Relevant Document Passages** beneath an answer to inspect the retrieved source passages, page numbers, and similarity scores.
5. Select **Extract Financial Figures from Original Documents** to extract financial data from the unique documents.
6. Review the Raw Extracted Data, Cleaned Financial Data, and Duplicate Data tabs.
7. After the 60-second cooldown, select **Calculate Financial Metrics** in the Calculated Financial Metrics tab.
8. Cross-check calculated results against AI-generated answers and the cited source passages.

## Testing

Test documents and cases include:

- Apple 10-K annual report.
- Tesla 10-K annual report.
- Roadzen Inc annual report (small US company).
- Hong Kong Stock Exchange financial report.
- Japanese IBM user manual.
- Harry Potter and the Deathly Hallows (large non-financial document).
- Corrupted PDF.
- Password-protected/encrypted PDF.
- Handwritten OCR PDF.
- Plain-text files.
- Non-PDF files renamed with a `.pdf` extension.
- Duplicate copies of PDFs uploaded under different filenames.

These are test cases used during development; support and results can vary depending on document structure, text extractability, language, and reporting conventions.

## Limitations

- Multilingual RAG supports cross-language question answering, but retrieval accuracy may vary by language and document.
- Financial statement detection and structured extraction primarily target common US and Canadian annual-report terminology and conventions. Other countries' accounting terminology and reporting formats may not be reliably supported.
- Scanned and handwritten documents require OCR, which is not currently implemented.
- Exact duplicate PDF detection identifies byte-identical files. Visually identical PDFs with different underlying bytes may not be recognised as duplicates.
- RAG retrieval may return different passages for general question answering and individual financial-component lookups. Calculated results therefore require verification against the source documents.
- Gemini API rate limits and quotas may affect extraction, question answering, or metric calculations, particularly when processing large documents or making multiple requests.
