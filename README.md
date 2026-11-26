# Business Document Assistant

An AI-powered application for analysing PDF documents, answering questions using document content, and extracting structured financial data.

## Features

### Document Question Answering (RAG)
- PDF text extraction using PyPDF.
- Text chunking with overlap.
- Local embeddings using Sentence Transformers.
- FAISS semantic search.
- Gemini-powered answers with document and page references.
- Insufficient-evidence handling.
- Multilingual question answering, including questions about financial information in foreign-language documents.

### Financial Data Extraction
- AI-powered financial figure extraction using Gemini.
- Extraction of metrics, fiscal years, values, currencies, scales and units.
- Financial statement classification.
- Data cleaning and quality checks using Pandas.
- Primarily supports common US and Canadian financial reporting terminology and conventions.

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
│   └── clean_tab_financial_analysis.py
├── test_files/
├── .streamlit/
├── .env
├── .gitignore
└── requirements.txt
```

## Installation

```powershell
git clone https://github.com/devroopb/Document-Assitant.git
cd Document-Assitant

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Create a `.env` file and add:

```text
GEMINI_API_KEY=your_api_key_here
```

Run the application:

```powershell
streamlit run app.py
```

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

## Limitations

- Multilingual RAG supports cross-language question answering, but retrieval accuracy may vary by language and document.
- Financial statement detection and structured extraction primarily target common US and Canadian annual-report terminology and conventions. Other countries' accounting terminology and reporting formats may not be reliably supported.
- Scanned and handwritten documents require OCR, which is not currently implemented.