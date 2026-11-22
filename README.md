# Business Document Assistant

An AI-powered application for analysing PDF documents, answering questions using document content, and extracting structured financial data.

## Features

### Document Question Answering (RAG)
- PDF text extraction using PyPDF.
- Text chunking with overlap.
- Local embeddings using Sentence Transformers.
- FAISS semantic search.
- Gemini-powered answers with page references.
- Insufficient-evidence handling.
- **Multilingual question answering:** Ask questions in different languages about document content, including questions about financial information such as total sales in foreign-language documents.

### Financial Data Extraction
- AI-powered financial figure extraction using Gemini.
- Extraction of metrics, fiscal years, values, currencies, scales and units.
- Financial statement classification.
- Data cleaning and quality checks using Pandas.
- Detection and extraction primarily support common US and Canadian financial reporting terminology and conventions.

### Technology Stack
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

Test documents include:
- Apple annual report (10-K).
- Tesla annual report (10-K).
- Red Metal Limited annual report.
- Harry Potter and the Deathly Hallows.
- Japanese Bible.
- Corrupted, encrypted, scanned, handwritten, large and invalid PDF files.

## Limitations

- Multilingual RAG has demonstrated cross-language question answering, but retrieval accuracy may vary by language and document.
- Financial statement detection and structured extraction primarily target common US and Canadian annual-report terminology and reporting conventions. Other countries' accounting terminology and formats may not be reliably supported.
- Scanned documents require further OCR evaluation.