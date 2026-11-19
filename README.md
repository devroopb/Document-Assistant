# Document Assistant

An AI-powered business document analysis application built with Python and Streamlit. It allows users to upload PDF documents, extract their contents, and retrieve relevant information through semantic search.

## Features

- **PDF Processing:** Extracts text from PDF documents while preserving page numbers.
- **Text Chunking:** Splits extracted content into smaller sections for efficient retrieval.
- **Semantic Search:** Uses Sentence Transformers to generate text embeddings.
- **FAISS Retrieval:** Finds the most relevant document passages based on user queries.
- **Source References:** Displays relevant passages alongside their original PDF page numbers.

## Tech Stack

- Python
- Streamlit
- Pandas
- PyPDF
- Sentence Transformers
- FAISS
- NumPy

## Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/devroopb/Document-Assitant.git
   cd Document-Assitant
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

5. Upload a PDF document and enter a question to retrieve relevant passages.

## Sample Document

A sample PDF is available in the `test_files/` directory for testing the application.

## Future Development

- AI-generated answers with document citations.
- Financial and operational data extraction.
- Business insights, calculations, and visualisations.
- Multi-document analysis and integration with a larger Business Intelligence platform.

## Author

**Devroop Banerjee**

[GitHub](https://github.com/devroopb)