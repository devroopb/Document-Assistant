
import streamlit as st
import hashlib
import time
import pandas as pd

from src.document_processor import extract_text_from_pdf
from src.text_chunker import split_into_chunks
from src.embedding_search import create_embeddings, search_documents
from src.answer_generator import generate_answer
from src.financial_extractor import extract_financial_figures
from src.clean_tab_financial_analysis import create_financial_analysis_dataframe


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Business Document Assistant",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# 2. APPLICATION HEADER
# --------------------------------------------------

st.title("Business Document Assistant")

st.caption(
    "Your workspace for business documents and evidence-based insights."
)

st.divider()


# --------------------------------------------------
# 3. DOCUMENT UPLOAD SECTION
# --------------------------------------------------

st.subheader("Document Workspace")

st.write("Upload one or more PDFs to begin analysing their contents.")

uploaded_files = st.file_uploader(
    "Choose business documents",
    type=["pdf"],
    accept_multiple_files=True
)


# --------------------------------------------------
# 4. PDF PROCESSING AND SESSION STATE
# --------------------------------------------------

if uploaded_files:

    # Generate a fingerprint to detect changes in uploaded documents.
    file_signature = hashlib.sha256(
        b"".join(
            uploaded_file.name.encode("utf-8")
            + hashlib.sha256(uploaded_file.getvalue()).digest()
            for uploaded_file in uploaded_files
        )
    ).hexdigest()

    # Process documents only when the uploaded files change.
    if st.session_state.get("file_signature") != file_signature:

        all_chunks = []
        document_pages = []
        processing_errors = []

        for uploaded_file in uploaded_files:

            try:
                # Extract text from the current PDF.
                pages = extract_text_from_pdf(uploaded_file)

                document_pages.append({
                    "name": uploaded_file.name,
                    "pages": pages
                })

                # Split the document into smaller chunks.
                chunks = split_into_chunks(
                    pages,
                    chunk_size=1000,
                    overlap=200
                )

                # Preserve the source filename in every chunk.
                for chunk in chunks:
                    chunk["document"] = uploaded_file.name

                all_chunks.extend(chunks)

            except Exception as error:
                processing_errors.append(
                    f"Unable to process {uploaded_file.name}: {error}"
                )

        # Save processed documents and chunks.
        st.session_state.file_signature = file_signature
        st.session_state.document_pages = document_pages
        st.session_state.all_chunks = all_chunks
        st.session_state.processing_errors = processing_errors

        # Clear financial results when the uploaded document set changes.
        st.session_state.financial_results = {}

        # Generate one shared FAISS index.
        if all_chunks:

            with st.spinner("Generating shared document embeddings..."):
                model, index = create_embeddings(all_chunks)

            st.session_state.embedding_model = model
            st.session_state.embedding_index = index

        else:
            st.session_state.embedding_model = None
            st.session_state.embedding_index = None

    # Retrieve stored data rather than processing the PDFs again.
    document_pages = st.session_state.document_pages
    all_chunks = st.session_state.all_chunks
    model = st.session_state.embedding_model
    index = st.session_state.embedding_index

    # Display processing errors, if any.
    for error in st.session_state.processing_errors:
        st.error(error)

    if all_chunks:

        st.success("Documents processed and shared index ready!")

        st.write(f"**Documents processed:** {len(document_pages)}")
        st.write(f"**Embeddings stored in FAISS:** {index.ntotal}")
        st.write(f"**Total text chunks:** {len(all_chunks)}")


        # --------------------------------------------------
        # 5. DOCUMENT QUESTION ANSWERING
        # --------------------------------------------------

        with st.form("document_question_form"):

            question = st.text_input(
                "Ask a question across your documents",
                placeholder="e.g. Compare the companies' total revenue."
            )

            submitted = st.form_submit_button("Ask Question")

        # Generate an answer only when the form is submitted.
        if submitted and question.strip():

            search_results = search_documents(
                question,
                model,
                index,
                all_chunks,
                top_k=20
            )

            start_time = time.perf_counter()

            try:
                with st.spinner("Generating answer..."):
                    answer = generate_answer(question, search_results)

                response_time = time.perf_counter() - start_time

                st.caption(
                    f"Gemini response time: {response_time:.2f} seconds"
                )

                st.subheader("AI-Generated Answer")
                st.write(answer)

                st.subheader("Relevant Document Passages")

                for number, result in enumerate(search_results, start=1):

                    with st.expander(
                        f"Result {number} — {result['document']} "
                        f"— Page {result['page']} "
                        f"(Similarity: {result['score']:.3f})"
                    ):
                        st.write(result["text"])

            except Exception as error:
                st.error(
                    f"Unable to generate an answer at the moment: {error}"
                )


        # --------------------------------------------------
        # 6. FINANCIAL DATA EXTRACTION
        # --------------------------------------------------

        st.divider()
        st.subheader("Financial Data Extraction")

        selected_document_index = st.selectbox(
            "Choose a document for financial extraction",
            options=range(len(document_pages)),
            format_func=lambda i: document_pages[i]["name"]
        )

        selected_document = document_pages[selected_document_index]
        selected_document_name = selected_document["name"]
        selected_pages = selected_document["pages"]

        # Initialise the financial results cache.
        if "financial_results" not in st.session_state:
            st.session_state.financial_results = {}

        # Extract figures only when requested and not already cached.
        if st.button("Extract Financial Figures"):

            if selected_document_name not in st.session_state.financial_results:

                try:
                    with st.spinner("Extracting financial figures with Gemini..."):

                        financial_data = extract_financial_figures(
                            selected_pages
                        )

                    st.session_state.financial_results[
                        selected_document_name
                    ] = financial_data

                    st.success("Financial figures extracted successfully!")

                except Exception as error:
                    st.error(
                        f"Financial extraction failed: {error}"
                    )

            else:
                st.info("Using previously extracted financial data.")

        # Display cached results for the selected document.
        if selected_document_name in st.session_state.financial_results:

            financial_data = st.session_state.financial_results[
                selected_document_name
            ]

            financial_df = pd.DataFrame(
                financial_data["figures"]
            )

            if not financial_df.empty:

                financial_df = financial_df.sort_values(
                    by=["fiscal_year", "metric_name"],
                    ascending=[False, True]
                ).reset_index(drop=True)

                raw_tab, cleaned_tab = st.tabs([
                    "Raw Extracted Data",
                    "Cleaned Financial Data"
                ])

                with raw_tab:

                    st.subheader("Raw Financial Data")

                    st.dataframe(
                        financial_df,
                        use_container_width=True,
                        hide_index=True
                    )

                with cleaned_tab:

                    st.subheader("Cleaned Financial Data")

                    analysis_df = create_financial_analysis_dataframe(
                        financial_data["figures"]
                    )

                    flagged_count = (
                        analysis_df["review_flag"].ne("").sum()
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Financial Figures",
                            len(analysis_df)
                        )

                    with col2:
                        st.metric(
                            "Figures Requiring Review",
                            int(flagged_count)
                        )

                    st.dataframe(
                        analysis_df,
                        use_container_width=True,
                        hide_index=True
                    )

            else:
                st.info("No financial figures are available to display.")


        # --------------------------------------------------
        # 7. DOCUMENT PREVIEWS
        # --------------------------------------------------

        with st.expander("Preview text chunks"):

            for chunk_number, chunk in enumerate(all_chunks, start=1):

                st.markdown(
                    f"**Chunk {chunk_number} — "
                    f"{chunk['document']} — Page {chunk['page']}**"
                )

                st.write(chunk["text"])
                st.divider()

        with st.expander("Preview extracted text"):

            for document in document_pages:

                st.markdown(f"### {document['name']}")

                st.write(
                    f"**Pages containing text:** {len(document['pages'])}"
                )

                for page in document["pages"]:

                    st.markdown(f"**Page {page['page']}**")
                    st.write(page["text"][:1500])
                    st.divider()

    else:
        st.warning(
            "No extractable text was found in the uploaded PDFs."
        )