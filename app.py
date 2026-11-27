
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
from src.duplicate_detector import detect_duplicates
from src.calculated_financial_analysis import calculate_metrics_with_rag


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Business Document Assistant",
    page_icon="📄",
    layout="wide",
)


# --------------------------------------------------
# 2. CALCULATED METRICS BUTTON AND COOLDOWN
# --------------------------------------------------

@st.fragment(run_every=1)
def render_calculated_metrics():

    st.subheader("Calculated Financial Metrics")

    st.write(
        "Financial figures are independently retrieved from "
        "your uploaded documents using RAG. Python then "
        "calculates the financial metrics.\n\n"
        "All calculated metrics are cross checked with the AI-Generated answers."
    )

    extraction_time = st.session_state.get(
        "financial_extraction_clicked_at"
    )

    remaining = 0

    if extraction_time is not None:
        elapsed = time.time() - extraction_time
        remaining = max(0, 60 - int(elapsed))

    if remaining > 0:

        st.info(
            f"Please wait {remaining} seconds before calculating "
            "financial metrics. This cooldown helps separate "
            "the financial extraction and calculation requests "
            "so that the token limit per minute is not exceeded."
        )

    elif extraction_time is not None:

        st.success("Cooldown complete. You can now calculate metrics.")

    if st.button(
        "Calculate Financial Metrics",
        key="calculate_metrics_button",
        disabled=remaining > 0,
    ):

        st.session_state.calculated_metrics = None

        try:
            with st.spinner(
                "Retrieving financial figures and calculating metrics..."
            ):

                calculated_df = calculate_metrics_with_rag(
                    st.session_state.embedding_model,
                    st.session_state.embedding_index,
                    st.session_state.all_chunks,
                )

            st.session_state.calculated_metrics = calculated_df

        except Exception as error:
            st.error(
                f"Unable to calculate financial metrics: {error}"
            )

    calculated_df = st.session_state.get("calculated_metrics")

    if calculated_df is None:

        st.info(
            "Click 'Calculate Financial Metrics' when the cooldown "
            "has finished."
        )

    elif calculated_df.empty:

        st.warning(
            "No financial figures could be retrieved for calculation. "
            "Check that the uploaded PDFs contain readable financial "
            "statements and try again."
        )

    else:

        st.success("Financial metrics calculated successfully.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Companies",
                calculated_df["Company"].nunique(),
            )

        with col2:
            st.metric(
                "Fiscal Years",
                calculated_df["Fiscal Year"].nunique(),
            )

        with col3:
            st.metric(
                "Calculated Metrics",
                len(calculated_df),
            )

        st.divider()

        for (company, fiscal_year), year_df in calculated_df.groupby(
            ["Company", "Fiscal Year"],
            sort=True,
        ):

            st.markdown(f"### {company} — FY {fiscal_year}")

            display_df = year_df[
                [
                    "Metric",
                    "Calculated Value",
                    "Unit",
                    "Currency",
                ]
            ].reset_index(drop=True)

            # Keep the calculated-value column numeric for Arrow/Streamlit.
            # Unavailable metrics remain blank rather than mixing strings and numbers.
            display_df["Calculated Value"] = pd.to_numeric(
                display_df["Calculated Value"],
                errors="coerce",
            )

            st.caption("A blank calculated value means the required figures were unavailable.")

            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True,
            )


# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.title("Business Document Assistant")

st.caption(
    "Your workspace for business documents and evidence-based insights."
)

st.divider()


# --------------------------------------------------
# 4. DOCUMENT UPLOAD
# --------------------------------------------------

st.subheader("Document Workspace")

st.write("Upload one or more PDFs to begin analysing their contents.")

uploaded_files = st.file_uploader(
    "Choose business documents",
    type=["pdf"],
    accept_multiple_files=True,
)


# --------------------------------------------------
# 5. SESSION STATE INITIALISATION
# --------------------------------------------------

if "calculated_metrics" not in st.session_state:
    st.session_state.calculated_metrics = None

if "financial_results" not in st.session_state:
    st.session_state.financial_results = {}


# --------------------------------------------------
# 6. PDF PROCESSING
# --------------------------------------------------

if uploaded_files:

    file_signature = hashlib.sha256(
        b"".join(
            uploaded_file.name.encode("utf-8")
            + hashlib.sha256(uploaded_file.getvalue()).digest()
            for uploaded_file in uploaded_files
        )
    ).hexdigest()

    if st.session_state.get("file_signature") != file_signature:

        all_chunks = []
        document_pages = []
        unique_document_pages = []
        duplicate_documents = []
        processing_errors = []
        original_by_hash = {}

        for uploaded_file in uploaded_files:

            try:
                file_bytes = uploaded_file.getvalue()

                # This SHA-256 content hash is the document_id used by the
                # duplicate-detection workflow. It is independent of filename.
                document_id = hashlib.sha256(file_bytes).hexdigest()

                if document_id in original_by_hash:
                    original_name = original_by_hash[document_id]

                    document_pages.append({
                        "name": uploaded_file.name,
                        "document_id": document_id,
                        "pages": [],
                        "is_duplicate": True,
                        "duplicate_of": original_name,
                    })

                    duplicate_documents.append({
                        "Duplicate File": uploaded_file.name,
                        "Original File": original_name,
                        "Status": "Excluded from analysis — exact duplicate",
                    })

                    # Do not chunk, embed, extract, or calculate from this copy.
                    continue

                # Only extract the PDF text after it passes the content-hash check.
                pages = extract_text_from_pdf(uploaded_file)
                original_by_hash[document_id] = uploaded_file.name

                document = {
                    "name": uploaded_file.name,
                    "document_id": document_id,
                    "pages": pages,
                    "is_duplicate": False,
                    "duplicate_of": None,
                }

                document_pages.append(document)
                unique_document_pages.append(document)

                chunks = split_into_chunks(
                    pages,
                    chunk_size=1000,
                    overlap=200,
                )

                for chunk in chunks:
                    chunk["document"] = uploaded_file.name

                all_chunks.extend(chunks)

            except Exception as error:
                processing_errors.append(
                    f"Unable to process {uploaded_file.name}: {error}"
                )

        st.session_state.file_signature = file_signature
        st.session_state.document_pages = document_pages
        st.session_state.unique_document_pages = unique_document_pages
        st.session_state.duplicate_documents = duplicate_documents
        st.session_state.all_chunks = all_chunks
        st.session_state.processing_errors = processing_errors

        # Reset results when uploaded documents change.
        st.session_state.financial_results = {}
        st.session_state.calculated_metrics = None

        st.session_state.pop("direct_financial_results", None)

        # Reset cooldown for the new document set.
        st.session_state.pop("financial_extraction_clicked_at", None)

        if all_chunks:

            with st.spinner("Generating shared document embeddings..."):
                model, index = create_embeddings(all_chunks)

            st.session_state.embedding_model = model
            st.session_state.embedding_index = index

        else:
            st.session_state.embedding_model = None
            st.session_state.embedding_index = None

    document_pages = st.session_state.document_pages
    unique_document_pages = st.session_state.unique_document_pages
    duplicate_documents = st.session_state.duplicate_documents
    all_chunks = st.session_state.all_chunks
    model = st.session_state.embedding_model
    index = st.session_state.embedding_index

    for error in st.session_state.processing_errors:
        st.error(error)

    if all_chunks:

        st.success("Documents processed and shared index ready!")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("PDFs Uploaded", len(document_pages))

        with col2:
            st.metric("Unique Documents Analysed", len(unique_document_pages))

        with col3:
            st.metric("Text Chunks", len(all_chunks))

        for duplicate in duplicate_documents:
            st.warning(
                f"{duplicate['Duplicate File']} was not used for RAG retrieval, "
                f"financial extraction, or calculations because it is an exact "
                f"duplicate of {duplicate['Original File']}."
            )


        # --------------------------------------------------
        # 7. DOCUMENT QUESTION ANSWERING
        # --------------------------------------------------

        st.divider()
        st.subheader("Document Question Answering")

        with st.form("document_question_form"):

            question = st.text_input(
                "Ask a question across your documents",
                placeholder="e.g. Compare the companies' total revenue.",
            )

            submitted = st.form_submit_button("Ask Question")

        if submitted and question.strip():

            search_results = search_documents(
                question,
                model,
                index,
                all_chunks,
                top_k=20,
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

                with st.expander("Relevant Document Passages", expanded=False):

                    for number, result in enumerate(search_results, start=1):

                        st.markdown(
                            f"**Result {number} — {result['document']} "
                            f"— Page {result['page']} "
                            f"(Similarity: {result['score']:.3f})**"
                        )
                        st.write(result["text"])

                        if number < len(search_results):
                            st.divider()

            except Exception as error:
                st.error(
                    f"Unable to generate an answer at the moment: {error}"
                )


        # --------------------------------------------------
        # 8. FINANCIAL DATA EXTRACTION
        # --------------------------------------------------

        st.divider()
        st.subheader("Financial Data Extraction")

        if st.button("Extract Financial Figures from Original Documents"):

            # Start the 60-second cooldown at button-click time.
            st.session_state.financial_extraction_clicked_at = time.time()

            extraction_errors = []

            for document in unique_document_pages:

                document_id = document["document_id"]
                document_name = document["name"]

                if document_id not in st.session_state.financial_results:

                    try:
                        with st.spinner(
                            f"Extracting financial figures from {document_name}..."
                        ):
                            financial_data = extract_financial_figures(
                                document["pages"]
                            )

                        st.session_state.financial_results[
                            document_id
                        ] = financial_data

                    except Exception as error:
                        extraction_errors.append(
                            f"{document_name}: {error}"
                        )

            for error in extraction_errors:
                st.error(f"Financial extraction failed — {error}")

            if not extraction_errors:
                st.success(
                    "Financial extraction completed for all unique documents!"
                )


        # --------------------------------------------------
        # 9. COMBINE RAW AND CLEANED DATA
        # --------------------------------------------------

        raw_frames = []
        cleaned_frames = []

        for document in unique_document_pages:

            document_id = document["document_id"]
            document_name = document["name"]

            financial_data = st.session_state.financial_results.get(
                document_id
            )

            if financial_data:

                figures = financial_data.get("figures", [])

                if figures:

                    raw_document_df = pd.DataFrame(figures)

                    raw_document_df["company"] = (
                        financial_data.get("company") or "NOT SPECIFIED"
                    )

                    raw_document_df["document_id"] = document_id
                    raw_document_df["source_document"] = document_name

                    raw_frames.append(raw_document_df)

                    cleaned_document_df = create_financial_analysis_dataframe(
                        figures,
                        company=financial_data.get("company"),
                        document_id=document_id,
                        document_name=document_name,
                    )

                    cleaned_frames.append(cleaned_document_df)

        raw_df = (
            pd.concat(raw_frames, ignore_index=True)
            if raw_frames else pd.DataFrame()
        )

        analysis_df = pd.DataFrame()

        if cleaned_frames:

            analysis_df = pd.concat(
                cleaned_frames,
                ignore_index=True,
            )

            analysis_df = detect_duplicates(analysis_df)


        # --------------------------------------------------
        # 10. FINANCIAL DATA TABS
        # --------------------------------------------------

        raw_tab, cleaned_tab, duplicate_tab, calculated_tab = st.tabs([
            "Raw Extracted Data",
            "Cleaned Financial Data",
            "Duplicate Data",
            "Calculated Financial Metrics",
        ])


        # --------------------------------------------------
        # RAW EXTRACTED DATA
        # --------------------------------------------------

        with raw_tab:

            st.subheader("Raw Extracted Financial Data")

            if not raw_df.empty:

                st.write(f"**Total extracted records:** {len(raw_df)}")

                st.dataframe(
                    raw_df,
                    width="stretch",
                    hide_index=True,
                )

            else:
                st.info(
                    "Click 'Extract Financial Figures from Original Documents' "
                    "to populate this tab."
                )


        # --------------------------------------------------
        # CLEANED FINANCIAL DATA
        # --------------------------------------------------

        with cleaned_tab:

            st.subheader("Cleaned Financial Data")

            if not analysis_df.empty:

                flagged_count = (
                    analysis_df["review_flag"].ne("").sum()
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Financial Figures", len(analysis_df))

                with col2:
                    st.metric(
                        "Figures Requiring Review",
                        int(flagged_count),
                    )

                with col3:
                    st.metric(
                        "Documents Analysed",
                        analysis_df["document_id"].nunique(),
                    )

                st.dataframe(
                    analysis_df,
                    width="stretch",
                    hide_index=True,
                )

            else:
                st.info(
                    "Click 'Extract Financial Figures from Original Documents' "
                    "to populate this tab."
                )


        # --------------------------------------------------
        # DUPLICATE DATA
        # --------------------------------------------------

        with duplicate_tab:

            st.subheader("Duplicate Document Detection")

            if duplicate_documents:

                st.warning(
                    f"{len(duplicate_documents)} duplicate PDF(s) were excluded "
                    "from RAG retrieval, financial extraction, and calculations."
                )

                document_comparison_df = pd.DataFrame(duplicate_documents)

                st.dataframe(
                    document_comparison_df,
                    width="stretch",
                    hide_index=True,
                )

            else:
                st.success("No exact duplicate PDF copies were uploaded.")

            st.divider()
            st.subheader("Duplicate Financial Record Detection")

            if not analysis_df.empty:

                duplicate_statuses = [
                    "Duplicate document copy",
                    "Confirmed duplicate - representative",
                    "Confirmed duplicate",
                    "Potential duplicate conflict",
                    "Potential duplicate - missing value",
                    "Insufficient context",
                ]

                duplicate_records_df = analysis_df[
                    analysis_df["duplicate_status"].isin(
                        duplicate_statuses
                    )
                ]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Duplicate / Review Records",
                        len(duplicate_records_df),
                    )

                with col2:
                    st.metric(
                        "Confirmed Duplicate Records",
                        int(
                            analysis_df["duplicate_status"].isin([
                                "Confirmed duplicate",
                                "Confirmed duplicate - representative",
                                "Duplicate document copy",
                            ]).sum()
                        ),
                    )

                with col3:
                    st.metric(
                        "Conflicting Records",
                        int(
                            (
                                analysis_df["duplicate_status"]
                                == "Potential duplicate conflict"
                            ).sum()
                        ),
                    )

                if not duplicate_records_df.empty:

                    st.dataframe(
                        duplicate_records_df,
                        width="stretch",
                        hide_index=True,
                    )

                else:
                    st.success(
                        "No duplicate or potentially conflicting "
                        "financial records were identified among the "
                        "unique documents."
                    )

            else:
                st.info(
                    "Extract financial figures from the original documents "
                    "to run financial-record duplicate detection."
                )


        # --------------------------------------------------
        # 11. CALCULATED FINANCIAL METRICS
        # --------------------------------------------------

        with calculated_tab:
            render_calculated_metrics()


        # --------------------------------------------------
        # 12. DOCUMENT PREVIEWS
        # --------------------------------------------------

        st.divider()

        with st.expander("Preview text chunks"):

            for chunk_number, chunk in enumerate(all_chunks, start=1):

                st.markdown(
                    f"**Chunk {chunk_number} — "
                    f"{chunk['document']} — Page {chunk['page']}**"
                )

                st.write(chunk["text"])
                st.divider()

        with st.expander("Preview extracted text"):

            for document in unique_document_pages:

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

else:
    st.info("Upload one or more PDFs to get started.")
