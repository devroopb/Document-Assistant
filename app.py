
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


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Business Document Assistant",
    page_icon="📄",
    layout="wide",
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
# 3. DOCUMENT UPLOAD
# --------------------------------------------------

st.subheader("Document Workspace")

st.write("Upload one or more PDFs to begin analysing their contents.")

uploaded_files = st.file_uploader(
    "Choose business documents",
    type=["pdf"],
    accept_multiple_files=True,
)


# --------------------------------------------------
# 4. PDF PROCESSING AND SESSION STATE
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
        processing_errors = []

        for uploaded_file in uploaded_files:

            try:
                pages = extract_text_from_pdf(uploaded_file)

                document_id = hashlib.sha256(
                    uploaded_file.getvalue()
                ).hexdigest()

                document_pages.append({
                    "name": uploaded_file.name,
                    "document_id": document_id,
                    "pages": pages,
                })

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
        st.session_state.all_chunks = all_chunks
        st.session_state.processing_errors = processing_errors
        st.session_state.financial_results = {}

        st.session_state.pop("direct_financial_results", None)

        if all_chunks:

            with st.spinner("Generating shared document embeddings..."):
                model, index = create_embeddings(all_chunks)

            st.session_state.embedding_model = model
            st.session_state.embedding_index = index

        else:
            st.session_state.embedding_model = None
            st.session_state.embedding_index = None

    document_pages = st.session_state.document_pages
    all_chunks = st.session_state.all_chunks
    model = st.session_state.embedding_model
    index = st.session_state.embedding_index

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

        if "financial_results" not in st.session_state:
            st.session_state.financial_results = {}

        if st.button("Extract Financial Figures from All Documents"):

            extraction_errors = []

            for document in document_pages:

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
                    "Financial extraction completed for all documents!"
                )

        # --------------------------------------------------
        # 7. COMBINE RAW AND CLEANED DATA
        # --------------------------------------------------

        raw_frames = []
        cleaned_frames = []

        for document in document_pages:

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

        if cleaned_frames:

            analysis_df = pd.concat(
                cleaned_frames,
                ignore_index=True,
            )

            analysis_df = detect_duplicates(analysis_df)

        # --------------------------------------------------
        # 8. FINANCIAL DATA TABS
        # --------------------------------------------------

        if raw_frames or cleaned_frames:

            raw_tab, cleaned_tab, duplicate_tab = st.tabs([
                "Raw Extracted Data",
                "Cleaned Financial Data",
                "Duplicate Data",
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
                        use_container_width=True,
                        hide_index=True,
                    )

                else:
                    st.info("No raw financial figures available.")

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
                        st.metric(
                            "Financial Figures",
                            len(analysis_df),
                        )

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
                        use_container_width=True,
                        hide_index=True,
                    )

                else:
                    st.info("No cleaned financial data available.")

            # --------------------------------------------------
            # DUPLICATE DATA
            # --------------------------------------------------

            with duplicate_tab:

                st.subheader("Duplicate Document Detection")

                if not analysis_df.empty:

                    document_comparison_df = (
                        analysis_df[
                            [
                                "document_id",
                                "source_document",
                                "document_duplicate_status",
                            ]
                        ]
                        .drop_duplicates()
                        .reset_index(drop=True)
                    )

                    st.dataframe(
                        document_comparison_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.divider()

                    st.subheader("Duplicate Financial Record Detection")

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
                            use_container_width=True,
                            hide_index=True,
                        )

                    else:
                        st.success(
                            "No duplicate or potentially conflicting "
                            "financial records were identified."
                        )

                else:
                    st.info("No duplicate financial records available.")

        else:

            st.info(
                "Upload documents and click 'Extract Financial Figures "
                "from All Documents' to begin."
            )

        # --------------------------------------------------
        # 9. DOCUMENT PREVIEWS
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