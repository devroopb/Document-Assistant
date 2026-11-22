
# Import Streamlit to build the web application.
import streamlit as st

# Import our PDF processing function from the src folder.
from src.document_processor import extract_text_from_pdf

# Import the function that splits extracted PDF text into chunks.
from src.text_chunker import split_into_chunks

# Import the function that converts text chunks into embeddings
# and stores them in a FAISS search index.
from src.embedding_search import create_embeddings

# Import the function that searches document chunks using semantic similarity.
from src.embedding_search import search_documents
from src.answer_generator import generate_answer
import time
from src.financial_extractor import extract_financial_figures
import pandas as pd
from src.clean_tab_financial_analysis import create_financial_analysis_dataframe




# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

# Set the browser tab title, icon, and page layout.
st.set_page_config(
    page_title="Business Document Assistant",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# 2. APPLICATION HEADER
# --------------------------------------------------

# Display the main application title.
st.title("Business Document Assistant")

# Display a short description below the title.
st.caption(
    "Your workspace for business documents and evidence-based insights."
)

# Add a horizontal divider.
st.divider()


# --------------------------------------------------
# 3. DOCUMENT UPLOAD SECTION
# --------------------------------------------------

# Display the document workspace heading.
st.subheader("Document Workspace")

# Tell the user what to do.
st.write("Upload a PDF to begin analysing its contents.")

# Create a file uploader that accepts PDF files only.
# The uploaded file is temporarily held in uploaded_file.
uploaded_file = st.file_uploader(
    "Choose a business document",
    type=["pdf"]
)


# --------------------------------------------------
# 4. PDF PROCESSING
# --------------------------------------------------

# Only process the document if the user has uploaded one.
if uploaded_file is not None:

    try:
        # Send the uploaded PDF to our extraction function.
        # The function returns the text from each page.
        pages = extract_text_from_pdf(uploaded_file)
        
        # Let the user choose when to run financial extraction.
        if st.button("Extract Financial Figures"):

            try:
                with st.spinner("Extracting financial figures with Gemini..."):

                    # Extract structured financial data from the PDF.
                    financial_data = extract_financial_figures(pages)

                st.success("Financial figures extracted successfully!")

            except ValueError as error:

                # Handle documents without financial statements separately.
                if (
                    "No relevant financial statement pages" in str(error)
                    or "No financial figures could be extracted" in str(error)
                ):
                    st.info(
                        "No financial statements or extractable financial "
                        "figures were detected in this document. "
                        "You can still use document question answering."
                    )

                else:
                    st.error(
                        f"Financial extraction could not be completed: {error}"
                    )

            except Exception as error:

                # Report unexpected extraction or API failures.
                st.error(
                    f"Financial extraction failed: {error}"
                )

            else:

                # Create the original extracted financial DataFrame.
                financial_df = pd.DataFrame(financial_data["figures"])

                financial_df = financial_df.sort_values(
                    by=["fiscal_year", "metric_name"],
                    ascending=[False, True]
                ).reset_index(drop=True)

                # Create separate raw and cleaned data tabs.
                raw_tab, cleaned_tab = st.tabs([
                    "Raw Extracted Data",
                    "Cleaned Financial Data"
                ])

                # Display all original extracted records.
                with raw_tab:

                    st.subheader("Raw Financial Data")

                    st.dataframe(
                        financial_df,
                        use_container_width=True,
                        hide_index=True
                    )

                # Display the simplified financial analysis.
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


        # Split the extracted PDF text into smaller sections.
        # Each chunk contains its text and original PDF page number.
        chunks = split_into_chunks(
            pages,
            chunk_size=1000,
            overlap=200
        )
        
        # --------------------------------------------------
        # GENERATE DOCUMENT EMBEDDINGS
        # --------------------------------------------------

        # Show a progress message while the model processes the text.
        with st.spinner("Generating document embeddings..."):

            # Convert all chunks into numerical embeddings
            # and store them in a FAISS search index.
            model, index = create_embeddings(chunks)

        # Display the number of embeddings successfully stored.
        st.success("Document embeddings generated successfully!")

        st.write(f"**Embeddings stored in FAISS:** {index.ntotal}")
        
        # --------------------------------------------------
        # DOCUMENT QUESTION INPUT
        # --------------------------------------------------

        # Allow the user to enter a question about the uploaded PDF.
        question = st.text_input(
            "Ask a question about your document",
            placeholder="e.g. What was the company's total revenue?"
        )
        
        # Only search the document when the user enters a question.
        if question:

            # Find the five most relevant text chunks.
            search_results = search_documents(
                question,
                model,
                index,
                chunks,
                top_k=20
            )
            
            # Measure how long Gemini takes to generate its answer.
            start_time = time.perf_counter()

            with st.spinner("Generating answer..."):
                answer = generate_answer(question, search_results)

            # Calculate and display the response time.
            response_time = time.perf_counter() - start_time
            st.caption(f"Gemini response time: {response_time:.2f} seconds")

            # Display the AI-generated answer.
            st.subheader("AI-Generated Answer")
            st.write(answer)
            
            # Display the retrieved passages.
            st.subheader("Relevant Document Passages")

            for number, result in enumerate(search_results, start=1):

                # Show the matching passage and its original PDF page.
                with st.expander(
                    f"Result {number} — Page {result['page']} "
                    f"(Similarity: {result['score']:.3f})"
                ):
                    st.write(result["text"])

        # Display the total number of chunks created.
        st.write(f"**Text chunks created:** {len(chunks)}")
        # Create an expandable section to inspect the generated chunks.
        with st.expander("Preview text chunks"):

            # Loop through every chunk and display its contents.
            for index, chunk in enumerate(chunks):

                # Show the chunk number and its original PDF page.
                st.markdown(
                    f"**Chunk {index + 1} — Page {chunk['page']}**"
                )

                # Display the extracted text inside this chunk.
                st.write(chunk["text"])

                # Separate chunks visually.
                st.divider()
        # Confirm that the PDF was processed successfully.
        st.success(f"Successfully processed: {uploaded_file.name}")

        # Display how many pages contained extractable text.
        st.write(f"**Pages containing text:** {len(pages)}")

        # Create an expandable section for the extracted text.
        with st.expander("Preview extracted text"):

            # Go through each page returned by the PDF processor.
            for page in pages:

                # Display the original PDF page number.
                st.markdown(f"**Page {page['page']}**")

                # Display the first 1,500 characters from that page.
                st.write(page["text"][:1500])

                # Separate the pages visually.
                st.divider()

    except Exception as error:
        # Display an error if the PDF cannot be processed.
        st.error(f"Unable to process this PDF: {error}")
