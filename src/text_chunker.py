
def split_into_chunks(pages, chunk_size=1000, overlap=200):
    """
    Split extracted PDF text into smaller sections.

    chunk_size: Maximum number of characters in each chunk.
    overlap: Number of characters repeated between consecutive chunks.

    Keeping some overlap helps prevent important information
    from being lost when a sentence crosses a chunk boundary.
    """

    chunks = []

    # Process each PDF page separately to preserve its page number.
    for page in pages:
        text = page["text"]
        page_number = page["page"]

        # Move through the page text in smaller sections.
        start = 0

        while start < len(text):

            # Calculate where the current chunk should end.
            end = start + chunk_size

            # Extract the current section of text.
            chunk_text = text[start:end].strip()

            # Save non-empty chunks along with their source page.
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            # Move forward, keeping the specified overlap.
            start += chunk_size - overlap

    return chunks