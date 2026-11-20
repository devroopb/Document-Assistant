
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load the API key from the project's .env file.
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


def generate_answer(question, search_results):
    """
    Generate an evidence-based answer using Gemini
    and passages retrieved through FAISS.
    """

    if not api_key:
        raise ValueError("Gemini API key not found. Check your .env file.")

    # Create the Gemini API client.
    client = genai.Client(api_key=api_key)

    # Combine retrieved passages while preserving page references.
    context = ""

    for result in search_results:
        context += (
            f"\n[Page {result['page']}]\n"
            f"{result['text']}\n"
        )

    # Build a prompt that restricts answers to document evidence.
    prompt = f"""
    You are a business document analysis assistant.

    Answer the user's question using only the document passages below.

    Instructions:
    - Do not invent or assume information.
    - Cite relevant page numbers in your answer.
    - If the evidence is insufficient, say so clearly.
    - Keep the answer concise, clear, and professional.

    DOCUMENT PASSAGES:
    {context}

    USER QUESTION:
    {question}
    """

    # Limit the response length and avoid unnecessary continuation requests.
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=400,
            temperature=0.2,
            automatic_continuation=False
        )
    )

    return response.text
