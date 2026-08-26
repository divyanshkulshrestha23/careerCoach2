import os
import chromadb
from chromadb.utils import embedding_functions

# Initialize the Persistent Client
DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=DB_PATH)

# Use ChromaDB's built-in local model (No API key needed!)
default_ef = embedding_functions.DefaultEmbeddingFunction()

# Initialize the Collection
resume_collection = client.get_or_create_collection(
    name="resume_store",
    embedding_function=default_ef
)


def store_resume_chunks(sections_dict):
    """
    Takes the parsed OCR resume sections dictionary and stores them as individual vector chunks.
    """
    documents = []
    metadatas = []
    ids = []

    # Iterate through each section identified by the YOLO model (e.g., 'Experience', 'Education')
    for idx, (section, content) in enumerate(sections_dict.items()):
        documents.append(content)
        metadatas.append({"section": section})
        ids.append(f"resume_sec_{idx}")

    if documents:
        # Upsert ensures that if a section ID already exists, it is overwritten with the new text
        resume_collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    return "Resume successfully stored in the local vector database!"


def query_relevant_experience(job_description_text, n_results=2):
    """
    Embeds the job description and performs a semantic search to retrieve
    the most relevant resume chunks for tailored advice and cover letters.
    """
    if resume_collection.count() == 0:
        return "No resume data found in the database. Please upload and parse a resume first."

    # Perform the similarity search
    # Chroma converts the query text into an embedding and uses similarity algorithms to return the closest matches.
    results = resume_collection.query(
        query_texts=[job_description_text],
        n_results=n_results
    )

    # Combine the top 'n' matched chunks into a single string for the Gemini LLM to use as context
    if results and results['documents'] and results['documents'][0]:
        matched_chunks = "\n\n".join(results['documents'][0])
        return matched_chunks

    return ""