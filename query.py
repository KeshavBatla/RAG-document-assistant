from dotenv import load_dotenv
import os
import json
import google.genai as genai
from google.genai import types
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv(override=True)
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME=os.getenv("MODEL_NAME")

client = genai.Client(api_key=API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#call llm for query modify and specify that is that require full summary, particular summary or any question
def query_modification(query,filenames):
    prompt = f"""
        You are an expert prompt analyzer. 
        Analyze the user's question and available files to select the correct action.
        
        Available files: {filenames}
        You need to analyze the uploaded question,filenames and categorize it into one of the only 3 mentioned categories:
        <categories>
        "summarize_all", "summarize_specific", "rag_search"
        </categories>
        Output MUST be valid JSON with this exact format:
        {{
            "category": "category",
            "target_file": "filename.pdf or null",
            "rewritten_query": "Optimized query string for similarity search if route is rag_search, else original query"
        }}
        If the question asks for summarizing all uploaded documents then the category is "sammarize_all".
        If the question is for summarizing a  specific document then the category is "summarize_specific" and mention the filename asked for summary in the target_file.
        If it is a general query about the uploaded document then the category is rag_search.
        If the question is identified as general query , optimize it for getting better results for similarity search in vector database.
        
    
    Question:
    {query}
    """
    
    client = genai.Client()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)

def retrieve(db, query, k=5):
    results = db.similarity_search(query, k=k)
    sources = list(set([
            c.metadata.get("file_name") 
            for c in results 
            if c.metadata and "file_name" in c.metadata
        ]))
    return "\n\n".join([doc.page_content for doc in results]) , sources

def generate_answer(context, prompt):
    prompt=f""" 
You are an expert Document Analyzer and Summarizer.
You need to solve users query based on the document it uploads.
If the query is about summarizing the entire document or a single document then only summarize the uploaded documents below and ignore the following instructions.
<query>
{prompt}
</query>

Here are the documents:
<documents>
{context}
</documents>

If the user asks for a question in the uploaded data, only answer from the uploaded data, do not hallucinate the answer.
If there is no answer in the documents for the user's query return with "sorry i do not find the answer for that"


"""
    
    client = genai.Client()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0)
    )
    return response.text

def query(db,user_query,filter_file=None):

    all_docs=db.get()               

    filenames = list(set([
        m.get("file_name") 
        for m in all_docs.get("metadatas", []) 
        if m and "file_name" in m
    ]))
    
    modified_query_json=query_modification(user_query,filenames)

    category=modified_query_json.get("category")

    if category=="summarize_all":
        full_text = "\n\n".join(all_docs["documents"])
        prompt = "Provide a comprehensive, structured summary of all documents below."
        return generate_answer(full_text,prompt),filenames

    
    elif category=="summarize_specific":

        targetfile=modified_query_json.get("target_file")
        results = db.get( where={"file_name": targetfile} ) 
        single_doc_chunks = results["documents"]

        prompt = "Provide a complete summary of the document below"
        return generate_answer(single_doc_chunks,prompt), [targetfile]

    else:
        rewritten_query = modified_query_json.get("rewritten_query")

        if filter_file:
            results = db.similarity_search(
                rewritten_query, k=5,
                filter={"file_name": filter_file}
            )
            context = "\n\n".join([doc.page_content for doc in results])
            sources = [filter_file]
        else:
            context, sources = retrieve(db, rewritten_query)
        
        return generate_answer(context,rewritten_query), sources


if __name__ == "__main__":

    from ingest import ingest
    CHROMA_PATH = "chroma_db"
    DATA_PATH = r"C:\Users\batla\OneDrive\Desktop\RAGdocumnets"
    
    db = ingest(DATA_PATH)
    answer, sources = query(db, "summarize all documents")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")