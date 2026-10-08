import os
import tempfile
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Import existing RAG components
from ingest import ingest, ingest_temporary, embeddings, CHROMA_PATH
from query import query
from langchain_community.vectorstores import Chroma

app = FastAPI(title="RAG Document Assistant API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    filter_file: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

class HealthResponse(BaseModel):
    status: str
    message: str

class UploadResponse(BaseModel):
    message: str
    files_processed: int

# Global variable to hold the vector database (in production, consider using a proper session/store)
# For Vercel serverless, we'll need to handle this differently - likely load on each request
db_instance = None

@app.on_event("startup")
async def startup_event():
    """Load default knowledge base on startup if chroma_db exists"""
    global db_instance
    try:
        if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
            db_instance = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embeddings
            )
            print("✅ Loaded existing ChromaDB on startup")
        else:
            # Load default documents from data directory
            DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
            if os.path.exists(DATA_PATH):
                db_instance = ingest(DATA_PATH)
                print("✅ Loaded default documents on startup")
            else:
                print("⚠️ No data directory found, starting with empty DB")
    except Exception as e:
        print(f"⚠️ Error loading DB on startup: {e}")
        db_instance = None

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global db_instance
    if db_instance is not None:
        return HealthResponse(status="healthy", message="RAG Document Assistant is running")
    else:
        return HealthResponse(status="degraded", message="Database not initialized")

@app.post("/api/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process documents"""
    global db_instance

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        # Save uploaded files
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # Process files - combine with existing DB if available
        if db_instance:
            new_db = ingest_temporary(temp_dir, existing_db=db_instance)
            db_instance = new_db
        else:
            db_instance = ingest_temporary(temp_dir)

        # Cleanup
        shutil.rmtree(temp_dir)

        return UploadResponse(
            message=f"✅ Successfully processed {len(files)} file(s)",
            files_processed=len(files)
        )
    except Exception as e:
        # Cleanup on error
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    global db_instance

    if db_instance is None:
        raise HTTPException(status_code=400, detail="Please upload documents or ensure default knowledge base is loaded")

    try:
        answer, sources = query(db_instance, request.question, filter_file=request.filter_file)
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/documents")
async def get_loaded_documents():
    """Get list of currently loaded documents"""
    global db_instance

    if db_instance is None:
        return {"documents": []}

    try:
        all_docs = db_instance.get()
        filenames = list(set([
            m.get("file_name")
            for m in all_docs.get("metadatas", [])
            if m and "file_name" in m
        ]))
        return {"documents": filenames}
    except Exception as e:
        return {"documents": [], "error": str(e)}

# Serve frontend
FRONTEND_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "frontend"
)

app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)