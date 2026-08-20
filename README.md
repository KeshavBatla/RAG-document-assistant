# 📄 RAG Document Assistant
Website link: https://rag-document-assistant-kb.streamlit.app/
An intelligent, context-aware RAG (Retrieval-Augmented Generation) document assistant built with **Streamlit**, **LangChain**, **ChromaDB**, and **Google Gemini (google-genai)**. 

The application dynamically routes queries to either perform vector similarity searches, provide single-document summaries, or synthesize multi-document summaries.

---

## 🛠️ Features

* **Intent Routing:** Uses Gemini to categorize user queries into `summary_all`, `summary_single`, or `rag_search`.
* **Multi-Format Ingestion:** Supports `.pdf`, `.csv`, `.txt`, and `.md` file parsing.
* **Persistent Vector Storage:** Leverages ChromaDB for efficient local text chunk vector embeddings.
* **Interactive UI:** Web chat interface powered by Streamlit with expandable source citation details.

---

## 📁 Repository Structure

```text
├── app.py              # Streamlit UI application
├── ingest.py           # Document loading, chunking, and ChromaDB indexing
├── query.py            # Gemini intent routing & RAG response generation
├── .env                # Environment keys (API keys, config)
├── .gitignore          # Git exclusion config
└── requirements.txt    # Required Python dependencies
