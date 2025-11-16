# 📄 Cerevyn Document Intelligence – AI PDF/Q&A Agent  
### 🧠 RAG System · Groq LLM · Enterprise Documentation Assistant

This project is built for **Problem Statement 10: Document Intelligence**.  
It enables users to upload one or multiple PDFs and ask natural-language questions.  
The system retrieves answers **accurately with page references** using a fast and lightweight **RAG pipeline**.

---

## 🚀 Features

### ✅ Core Requirements
- PDF text extraction (page-level)
- Embedding + vector similarity search
- Retrieval-Augmented Generation (RAG)
- Chat interface with full conversation history
- Page-accurate citations
- Multi-PDF support

### 🎯 Industry-Level Features
- Clean & responsive Streamlit UI  
- Real-time vector indexing  
- Extremely fast Groq Llama-3.1 inference  
- FAISS vector store (stable on Windows, no corruption)
  
---

## 🛠️ Tech Stack

### Backend
- **LangChain (LCEL)**
- **Groq LLM (Llama 3.1 8B Instant)**
- **FAISS Vector Database**
- **HuggingFace Embeddings**
- **PyPDFLoader**

### Frontend
- **Streamlit**

---

## 🧩 Skills Demonstrated
- RAG pipeline development  
- Vector databases  
- NLP & embeddings  
- Prompt engineering  
- AI application design  
- Streamlit UI engineering  
- System architecture & optimization  
- End-to-end pipeline development  

---

## ▶️ How It Works (Workflow)

1. User uploads multiple PDFs  
2. System extracts text per page  
3. Text is chunked into overlapping segments  
4. Chunks → embeddings → FAISS vector store  
5. User submits a question  
6. Retriever pulls best matching chunks  
7. Groq LLM answers using ONLY retrieved context  
8. Answer + page numbers shown in chat UI  



