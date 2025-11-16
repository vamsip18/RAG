import streamlit as st
import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

api_key = st.secrets["GROQ_API_KEY"]

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ----------------------------- UI -----------------------------
st.title("📄 Cerevyn Document Intelligence – AI PDF/Q&A Agent")
st.write("Upload PDFs and ask questions. Answers include **page references**.")


llm = ChatGroq(
    groq_api_key=api_key,
    model_name="openai/gpt-oss-120b"
)

session_id = st.text_input("💬 Session ID", value="default_session")

if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("📎 Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

# ----------------------------- PDF HANDLING -----------------------------
if uploaded_files:
    documents = []
    temp_dir = "./uploaded_pdfs"
    os.makedirs(temp_dir, exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 🔥 ADD METADATA FOR PAGE NUMBER + DOCUMENT NAME
        for d in docs:
            d.metadata["source"] = uploaded_file.name
            d.metadata["page_number"] = d.metadata.get("page", None)

        documents.extend(docs)

    # ----------------------------- Indexing -----------------------------
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    splits = text_splitter.split_documents(documents)

    # Persistent vector DB (fast if many queries)
    vectorstore = Chroma.from_documents(
        splits,
        embedding=embeddings,
        persist_directory="./chroma_store"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ----------------------------- Question Re-writing -----------------------------
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite follow-up questions into standalone questions using chat history."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_q_prompt
    )

    # ----------------------------- QA Prompt -----------------------------
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant.\n"
         "Use the provided context to answer the question.\n"
         "If the answer is not in the document, reply that you cannot find it.\n\n"
         "{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # ----------------------------- Chat History Manager -----------------------------
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # ----------------------------- CHAT INPUT -----------------------------
    user_input = st.chat_input("Ask something about your PDFs...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            result = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            st.markdown(result["answer"])

            # ----------------------------- SHOW SOURCES -----------------------------
            st.markdown("### 📚 Sources")
            for doc in result["context"]:
                source = doc.metadata.get("source", "Unknown PDF")
                page = doc.metadata.get("page_number", "?")

                st.markdown(f"• **{source} – Page {page}**")
else:
    st.info("Upload at least one PDF to begin.")