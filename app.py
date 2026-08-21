import streamlit as st
import anthropic
import chromadb
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
chroma_client = chromadb.Client()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("Knowledge Base Assistant")
st.caption("Upload documents and ask questions. Every answer cites its source.")

if "collection" not in st.session_state:
    st.session_state.collection = chroma_client.get_or_create_collection("kb-assistant")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages)
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

uploaded_files = st.file_uploader(
    "Upload your documents", 
    type=["txt", "pdf", "docx"], 
    accept_multiple_files=True
)

if uploaded_files and not st.session_state.docs_loaded:
    with st.spinner("Processing documents..."):
        for file in uploaded_files:
            text = extract_text(file)
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                embedding = embedding_model.encode(chunk).tolist()
                st.session_state.collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{"source": file.name, "chunk": i+1}],
                    ids=[f"{file.name}_chunk_{i+1}"]
                )
        st.session_state.docs_loaded = True
    st.success(f"Loaded {len(uploaded_files)} document(s). Ask your question below.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    question_embedding = embedding_model.encode(question).tolist()
    results = st.session_state.collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    chunks_text = ""
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        chunks_text += f"\n[Source: {meta['source']}, Chunk {meta['chunk']}]\n{doc}\n"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system="Answer the user's question using ONLY the document excerpts provided. For every claim, cite the exact source filename and chunk number. If the answer is not in the excerpts, respond exactly: 'I don't know based on the available documents.' Do not use outside knowledge. Do not suggest external resources, emails, or actions not mentioned in the excerpts.",
        messages=[{"role": "user", "content": f"Document excerpts:\n{chunks_text}\n\nQuestion: {question}"}]
    )

    answer = response.content[0].text
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
        st.text_area("Copy response:", value=answer, height=100,
                     key=f"copy_{len(st.session_state.messages)}")