import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from rag import MediQueryRAG

load_dotenv()

st.set_page_config(page_title="MediQuery", page_icon="🩺", layout="wide")

st.title("🩺 MediQuery")
st.caption("Ask questions about your medical documents using AI-powered document retrieval.")

st.info(
    "MediQuery is an information-retrieval assistant, not a diagnostic or treatment tool. "
    "Always consult a qualified healthcare professional for medical advice."
)

if "rag" not in st.session_state:
    st.session_state.rag = MediQueryRAG()
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "files" not in st.session_state:
    st.session_state.files = []

with st.sidebar:
    st.header("📄 Upload reports")
    uploads = st.file_uploader(
        "Upload PDF medical reports",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process documents", type="primary", use_container_width=True):
        if not uploads:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Reading and indexing your reports..."):
                st.session_state.rag = MediQueryRAG()
                results = []
                for uploaded in uploads:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = tmp.name
                    try:
                        count = st.session_state.rag.add_pdf(tmp_path, uploaded.name)
                        results.append(f"✓ {uploaded.name} — {count} chunks")
                    finally:
                        os.unlink(tmp_path)

                st.session_state.indexed = True
                st.session_state.files = [u.name for u in uploads]
            st.success("Documents processed!")
            for item in results:
                st.caption(item)

    if st.session_state.files:
        st.divider()
        st.subheader("Loaded reports")
        for name in st.session_state.files:
            st.write(f"📄 {name}")

question = st.text_input(
    "Ask a question",
    placeholder="e.g. What is my hemoglobin value?"
)

ask = st.button("🔍 Ask MediQuery", type="primary", disabled=not st.session_state.indexed)

if ask:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching your reports..."):
            try:
                result = st.session_state.rag.ask(question)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                result = None

        if result:
            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("📚 Sources")
            for source in result["sources"]:
                with st.container(border=True):
                    st.write(f"**{source['filename']}** — Page {source['page']}")
                    st.caption(source["text"][:700] + ("..." if len(source["text"]) > 700 else ""))

if not st.session_state.indexed:
    st.markdown(
        """
        ### How to use MediQuery
        1. Upload your medical PDF reports from the sidebar.
        2. Click **Process documents**.
        3. Ask a question about the uploaded reports.
        4. MediQuery retrieves the relevant passages and shows their page sources.
        """
    )
