# 🩺 MediQuery

MediQuery is a simple AI-powered medical document assistant using Retrieval-Augmented Generation (RAG).

It lets users:
- Upload medical PDF reports
- Search them using natural-language questions
- Retrieve relevant passages semantically
- Ask an LLM to answer using only the uploaded documents
- See source filename and page number
- Compare information across multiple uploaded reports

## Tech stack

- Streamlit
- PyMuPDF
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- Groq API

The embeddings and vector search run locally. Groq is used only for the LLM response.

## Run locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create your environment file

Copy `.env.example` to `.env` and add your Groq API key:

```text
GROQ_API_KEY=your_key_here
```

### 3. Start the app

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload these files.
3. Open Streamlit Community Cloud.
4. Select your GitHub repository and `app.py`.
5. In the app settings, add a secret:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "llama-3.1-8b-instant"
```

6. Deploy.

## Important limitation

This MVP handles text-based PDFs. Scanned/image-only medical reports require OCR, which can be added later.

## Safety

MediQuery is an information-retrieval prototype. It is not a medical diagnostic system and should not be used to make medical decisions. Always consult a qualified healthcare professional for medical advice.
