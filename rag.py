import os
import re
import numpy as np
import faiss
import fitz
from sentence_transformers import SentenceTransformer
from groq import Groq


class MediQueryRAG:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add it to your environment variables or Streamlit secrets."
            )

        self.client = Groq(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    def _clean_text(self, text):
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _chunk_text(self, text, chunk_size=900, overlap=150):
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(words):
                break
            start = end - overlap

        return chunks

    def add_pdf(self, pdf_path, filename):
        doc = fitz.open(pdf_path)
        added = 0

        for page_number, page in enumerate(doc, start=1):
            text = self._clean_text(page.get_text())

            if not text:
                continue

            for chunk in self._chunk_text(text):
                self.chunks.append({
                    "text": chunk,
                    "page": page_number,
                    "filename": filename
                })
                added += 1

        doc.close()

        if not self.chunks:
            raise ValueError(
                "No readable text was found. This MVP supports text-based PDFs; "
                "scanned/image-only PDFs need OCR."
            )

        embeddings = self.embedder.encode(
            [c["text"] for c in self.chunks],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        return added

    def _retrieve(self, question, k=5):
        if self.index is None:
            return []

        query_embedding = self.embedder.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            min(k, len(self.chunks))
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.chunks[idx])
            item["score"] = float(score)
            results.append(item)

        return results

    def ask(self, question):
        retrieved = self._retrieve(question, k=5)

        if not retrieved:
            return {
                "answer": "I couldn't find this information in the uploaded documents.",
                "sources": []
            }

        context_parts = []
        for i, item in enumerate(retrieved, start=1):
            context_parts.append(
                f"[Source {i} | {item['filename']} | Page {item['page']}]\n"
                f"{item['text']}"
            )

        context = "\n\n".join(context_parts)

        system_prompt = """You are MediQuery, a medical document question-answering assistant.

Your job is to help users retrieve and understand information that is explicitly present
in their uploaded medical documents.

Rules:
- Answer ONLY from the supplied document context.
- Do not diagnose diseases.
- Do not prescribe or recommend treatment, medication, dosage, or medical action.
- Do not invent values or facts.
- If the requested information is not clearly present in the context, say:
  "I couldn't find this information in the uploaded documents."
- For comparisons, clearly identify which document/date each value came from.
- Keep answers concise and easy to understand.
- You may explain what a document says, but do not turn that explanation into medical advice.
"""

        user_prompt = f"""DOCUMENT CONTEXT:

{context}

USER QUESTION:
{question}

Answer using only the document context."""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = response.choices[0].message.content.strip()

        sources = []
        seen = set()
        for item in retrieved:
            key = (item["filename"], item["page"])
            if key not in seen:
                sources.append({
                    "filename": item["filename"],
                    "page": item["page"],
                    "text": item["text"]
                })
                seen.add(key)

        return {
            "answer": answer,
            "sources": sources
        }
