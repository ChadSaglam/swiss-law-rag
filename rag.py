import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


def load_sources(path=ROOT / "sources.json"):
    sources = json.loads(Path(path).read_text(encoding="utf-8"))
    return {source["file"]: source for source in sources}


def load_chunks(folder=ROOT / "docs", size=900, overlap=150):
    if size <= overlap:
        raise ValueError("Chunk size must be greater than the overlap.")

    chunks = []
    for path in sorted(Path(folder).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for start in range(0, len(text), size - overlap):
            passage = text[start : start + size].strip()
            if passage:
                chunks.append(Chunk(source=path.name, text=passage))
    return chunks


class Retriever:
    def __init__(self, model_name=EMBEDDING_MODEL, size=900, overlap=150):
        self.model = SentenceTransformer(model_name)
        self.chunks = load_chunks(size=size, overlap=overlap)
        if not self.chunks:
            raise FileNotFoundError("No documents found. Add .txt files to docs/.")
        self.vectors = self.model.encode(
            [f"passage: {chunk.text}" for chunk in self.chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, question, k=4):
        query = self.model.encode(
            [f"query: {question}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        ranked = np.argsort(-(self.vectors @ query))[:k]
        return [self.chunks[index] for index in ranked]


def build_context(chunks):
    return "\n\n".join(
        f"[{index}] ({chunk.source}) {chunk.text}"
        for index, chunk in enumerate(chunks, start=1)
    )


def build_prompt(question, chunks):
    return (
        "Answer only from the sources below, in the question's language. "
        "Cite sources like [1]. If the answer is not in the sources, say so.\n\n"
        f"Sources:\n{build_context(chunks)}\n\nQuestion: {question}"
    )
