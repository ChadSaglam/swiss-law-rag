import os

import gradio as gr
from huggingface_hub import InferenceClient

from rag import EMBEDDING_MODEL, Retriever, build_prompt, load_sources

LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SOURCES = load_sources()
retriever = Retriever()
llm = InferenceClient(model=LLM_MODEL, token=os.getenv("HF_TOKEN"))


def format_sources(chunks):
    sections = []
    for index, chunk in enumerate(chunks, start=1):
        source = SOURCES[chunk.source]
        sections.append(
            f"**[{index}] [{source['title']}]({source['url']})**\n\n"
            f"```text\n{chunk.text}\n```"
        )
    return "\n\n".join(sections)


def answer(question):
    if not question or not question.strip():
        raise gr.Error("Enter a question first.")

    chunks = retriever.search(question.strip(), k=4)
    response = llm.chat_completion(
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
        max_tokens=500,
    )
    return response.choices[0].message.content, format_sources(chunks)


with gr.Blocks(title="Swiss Law RAG") as demo:
    gr.Markdown(
        "# Swiss Law RAG\n"
        "Ask about the accounting and VAT provisions in the bundled Fedlex excerpts. "
        "Answers are limited to the retrieved passages and include numbered citations.\n\n"
        f"[Embedding model](https://huggingface.co/{EMBEDDING_MODEL}) · "
        f"[Answer model](https://huggingface.co/{LLM_MODEL}) · "
        "[Source register](https://github.com/ChadSaglam/swiss-law-rag/blob/main/sources.json)\n\n"
        "**Not legal or tax advice. Check the current consolidated text on Fedlex.**"
    )
    question = gr.Textbox(
        label="Question (English / Deutsch / Français)",
        placeholder="How long must accounting records be retained?",
        lines=2,
    )
    button = gr.Button("Retrieve and answer", variant="primary")
    result = gr.Markdown(label="Answer")
    passages = gr.Markdown(label="Retrieved sources")
    gr.Examples(
        [
            "How long must accounting records be retained?",
            "Welche Angaben muss eine MWST-Rechnung enthalten?",
            "Quel est le taux normal de la TVA suisse?",
        ],
        inputs=question,
    )
    button.click(answer, question, [result, passages])
    question.submit(answer, question, [result, passages])


if __name__ == "__main__":
    demo.launch()
