# Swiss Law RAG

A small retrieval-augmented question-answering project over six official Swiss legal excerpts. It supports questions in German, English and French, retrieves relevant passages and asks an LLM to answer only from those passages with numbered citations.

**Not legal or tax advice. The bundled excerpts are a limited snapshot; check the current consolidated text on Fedlex.**

## Resources

- [Kaggle retrieval evaluation](https://www.kaggle.com/code/saglamchad/swiss-law-rag)
- [Embedding model: multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
- [Answer model: Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Source register](sources.json)
- [Measured evaluation](evaluation.json)

## Legal sources

The corpus contains the original German wording of six provisions retrieved from Fedlex on 28 October 2026. Each text file includes its canonical source URL.

| File | Official source |
|---|---|
| `or-art-957-de.txt` | [Code of Obligations, Art. 957](https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_957) |
| `or-art-958-de.txt` | [Code of Obligations, Art. 958](https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_958) |
| `or-art-958f-de.txt` | [Code of Obligations, Art. 958f](https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_958_f) |
| `mwstg-art-25-de.txt` | [Value Added Tax Act, Art. 25](https://www.fedlex.admin.ch/eli/cc/2009/615/de#art_25) |
| `mwstg-art-26-de.txt` | [Value Added Tax Act, Art. 26](https://www.fedlex.admin.ch/eli/cc/2009/615/de#art_26) |
| `mwstg-art-70-de.txt` | [Value Added Tax Act, Art. 70](https://www.fedlex.admin.ch/eli/cc/2009/615/de#art_70) |

## Evaluation

Measured with `run_eval.py` on 40 hand-written questions in German, English and French, using the same `Retriever` class as the app:

| Metric | Result |
|---|---:|
| Hit@1 | 87.5% |
| Hit@4 | 100% |
| Questions with the expected source in the top 4 | 40 / 40 |
| Indexed chunks | 16 |

Hit@4 was 100% in each language subset: 14 German, 14 English and 12 French questions. The passages use 900-character chunks with a 150-character overlap and `passage:` / `query:` prefixes for the E5 model.

This is a small, closed-corpus retrieval test written around the six bundled provisions. It does not measure answer correctness, citation faithfulness or performance on unseen areas of law. No score is reported for the LLM.

## Run the evaluation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_eval.py
```

The script writes `evaluation.json`, including per-language results and any failed queries. The first run downloads the embedding model from Hugging Face. No LLM token is needed for retrieval evaluation.

## Run the Gradio app

```bash
HF_TOKEN=hf_xxx python app.py
```

The token needs access to an inference provider for `Qwen/Qwen2.5-7B-Instruct`. Set `LLM_MODEL` to use another compatible chat model.

The app returns both the generated answer and the four retrieved passages with clickable Fedlex citations. An empty corpus or missing inference access must be fixed before using the app; the retrieval evaluation does not test either service.

## Files

- `rag.py`: chunking, embeddings, retrieval and prompt construction
- `app.py`: Gradio interface and source presentation
- `run_eval.py`: reproducible retrieval evaluation
- `docs/`: attributed German legal excerpts
- `sources.json`: canonical source metadata
- `eval/questions.json`: 40 multilingual retrieval questions
- `swiss_law_rag.ipynb`: Kaggle version of the evaluation
