import json
from collections import defaultdict
from datetime import date

from rag import EMBEDDING_MODEL, ROOT, Retriever


def main():
    questions = json.loads((ROOT / "eval/questions.json").read_text(encoding="utf-8"))
    assert len(questions) == 40, "The evaluation set must contain 40 questions."
    assert len({item["id"] for item in questions}) == len(questions)

    retriever = Retriever()
    rows = []
    language_totals = defaultdict(lambda: {"questions": 0, "hits_at_4": 0})

    for item in questions:
        results = retriever.search(item["question"], k=4)
        retrieved = [chunk.source for chunk in results]
        hit_at_1 = retrieved[0] == item["expected_source"]
        hit_at_4 = item["expected_source"] in retrieved
        language = language_totals[item["language"]]
        language["questions"] += 1
        language["hits_at_4"] += int(hit_at_4)
        rows.append({**item, "retrieved": retrieved, "hit_at_1": hit_at_1, "hit_at_4": hit_at_4})

    total = len(rows)
    report = {
        "date": date.today().isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": 900,
        "chunk_overlap": 150,
        "chunks": len(retriever.chunks),
        "questions": total,
        "hit_at_1": sum(row["hit_at_1"] for row in rows) / total,
        "hit_at_4": sum(row["hit_at_4"] for row in rows) / total,
        "by_language": {
            language: {
                **values,
                "hit_at_4": values["hits_at_4"] / values["questions"],
            }
            for language, values in sorted(language_totals.items())
        },
        "scope": "Retrieval only; answer correctness and citation faithfulness are not scored.",
        "failures": [row for row in rows if not row["hit_at_4"]],
    }

    output = ROOT / "evaluation.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Questions: {total}; chunks: {len(retriever.chunks)}")
    print(f"Hit@1: {report['hit_at_1']:.1%}")
    print(f"Hit@4: {report['hit_at_4']:.1%}")
    print(f"Failures: {len(report['failures'])}")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
