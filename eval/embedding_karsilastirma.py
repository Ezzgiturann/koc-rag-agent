"""İki yerel embedding modelini küçük bir test setiyle karşılaştırır.

Her soru için "kanıt" ifadeleri var: arama sonucundaki chunk bunlardan HERHANGİ
birini içeriyorsa doğru bilgi bulunmuş sayılır. Aynı bilgi birden çok sayfada ve
farklı yazımlarla geçebilir ("%8", "% 8", "yüzde 8"); hepsini listelemek gerekir.

Metrikler:
  Hit@1  : doğru chunk 1. sırada mı?
  Hit@5  : doğru chunk ilk 5'te mi? (RAG'de modele 5 chunk vereceğiz)
  MRR@10 : 1 / (doğru chunk'ın sırası); ilk 10'da yoksa 0. Sorular üzerinden ortalama.

Üretilen vektörler indexing/embedder.py'nin önbelleğine yazılır; kazanan model
için indekslemede yeniden üretilmez.

Çalıştırma:
    python -m eval.embedding_karsilastirma
"""

import json
import time
from pathlib import Path

from openai import OpenAI

from indexing.embedder import EMBEDDINGS_PATH, cache_key, load_cache

QUESTIONS_PATH = Path("eval/embedding_sorulari.json")
CHUNKS_PATH = Path("data/chunks.jsonl")
DIMS = 768
BATCH = 64
TOP_K = 10

# Her modelin kendi dokümanında önerdiği önek formatı
MODELS = {
    "embeddinggemma:300m": {
        "doc": lambda title, text: f"title: {title or 'none'} | text: {text}",
        "query": lambda q: f"task: search result | query: {q}",
    },
    "qwen3-embedding:0.6b": {
        # Qwen: belgeye talimat yok (başlığı metnin başına koyuyoruz), sorguya talimat var
        "doc": lambda title, text: f"{title}\n{text}" if title else text,
        "query": lambda q: f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{q}",
    },
}


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def embed_all(client: OpenAI, model: str, texts: list[str], cache: dict) -> list[list[float]]:
    """Metinleri önbellek + yerel modelle vektöre çevirir; yenileri önbelleğe ekler."""
    keys = [cache_key(model, DIMS, t) for t in texts]
    missing = [(k, t) for k, t in zip(keys, texts) if k not in cache]
    with EMBEDDINGS_PATH.open("a", encoding="utf-8") as out:
        for i in range(0, len(missing), BATCH):
            batch = missing[i:i + BATCH]
            response = client.embeddings.create(model=model, input=[t for _, t in batch], dimensions=DIMS)
            for (key, _), item in zip(batch, response.data):
                cache[key] = item.embedding
                out.write(json.dumps({"key": key, "vector": item.embedding}) + "\n")
    return [cache[k] for k in keys]


def first_hit_rank(ranked_chunks: list[dict], evidence: list[str]) -> int | None:
    """Kanıt ifadelerinden birini içeren ilk chunk'ın sırası (1'den başlar)."""
    for rank, chunk in enumerate(ranked_chunks, start=1):
        if any(e in chunk["metin"] for e in evidence):
            return rank
    return None


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # yerel Ollama
    cache = load_cache()

    # Test setinin kendisini kontrol et: kanıtı hiçbir chunk'ta olmayan soru anlamsızdır
    for q in questions:
        n = sum(any(e in c["metin"] for e in q["kanit"]) for c in chunks)
        assert n > 0, f"Kanıt bulunamadı: {q['kanit']}"

    ranks: dict[str, list[int | None]] = {}
    for model, fmt in MODELS.items():
        start = time.perf_counter()
        doc_vecs = embed_all(client, model, [fmt["doc"](c["baslik"], c["metin"]) for c in chunks], cache)
        query_vecs = embed_all(client, model, [fmt["query"](q["soru"]) for q in questions], cache)
        print(f"{model}: embedding {time.perf_counter() - start:.0f} sn")

        ranks[model] = []
        for q, qv in zip(questions, query_vecs):
            scores = sorted(((dot(qv, dv), c) for dv, c in zip(doc_vecs, chunks)), key=lambda p: p[0], reverse=True)
            ranks[model].append(first_hit_rank([c for _, c in scores[:TOP_K]], q["kanit"]))

    # Soru bazında sonuç: doğru chunk kaçıncı sırada geldi ("-" = ilk 10'da yok)
    names = list(MODELS)
    print(f"\n{'Soru':<62}" + "".join(f"{n.split(':')[0]:>16}" for n in names))
    for i, q in enumerate(questions):
        print(f"{q['soru'][:60]:<62}" + "".join(f"{(ranks[n][i] or '-'):>16}" for n in names))

    print()
    for n in names:
        r = ranks[n]
        hit1 = sum(x == 1 for x in r) / len(r)
        hit5 = sum(x is not None and x <= 5 for x in r) / len(r)
        mrr = sum(1 / x for x in r if x) / len(r)
        print(f"{n:<22} Hit@1 {hit1:.2f}   Hit@5 {hit5:.2f}   MRR@10 {mrr:.3f}")


if __name__ == "__main__":
    main()
