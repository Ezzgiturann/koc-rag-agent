"""Retrieval değerlendirmesi: bir arama fonksiyonunu test setiyle ölçer.

Her sorunun "kanıt" ifadeleri var; sonuçtaki bir chunk'ın başlığında veya
metninde bunlardan biri geçiyorsa doğru bilgi bulunmuş sayılır. Başlığa da
bakıyoruz: özgeçmiş sayfalarında kişinin adı metinde değil, yalnızca başlıkta.

Metrikler (embedding_karsilastirma.py ile aynı):
  Hit@1  : doğru chunk 1. sırada mı?
  Hit@5  : doğru chunk ilk 5'te mi? (RAG'de modele 5 chunk veriyoruz)
  MRR@10 : 1 / (doğru chunk'ın sırası); ilk 10'da yoksa 0.

Arama fonksiyonu parametre olarak verilir; hibrit aramayı da aynı kodla ölçeceğiz.

Çalıştırma:
    python -m eval.retrieval_eval
"""

import json
from collections.abc import Callable
from pathlib import Path

from retrieval.search import check_index_model, search  # GRPC ayarı için pymilvus'tan önce

from indexing.build_index import COLLECTION, get_milvus_client
from indexing.embedder import get_embedding_client

QUESTIONS_PATH = Path("eval/retrieval_sorulari.json")
CHUNKS_PATH = Path("data/chunks.jsonl")
TOP_K = 10

# (soru, k) → sıralı chunk listesi. Vektör, BM25 ve hibrit arama bu imzaya uyacak.
SearchFn = Callable[[str, int], list[dict]]


def has_evidence(chunk: dict, evidence: list[str]) -> bool:
    """Chunk'ın başlığında veya metninde kanıt ifadelerinden biri geçiyor mu?"""
    text = f"{chunk['baslik']}\n{chunk['metin']}"
    return any(e in text for e in evidence)


def first_hit_rank(chunks: list[dict], evidence: list[str]) -> int | None:
    """Kanıt içeren ilk chunk'ın sırası (1'den başlar); yoksa None."""
    for rank, chunk in enumerate(chunks, start=1):
        if has_evidence(chunk, evidence):
            return rank
    return None


def check_questions(questions: list[dict]) -> None:
    """Kanıtı hiçbir chunk'ta olmayan soru ölçülemez; baştan yakala."""
    chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]
    for q in questions:
        count = sum(has_evidence(c, q["kanit"]) for c in chunks)
        if count == 0:
            raise SystemExit(f"Kanıt hiçbir chunk'ta yok: {q['soru']}")


def evaluate(search_fn: SearchFn, questions: list[dict]) -> list[dict]:
    """Her soru için aramayı çalıştırır; sırayı ve ilk 3 sonucu kaydeder."""
    results = []
    for q in questions:
        ranked = search_fn(q["soru"], TOP_K)
        results.append({
            **q,
            "sira": first_hit_rank(ranked, q["kanit"]),
            "ilk3": [c["baslik"] for c in ranked[:3]],
        })
    return results


def metrics(results: list[dict]) -> dict[str, float]:
    """Hit@1, Hit@5 ve MRR@10'u hesaplar."""
    n = len(results)
    ranks = [r["sira"] for r in results]
    return {
        "Hit@1": sum(r == 1 for r in ranks) / n,
        "Hit@5": sum(r is not None and r <= 5 for r in ranks) / n,
        "MRR@10": sum(1 / r for r in ranks if r is not None) / n,
    }


def report(results: list[dict]) -> None:
    """Türe göre ve toplam metrikleri, ardından ilk 5'e giremeyen soruları yazdırır."""
    # Ortalama tek başına yanıltabilir: bir türdeki iyileşme diğerindeki kötüleşmeyi gizler
    groups = sorted({r["tur"] for r in results})
    print(f"{'tür':<10}{'n':>4}{'Hit@1':>8}{'Hit@5':>8}{'MRR@10':>8}")
    for name, group in [(g, [r for r in results if r["tur"] == g]) for g in groups] + [("TOPLAM", results)]:
        m = metrics(group)
        print(f"{name:<10}{len(group):>4}{m['Hit@1']:>8.2f}{m['Hit@5']:>8.2f}{m['MRR@10']:>8.3f}")

    misses = [r for r in results if r["sira"] is None or r["sira"] > 5]
    print(f"\nİlk 5'e giremeyen sorular ({len(misses)}):")
    for r in misses:
        print(f"- [{r['tur']}] {r['soru']}  → sıra: {r['sira'] or '10+'}")
        for title in r["ilk3"]:
            print(f"      · {title}")


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    check_questions(questions)

    milvus = get_milvus_client()
    check_index_model(milvus)
    milvus.load_collection(COLLECTION)
    emb_client = get_embedding_client()

    # Mevcut vektör aramasını SearchFn imzasına uyduruyoruz
    def vector_search(question: str, k: int) -> list[dict]:
        return search(milvus, emb_client, question, top_k=k)

    print("Arama: vektör (EmbeddingGemma + Milvus FLAT)\n")
    report(evaluate(vector_search, questions))
    milvus.close()


if __name__ == "__main__":
    main()
