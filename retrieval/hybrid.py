"""Hibrit arama: vektör araması + BM25, Reciprocal Rank Fusion (RRF) ile birleştirilir.

RRF skorlara değil sıralara bakar: her listede chunk'a 1 / (k + sıra) puan verir
ve puanları toplar. Kosinüs (0–1) ile BM25 (0–30) farklı ölçekte olduğu için
skorları toplamak anlamsız olurdu; sıra ise her yöntemde aynı anlama gelir.

Çalıştırma:
    python -m retrieval.hybrid "Koç Holding'de yatırımcılarla ilişkilerden sorumlu koordinatör kim?"
"""

import sys

from retrieval.search import check_index_model, search  # GRPC ayarı için pymilvus'tan önce

from openai import OpenAI
from pymilvus import MilvusClient

from indexing.build_index import COLLECTION, get_milvus_client
from indexing.embedder import get_embedding_client
from retrieval.bm25 import BM25Index, load_chunks

RRF_K = 60  # Cormack ve ark. (2009); büyüdükçe 1. ile 10. sıra arasındaki fark küçülür
# Her aramadan kaç aday alınır. Ölçüldü (25 soru, Hit@5): 5 → 0,96 · 10 → 0,92 · 20 → 0,88.
# Derin listede iki listenin sonlarındaki zayıf ama "ortak" chunk'lar (1/79 + 1/80)
# tek listede 1. olan chunk'ı (1/61) geçiyor. DEPTH=5'te sonuç k'dan bağımsız (1–60).
DEPTH = 5


def rrf_fuse(ranked_lists: dict[str, list[dict]], k: int = RRF_K) -> list[dict]:
    """Birden çok sıralı listeyi RRF ile tek listeye birleştirir.

    ranked_lists: {"vektor": [...], "bm25": [...]} — her liste en iyiden kötüye.
    Dönen her chunk'ta birleşik "skor" ve her listedeki sırası ("vektor_sira" vb.) var.
    """
    fused: dict[str, dict] = {}  # chunk_id → chunk; aynı chunk iki listede olabilir
    for name, ranked in ranked_lists.items():
        for rank, chunk in enumerate(ranked, start=1):
            entry = fused.setdefault(chunk["chunk_id"], {**chunk, "skor": 0.0})
            entry["skor"] += 1 / (k + rank)
            entry[f"{name}_sira"] = rank  # bu listede yoksa alan hiç oluşmaz

    return sorted(fused.values(), key=lambda c: c["skor"], reverse=True)


def hybrid_search(
    milvus: MilvusClient, emb_client: OpenAI, bm25: BM25Index, question: str,
    top_k: int = 5, depth: int = DEPTH, year_from: int | None = None, year_to: int | None = None,
) -> list[dict]:
    """Vektör ve BM25 aramasını `depth` derinliğinde çalıştırıp RRF ile ilk top_k'yı döndürür.

    Reranker için aday toplarken depth büyütülür: orada RRF sırası değil, doğru chunk'ın
    havuzda olması (recall) önemli.
    """
    ranked_lists = {
        # Filtre her aramanın İÇİNDE uygulanır: sonradan elemek havuzu boşaltabilirdi
        "vektor": search(milvus, emb_client, question, top_k=depth, year_from=year_from, year_to=year_to),
        "bm25": bm25.search(question, k=depth, year_from=year_from, year_to=year_to),
    }
    return rrf_fuse(ranked_lists)[:top_k]


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Koç Holding'de yatırımcılarla ilişkilerden sorumlu koordinatör kim?"
    milvus = get_milvus_client()
    check_index_model(milvus)
    milvus.load_collection(COLLECTION)
    bm25 = BM25Index(load_chunks(), stem=True)  # ölçümde F5 daha iyiydi

    print(f"Soru: {question}\n")
    print(f"{'':>3} {'RRF':>7} {'vektör':>7} {'bm25':>5}  başlık")
    for i, r in enumerate(hybrid_search(milvus, get_embedding_client(), bm25, question), start=1):
        # "-" = bu chunk o aramanın ilk DEPTH sonucunda yok
        vec, bm = r.get("vektor_sira", "-"), r.get("bm25_sira", "-")
        print(f"{i:>2}. {r['skor']:.4f} {vec:>7} {bm:>5}  {r['baslik'][:60]}")
    milvus.close()


if __name__ == "__main__":
    main()
