"""Reranker: hibrit aramanın adaylarını cross-encoder ile yeniden sıralar.

Vektör araması soru ve chunk'ı AYRI ayrı vektöre çevirip karşılaştırır (bi-encoder).
Cross-encoder ikisini BİRLİKTE okur ve tek bir alaka skoru üretir: kelimeler arası
etkileşimi gördüğü için daha isabetli ama çok daha yavaş. Bu yüzden 1.860 chunk'a
değil, hibrit aramanın getirdiği ~24 adaya uygulanır.

Model: BAAI/bge-reranker-v2-m3 (çok dilli, Apache 2.0). İlk çalıştırmada ~2 GB indirilir
ve Hugging Face önbelleğine (~/.cache/huggingface) yazılır; yalnızca yerelde çalışır.

Çalıştırma:
    python -m retrieval.rerank "Koç Holding Ekonomik Araştırmalar Koordinatörü kimdir?"
"""

import sys
import time

from retrieval.search import check_index_model  # GRPC ayarı için pymilvus'tan önce

from openai import OpenAI
from pymilvus import MilvusClient
from sentence_transformers import CrossEncoder

from indexing.build_index import COLLECTION, get_milvus_client
from indexing.embedder import get_embedding_client
from retrieval.bm25 import BM25Index, load_chunks
from retrieval.hybrid import hybrid_search

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# Aday havuzu: her aramadan ilk 15 → RRF birleşimi ~24 aday. Ölçüldü: derinlik 10'dan
# itibaren doğru chunk 25/25 soruda havuzda; 15 küçük bir güvenlik payı.
CANDIDATE_DEPTH = 15
MAX_LENGTH = 512  # soru + chunk token sınırı; chunk'larımız ~400 token, sığıyor


def load_reranker() -> CrossEncoder:
    """Modeli yükler; Mac'te Apple GPU'sunu (MPS) kendiliğinden kullanır."""
    return CrossEncoder(RERANK_MODEL, max_length=MAX_LENGTH)


def rerank(reranker: CrossEncoder, question: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Chunk'ları soruyla alakasına göre yeniden sıralar; ilk top_k'yı döndürür."""
    # Başlık da girdide: özgeçmiş sayfalarında kişinin adı yalnızca başlıkta
    pairs = [(question, f"{c['baslik']}\n{c['metin']}") for c in chunks]
    scores = reranker.predict(pairs)  # her (soru, chunk) çifti için bir alaka skoru
    for c, s in zip(chunks, scores):
        c["rerank_skor"] = float(s)
    return sorted(chunks, key=lambda c: c["rerank_skor"], reverse=True)[:top_k]


def search_and_rerank(
    milvus: MilvusClient, emb_client: OpenAI, bm25: BM25Index, reranker: CrossEncoder,
    question: str, top_k: int = 5,
) -> list[dict]:
    """Hibrit aramayla geniş aday havuzu topla, reranker ile en iyi top_k'yı seç."""
    candidates = hybrid_search(milvus, emb_client, bm25, question, top_k=100, depth=CANDIDATE_DEPTH)
    return rerank(reranker, question, candidates, top_k)


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Koç Holding Ekonomik Araştırmalar Koordinatörü kimdir?"
    milvus = get_milvus_client()
    check_index_model(milvus)
    milvus.load_collection(COLLECTION)
    bm25 = BM25Index(load_chunks(), stem=True)

    start = time.perf_counter()
    reranker = load_reranker()
    print(f"Model yüklendi: {time.perf_counter() - start:.1f} sn ({reranker.device})\n")

    start = time.perf_counter()
    results = search_and_rerank(milvus, get_embedding_client(), bm25, reranker, question)
    print(f"Soru: {question}  ({time.perf_counter() - start:.1f} sn)\n")
    print(f"{'':>3} {'rerank':>7} {'vektör':>7} {'bm25':>5}  başlık")
    for i, r in enumerate(results, start=1):
        vec, bm = r.get("vektor_sira", "-"), r.get("bm25_sira", "-")
        print(f"{i:>2}. {r['rerank_skor']:7.3f} {vec:>7} {bm:>5}  {r['baslik'][:60]}")
    milvus.close()


if __name__ == "__main__":
    main()
