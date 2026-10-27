"""Anlamsal arama: soruyu vektöre çevirip Milvus'ta en yakın chunk'ları bulur.

Çalıştırma:
    python -m retrieval.search "Koç Holding'in CEO'su kim?"
"""

import os
import sys

# Milvus Lite'ın gRPC "fork" bilgi mesajlarını gizle. pymilvus import edilmeden ÖNCE
# ayarlanmalı (gRPC ayarı yüklenirken okur); bu yüzden .env'de işe yaramıyor.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

from openai import OpenAI
from pymilvus import MilvusClient

from indexing.build_index import COLLECTION, get_milvus_client
from indexing.embedder import embed_query, embedding_settings, get_embedding_client

OUTPUT_FIELDS = ["metin", "baslik", "url", "tarih", "chunk_no"]


def check_index_model(milvus: MilvusClient) -> None:
    """İndeks hangi embedding modeliyle kurulduysa sorgu da onunla yapılmalı.

    Farklı modellerin vektörleri karşılaştırılamaz; arama hata vermez ama
    anlamsız sonuç döner. Bu yüzden baştan kontrol ediyoruz.
    """
    model, dims = embedding_settings()
    expected = f"model={model} dim={dims}"
    actual = milvus.describe_collection(COLLECTION)["properties"].get("embedding", "")
    if actual != expected:
        raise SystemExit(f"İndeks '{actual}' ile kurulmuş, .env '{expected}' diyor. "
                         "'python -m indexing.build_index' ile indeksi yeniden kur.")


def year_filter(year_from: int | None, year_to: int | None) -> str:
    """Yıl aralığını Milvus filtre ifadesine çevirir; ör. 2016–2016 → tarih >= "2016" and tarih < "2017".

    tarih metin ("2016-06-04" veya "2016"); bu biçimde metin karşılaştırması yıl sırasıyla aynı.
    Tarihsiz (null) chunk'lar karşılaştırmada "yanlış" sayılır, filtre varken hiç gelmez.
    """
    parts = []
    if year_from is not None:
        parts.append(f'tarih >= "{year_from}"')
    if year_to is not None:
        parts.append(f'tarih < "{year_to + 1}"')
    return " and ".join(parts)


def search(
    milvus: MilvusClient, emb_client: OpenAI, question: str, top_k: int = 5,
    year_from: int | None = None, year_to: int | None = None,
) -> list[dict]:
    """Soruya anlamca en yakın top_k chunk'ı skorlarıyla döndürür; yıl verilirse o aralıkta arar."""
    query_vector = embed_query(emb_client, question)  # sorgu önekiyle: "task: search result | query: ..."
    hits = milvus.search(
        COLLECTION, data=[query_vector], limit=top_k, output_fields=OUTPUT_FIELDS,
        filter=year_filter(year_from, year_to),  # boş metin = filtre yok
    )[0]
    return [{"chunk_id": h["chunk_id"], "skor": h["distance"], **h["entity"]} for h in hits]


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Koç Holding'in CEO'su kim?"
    milvus = get_milvus_client()
    check_index_model(milvus)
    milvus.load_collection(COLLECTION)  # yeni süreçte koleksiyon bellekte değil

    results = search(milvus, get_embedding_client(), question)
    print(f"Soru: {question}\n")
    for i, r in enumerate(results, start=1):
        tarih = r["tarih"] or "tarihsiz"
        print(f"{i}. [{r['skor']:.3f}] {r['baslik']}  ({tarih})")
        print(f"   {r['url']}")
        print(f"   {r['metin'][:220].replace(chr(10), ' ')}...\n")
    milvus.close()


if __name__ == "__main__":
    main()
