"""Chunk'ları ve embedding'lerini Milvus'a yazar (koleksiyonu her seferinde baştan kurar).

1. gündeki testten farkı: şemayı kendimiz tanımlıyoruz (hangi alan, hangi tür,
hangi uzunluk). Böylece yanlış tipte veri yazılırsa Milvus hata verir.

Girdi : data/chunks.jsonl + data/embeddings.jsonl (önce indexing.embedder çalışmalı)
Çıktı : Milvus koleksiyonu "koc_chunks"

Çalıştırma:
    python -m indexing.build_index
"""

import json
import os
import random

from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient

from indexing.embedder import CHUNKS_PATH, as_document, cache_key, embedding_settings, load_cache

COLLECTION = "koc_chunks"
INSERT_BATCH = 500


def get_milvus_client() -> MilvusClient:
    load_dotenv()
    return MilvusClient(uri=os.environ["VECTOR_DB_URI"])  # dosya yolu → Milvus Lite, http → sunucu


def create_collection(client: MilvusClient, model: str, dims: int) -> None:
    """Koleksiyonu şema ve vektör indeksiyle sıfırdan oluşturur."""
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)  # baştan kur: eski/yarım veri kalmasın

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=32, is_primary=True)  # ör. "a3f9c2e81b4d-2"
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dims)
    schema.add_field("metin", DataType.VARCHAR, max_length=8192)  # bayt cinsinden; en uzun chunk ~2.200
    schema.add_field("baslik", DataType.VARCHAR, max_length=512)
    schema.add_field("url", DataType.VARCHAR, max_length=512)
    schema.add_field("tarih", DataType.VARCHAR, max_length=16, nullable=True)  # sabit sayfalarda boş
    schema.add_field("chunk_no", DataType.INT64)

    # FLAT: sorguyu her vektörle tek tek karşılaştırır → her zaman kesin sonuç.
    # ~2.000 vektörde milisaniyeler sürer; milyonlarda HNSW gibi yaklaşık yöntemler gerekir.
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index("vector", index_type="FLAT", metric_type="COSINE")

    client.create_collection(COLLECTION, schema=schema, index_params=index_params)
    # Modeli koleksiyona not et: arama farklı modelle yapılırsa fark edebilsin.
    # (Şema "description" alanı Milvus Lite'ta saklanmıyor; "properties" saklanıyor.)
    client.alter_collection_properties(COLLECTION, properties={"embedding": f"model={model} dim={dims}"})


def main() -> None:
    model, dims = embedding_settings()
    chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]
    cache = load_cache()

    # Her chunk'ın vektörünü önbellekten bul (anahtar embedder.py ile birebir aynı üretilmeli)
    rows = []
    for c in chunks:
        key = cache_key(model, dims, as_document(c["metin"], c["baslik"]))
        if key not in cache:
            raise SystemExit(f"Vektör yok: {c['chunk_id']}. Önce 'python -m indexing.embedder' çalıştır.")
        rows.append({
            "chunk_id": c["chunk_id"], "vector": cache[key], "metin": c["metin"],
            "baslik": c["baslik"], "url": c["url"], "tarih": c["tarih"], "chunk_no": c["chunk_no"],
        })

    client = get_milvus_client()
    create_collection(client, model, dims)
    for i in range(0, len(rows), INSERT_BATCH):
        client.insert(COLLECTION, rows[i:i + INSERT_BATCH])

    # Strong: sorgu, o ana kadarki tüm yazmalar görünür olana kadar bekler. Varsayılan (Bounded)
    # Standalone'da son birkaç saniyenin yazmalarını henüz görmeyebilir (7. gün: 4.173 yerine 3.000);
    # Lite tek süreçli olduğu için bu fark orada görünmüyordu.
    count = client.query(COLLECTION, filter="", output_fields=["count(*)"],
                         consistency_level="Strong")[0]["count(*)"]
    print(f"'{COLLECTION}' koleksiyonuna {count} chunk yazıldı ({model}, {dims} boyut)")

    # Sağlama: bir chunk'ın kendi vektörüyle arama yapınca ilk sonuç kendisi olmalı (skor ≈ 1)
    for row in random.sample(rows, 3):
        hit = client.search(COLLECTION, data=[row["vector"]], limit=1, output_fields=["baslik"],
                            consistency_level="Strong")[0][0]
        # Sonuçta birincil anahtar, şemadaki adıyla ("chunk_id") gelir
        status = "✓" if hit["chunk_id"] == row["chunk_id"] else "✗"
        print(f"  {status} {row['chunk_id']} → {hit['chunk_id']} (skor {hit['distance']:.4f}) {hit['entity']['baslik'][:50]}")

    client.close()


if __name__ == "__main__":
    main()
