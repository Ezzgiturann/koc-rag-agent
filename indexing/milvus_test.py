"""Milvus Lite'ın kurulu ve çalışır olduğunu gösteren küçük test.

Gerçek embedding yerine elle yazılmış 4 boyutlu vektörler kullanıyoruz.
Her boyutun bir "anlamı" var, böylece arama sonucunu gözle kontrol edebiliriz:
    [otomotiv, enerji, finans, dayanıklı tüketim]
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pymilvus import MilvusClient

COLLECTION = "deneme_sirketler"
DIM = 4  # vektör boyutu; gerçek embedding'de bu 768-3072 arası olacak


def main() -> None:
    load_dotenv()
    uri = os.getenv("VECTOR_DB_URI", "./data/milvus.db")
    Path(uri).parent.mkdir(parents=True, exist_ok=True)  # data/ klasörü yoksa oluştur

    # uri bir dosya yolu ise Milvus Lite çalışır; http://... ise sunucuya bağlanır
    client = MilvusClient(uri=uri)

    # Test her çalıştığında temiz başlasın
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)

    # Koleksiyon = tablo. Hızlı kurulum: id + vector alanları otomatik oluşur.
    # COSINE: iki vektörün yönü ne kadar benzer (1 = aynı yön, 0 = ilgisiz)
    client.create_collection(
        collection_name=COLLECTION,
        dimension=DIM,
        metric_type="COSINE",
    )

    # Şemada tanımlı olmayan alanlar (ad, metin) "dinamik alan" olarak saklanır
    rows = [
        {"id": 1, "vector": [0.9, 0.1, 0.0, 0.2], "ad": "Ford Otosan", "metin": "Ticari araç üretimi"},
        {"id": 2, "vector": [0.1, 0.9, 0.1, 0.0], "ad": "Tüpraş", "metin": "Petrol rafinerisi"},
        {"id": 3, "vector": [0.0, 0.1, 0.9, 0.1], "ad": "Yapı Kredi", "metin": "Bankacılık hizmetleri"},
        {"id": 4, "vector": [0.2, 0.0, 0.1, 0.9], "ad": "Arçelik", "metin": "Beyaz eşya üretimi"},
    ]
    result = client.insert(collection_name=COLLECTION, data=rows)
    print(f"Eklenen kayıt sayısı: {result['insert_count']}")

    # Sorgu vektörü: "otomotive çok, dayanıklı tüketime biraz yakın" bir soru
    query = [0.8, 0.0, 0.0, 0.4]
    hits = client.search(
        collection_name=COLLECTION,
        data=[query],  # birden fazla sorgu gönderilebildiği için liste içinde liste
        limit=3,
        output_fields=["ad", "metin"],
    )

    print(f"\nSorgu vektörü: {query}")
    print("En yakın 3 sonuç:")
    for hit in hits[0]:  # hits[0] = ilk (ve tek) sorgunun sonuçları
        print(f"  skor={hit['distance']:.3f}  {hit['entity']['ad']:<12} → {hit['entity']['metin']}")

    client.close()


if __name__ == "__main__":
    main()
