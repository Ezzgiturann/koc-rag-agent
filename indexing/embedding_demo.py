"""Embedding sezgisi: birkaç cümlenin vektörlerini karşılaştıran küçük deney.

Üç soruya cevap arıyoruz:
  1. Anlamca yakın cümleler (ortak kelime olmasa bile) yakın vektörler mi üretiyor?
  2. gemini-embedding-2'nin "sorgu/belge" önekleri sıralamayı değiştiriyor mu?
  3. 768 boyut, 3072 boyutla aynı sıralamayı veriyor mu?

Çalıştırma:
    python -m indexing.embedding_demo
"""

import os

from openai import OpenAI

from agent.ask import get_client

SENTENCES = {
    "S1": "Tüpraş, Türkiye'nin en büyük petrol rafinerisini işletiyor.",
    "S2": "Koç Topluluğu'nun enerji şirketi ham petrolden akaryakıt üretiyor.",
    "S3": "Arçelik buzdolabı ve çamaşır makinesi üretiyor.",
    "S4": "Yapı Kredi bireysel müşterilere kredi kartı sunuyor.",
    "S5": "Tüpraş operates Turkey's largest oil refinery.",  # S1'in İngilizcesi
    "S6": "Bugün hava çok güzel, parka gidelim.",  # konu dışı
}
QUERY = "Koç'un petrol işleyen şirketi hangisi?"


# gemini-embedding-2 "task_type" parametresi yerine metnin başına önek istiyor (dokümandan)
def as_document(text: str, title: str = "none") -> str:
    return f"title: {title} | text: {text}"


def as_query(text: str) -> str:
    return f"task: search result | query: {text}"


def embed(client: OpenAI, texts: list[str], dims: int) -> list[list[float]]:
    """Metin listesini tek istekte vektörlere çevirir."""
    response = client.embeddings.create(model=os.environ["EMBEDDING_MODEL"], input=texts, dimensions=dims)
    return [item.embedding for item in response.data]


def dot(a: list[float], b: list[float]) -> float:
    """İç çarpım. Vektörler normalize (uzunluğu 1) olduğu için bu doğrudan cosine benzerliği."""
    return sum(x * y for x, y in zip(a, b))


def rank(query_vec: list[float], doc_vecs: dict[str, list[float]]) -> list[tuple[str, float]]:
    """Belgeleri sorguya benzerliğe göre büyükten küçüğe sıralar."""
    scores = [(key, dot(query_vec, vec)) for key, vec in doc_vecs.items()]
    return sorted(scores, key=lambda pair: pair[1], reverse=True)


def main() -> None:
    client = get_client()
    keys = list(SENTENCES)

    # --- Deney 1: cümleler arası benzerlik matrisi (768 boyut, belge öneki) ---
    vecs = embed(client, [as_document(SENTENCES[k]) for k in keys], dims=768)
    doc_vecs = dict(zip(keys, vecs))
    print("Deney 1 — benzerlik matrisi (1 = aynı anlam)")
    print("      " + "  ".join(f"{k:>5}" for k in keys))
    for a in keys:
        print(f"{a:>5} " + "  ".join(f"{dot(doc_vecs[a], doc_vecs[b]):5.2f}" for b in keys))

    # --- Deney 2: sorgu öneki var / yok ---
    print(f"\nDeney 2 — sorgu: {QUERY!r}")
    with_prefix, without_prefix = embed(client, [as_query(QUERY), QUERY], dims=768)
    for label, qvec in [("önekli ", with_prefix), ("öneksiz", without_prefix)]:
        print(f"  {label}: " + "  ".join(f"{k}={s:.2f}" for k, s in rank(qvec, doc_vecs)))

    # --- Deney 3: 768 ve 3072 boyut aynı sırayı veriyor mu? ---
    print("\nDeney 3 — boyut karşılaştırması (önekli sorgu)")
    for dims in (768, 3072):
        vecs = embed(client, [as_query(QUERY)] + [as_document(SENTENCES[k]) for k in keys], dims=dims)
        ranking = rank(vecs[0], dict(zip(keys, vecs[1:])))
        print(f"  {dims:>4} boyut: " + "  ".join(f"{k}={s:.2f}" for k, s in ranking))


if __name__ == "__main__":
    main()
