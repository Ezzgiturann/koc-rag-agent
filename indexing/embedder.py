"""Chunk'ları embedding'e çevirir; sonuçları diskte önbellekler.

Model ve sunucu .env'den gelir: şu an yerel Ollama + EmbeddingGemma (Gemini'nin
ücretsiz katmanı günde 1.000 embedding ile sınırlı olduğu için, bkz. eval/embedding_karsilastirma.py).
EmbeddingGemma ile gemini-embedding-2 aynı önek formatını kullanır.

Önbellek anahtarı = hash(model + boyut + modele giden metin). Böylece model,
boyut, önek ya da chunk metni değişirse eski vektör kendiliğinden geçersiz olur.

Girdi : data/chunks.jsonl
Çıktı : data/embeddings.jsonl  ({"key": ..., "vector": [...]} satırları)

Çalıştırma:
    python -m indexing.embedder
"""

import hashlib
import json
import os
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI

from agent.ask import get_client
from crawler.fetcher import RateLimiter

CHUNKS_PATH = Path("data/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/embeddings.jsonl")
BATCH_SIZE = 100  # Gemini sınırı: bir istekte en fazla 100 metin (denenerek doğrulandı)
MAX_RETRIES = 5


# EmbeddingGemma ve gemini-embedding-2 "task_type" yerine metnin başına önek istiyor
def as_document(text: str, title: str | None) -> str:
    return f"title: {title or 'none'} | text: {text}"


def as_query(text: str) -> str:
    return f"task: search result | query: {text}"


def embedding_settings() -> tuple[str, int]:
    """.env'den model adı ve vektör boyutu."""
    load_dotenv()  # get_client()'tan önce çağrılabileceği için .env'i burada da yükle
    return os.environ["EMBEDDING_MODEL"], int(os.environ["EMBEDDING_DIM"])


def get_embedding_client() -> OpenAI:
    """Embedding sunucusuna bağlanan istemci (cevap modelinden bağımsız)."""
    load_dotenv()
    base_url = os.environ["EMBEDDING_BASE_URL"]
    if base_url == os.environ["LLM_BASE_URL"]:
        return get_client()  # aynı sağlayıcı (ör. Gemini): aynı anahtarı kullan
    return OpenAI(base_url=base_url, api_key="ollama")  # yerel Ollama anahtar istemez


def cache_key(model: str, dims: int, model_input: str) -> str:
    return hashlib.sha256(f"{model}|{dims}|{model_input}".encode()).hexdigest()


def embed_batch(client: OpenAI, texts: list[str], model: str, dims: int) -> list[list[float]]:
    """Tek istekte en fazla 100 metni vektöre çevirir; geçici hatalarda bekleyip tekrar dener."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(model=model, input=texts, dimensions=dims)
            return [item.embedding for item in response.data]
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as exc:
            if attempt == MAX_RETRIES - 1:
                raise  # son denemeden sonra boşuna bekleme; hatayı olduğu gibi yükselt
            # Üstel geri çekilme: 2, 4, 8, 16 sn bekle. 429 (limit) ve 5xx geçici hatalardır.
            wait = 2 ** (attempt + 1)
            print(f"  {type(exc).__name__}: {str(exc)[:200]}")
            print(f"  → {wait} sn bekleniyor (deneme {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
    raise RuntimeError("buraya ulaşılmamalı")


def embed_query(client: OpenAI, question: str) -> list[float]:
    """Kullanıcı sorusunu sorgu önekiyle vektöre çevirir (arama tarafında kullanılır)."""
    model, dims = embedding_settings()
    return embed_batch(client, [as_query(question)], model, dims)[0]


def load_cache() -> dict[str, list[float]]:
    if not EMBEDDINGS_PATH.exists():
        return {}
    cache = {}
    for line in EMBEDDINGS_PATH.open(encoding="utf-8"):
        row = json.loads(line)
        cache[row["key"]] = row["vector"]
    return cache


def main() -> None:
    model, dims = embedding_settings()
    chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]
    cache = load_cache()

    # Hangi chunk'ların vektörü henüz yok?
    todo: list[tuple[str, str]] = []  # (anahtar, modele gidecek metin)
    for chunk in chunks:
        model_input = as_document(chunk["metin"], chunk["baslik"])
        key = cache_key(model, dims, model_input)
        if key not in cache:
            todo.append((key, model_input))
    print(f"{len(chunks)} chunk | önbellekte {len(chunks) - len(todo)} | üretilecek {len(todo)}")

    client = get_embedding_client()
    # Paketler arası bekleme: yerelde 0; Gemini ücretsiz katmanda 61 sn (dakikada 100 embedding,
    # paketteki her metin ayrı sayılıyor). 2. gündeki fetcher'ın hız sınırlayıcısı.
    limiter = RateLimiter(float(os.environ.get("EMBEDDING_MIN_INTERVAL", "0")))
    start = time.perf_counter()
    with EMBEDDINGS_PATH.open("a", encoding="utf-8") as out:
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            limiter.wait()
            vectors = embed_batch(client, [text for _, text in batch], model, dims)
            for (key, _), vector in zip(batch, vectors):
                out.write(json.dumps({"key": key, "vector": vector}) + "\n")
            out.flush()  # yarıda kesilirse o ana kadarki vektörler diskte kalsın
            print(f"  {min(i + BATCH_SIZE, len(todo))}/{len(todo)} tamam")

    print(f"Bitti: {time.perf_counter() - start:.1f} sn, {(len(todo) + BATCH_SIZE - 1) // BATCH_SIZE} istek")


if __name__ == "__main__":
    main()
