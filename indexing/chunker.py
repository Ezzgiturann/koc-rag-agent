"""Sayfaları ~400 token'lık, 60 token örtüşmeli parçalara (chunk) böler.

En küçük birim cümledir: chunk'lar cümle ortasında bitmez. Yeni chunk, önceki
chunk'ın son cümleleriyle (en fazla OVERLAP_TOKENS) başlar.

Girdi : data/pages.jsonl
Çıktı : data/chunks.jsonl

Çalıştırma:
    python -m indexing.chunker
"""

import hashlib
import json
import re
import statistics
from pathlib import Path

import tiktoken

PAGES_PATH = Path("data/pages.jsonl")
CHUNKS_PATH = Path("data/chunks.jsonl")
MAX_TOKENS = 400
OVERLAP_TOKENS = 60

ENCODING = tiktoken.get_encoding("o200k_base")  # bir kez yükle, her çağrıda tekrar yükleme

# Cümle sonu: . ! ? … işaretinden sonra boşluk ve büyük harf/rakam/tırnak geliyorsa.
# Bilinen kusur: "A.Ş. Yönetim" gibi kısaltmalarda da böler (örtüşme bu kaybı azaltır).
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÇĞİÖŞÜ0-9\"“‘'(])")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def split_sentences(text: str) -> list[str]:
    """Metni önce paragraflara, sonra cümlelere ayırır."""
    sentences = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if paragraph:
            sentences.extend(s.strip() for s in SENTENCE_END.split(paragraph) if s.strip())
    return sentences


def split_long_sentence(sentence: str, max_tokens: int) -> list[str]:
    """MAX_TOKENS'tan uzun tek cümleyi token sınırında parçalara keser (nadir durum)."""
    tokens = ENCODING.encode(sentence)
    return [ENCODING.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


def overlap_tail(units: list[tuple[str, int]], limit: int) -> list[tuple[str, int]]:
    """Listenin sonundan, toplamı `limit` token'ı geçmeyecek kadar cümle alır."""
    tail: list[tuple[str, int]] = []
    total = 0
    for sentence, n in reversed(units):
        if total + n > limit:
            break
        tail.insert(0, (sentence, n))
        total += n
    return tail


def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    """Metni cümle sınırlarına saygı göstererek örtüşmeli chunk'lara böler."""
    # Her cümleyi token sayısıyla birlikte tut; uzun cümleleri önce parçala
    units: list[tuple[str, int]] = []
    for sentence in split_sentences(text):
        n = count_tokens(sentence)
        pieces = [sentence] if n <= max_tokens else split_long_sentence(sentence, max_tokens)
        units.extend((p, count_tokens(p)) for p in pieces)

    chunks: list[str] = []
    current: list[tuple[str, int]] = []
    current_tokens = 0
    for sentence, n in units:
        if current and current_tokens + n > max_tokens:
            chunks.append(" ".join(s for s, _ in current))
            current = overlap_tail(current, overlap)  # yeni chunk önceki chunk'ın sonuyla başlar
            if sum(c for _, c in current) + n > max_tokens:
                current = []  # örtüşme + yeni cümle sığmıyorsa örtüşmeden vazgeç
            current_tokens = sum(c for _, c in current)
        current.append((sentence, n))
        current_tokens += n
    if current:
        chunks.append(" ".join(s for s, _ in current))
    return chunks


def main() -> None:
    pages = [json.loads(line) for line in PAGES_PATH.open(encoding="utf-8")]
    all_chunks = []
    for page in pages:
        url_hash = hashlib.sha256(page["url"].encode()).hexdigest()[:12]
        for i, text in enumerate(chunk_text(page["metin"])):
            all_chunks.append({
                "chunk_id": f"{url_hash}-{i}",  # sayfa + sıra: tekrar çalıştırınca aynı id
                "url": page["url"],
                "baslik": page["baslik"],
                "tarih": page["tarih"],
                "chunk_no": i,
                "metin": text,
                "token": count_tokens(text),
            })

    with CHUNKS_PATH.open("w", encoding="utf-8") as out:
        for chunk in all_chunks:
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    sizes = sorted(c["token"] for c in all_chunks)
    per_page = [sum(1 for c in all_chunks if c["url"] == p["url"]) for p in pages]
    print(f"{len(pages)} sayfa → {len(all_chunks)} chunk  ({CHUNKS_PATH})")
    print(f"Chunk token: min {sizes[0]} | medyan {int(statistics.median(sizes))} | ortalama {statistics.mean(sizes):.0f} | max {sizes[-1]}")
    print(f"Tek chunk'lık sayfa: {per_page.count(1)} | en çok chunk'lı sayfa: {max(per_page)}")
    print(f"Embedding'e gidecek toplam token (örtüşme dahil): {sum(sizes)}")


if __name__ == "__main__":
    main()
