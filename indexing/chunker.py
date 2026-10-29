"""Sayfaları ~400 token'lık, 60 token örtüşmeli parçalara (chunk) böler.

En küçük birim cümledir: chunk'lar cümle ortasında bitmez. Yeni chunk, önceki
chunk'ın son cümleleriyle (en fazla OVERLAP_TOKENS) başlar.

Markdown tablolarında (PDF'lerden) en küçük birim SATIRDIR. Tablo iki chunk'a bölünürse
yeni chunk tablonun başlık satırlarıyla başlar: "20.306" hangi sütunun (yılın) değeri,
her parçada belli olsun.

Girdi : data/pages.jsonl (HTML) + data/pdf/*.pages.jsonl (PDF)
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
PDF_PAGES_GLOB = "data/pdf/*.pages.jsonl"
CHUNKS_PATH = Path("data/chunks.jsonl")
MAX_TOKENS = 400
OVERLAP_TOKENS = 60

ENCODING = tiktoken.get_encoding("o200k_base")  # bir kez yükle, her çağrıda tekrar yükleme

# Cümle sonu: . ! ? … işaretinden sonra boşluk ve büyük harf/rakam/tırnak geliyorsa.
# Bilinen kusur: "A.Ş. Yönetim" gibi kısaltmalarda da böler (örtüşme bu kaybı azaltır).
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÇĞİÖŞÜ0-9\"“‘'(])")
TABLE_SEPARATOR = re.compile(r"^\|[-:| ]+\|$")  # |---|---| : başlık ile gövdeyi ayıran satır
MAX_BYTES = 8192  # Milvus şemasındaki metin alanı sınırı (bayt)

# Birim: (metin, token sayısı, tablo başlığı). Tablo satırında başlık dolu, düz cümlede None.
Unit = tuple[str, int, str | None]


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


def split_units(text: str) -> list[tuple[str, str | None]]:
    """Metni birimlere ayırır: düz metin → cümleler; markdown tablosu → satırlar (başlığıyla)."""
    units: list[tuple[str, str | None]] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("|"):
            # Tablo olmayan ardışık satırlar: eskisi gibi cümlelere
            start = i
            while i < len(lines) and not lines[i].strip().startswith("|"):
                i += 1
            units.extend((s, None) for s in split_sentences("\n".join(lines[start:i])))
            continue
        start = i
        while i < len(lines) and lines[i].strip().startswith("|"):
            i += 1
        # Docling hücreleri boşlukla hizalar; token az ama bayt çok → fazla boşluğu sıkıştır
        table = [re.sub(r" {2,}", " ", line.strip()) for line in lines[start:i]]
        # Docling başlıktan sonra |---| ayracı koyar; trafilatura (HTML) koymaz, başlık yine ilk satırdır
        n_header = 2 if len(table) >= 2 and TABLE_SEPARATOR.match(table[1]) else 1
        header = "\n".join(table[:n_header])
        units.append((header, header))  # başlığın kendisi de tablonun parçası
        units.extend((row, header) for row in table[n_header:])
    return units


def join_units(units: list[Unit]) -> str:
    """Birimleri birleştirir: cümleler boşlukla, tablo satırları alt satırla (markdown bozulmasın)."""
    text = ""
    for k, (piece, _, header) in enumerate(units):
        if k:
            text += "\n" if header is not None or units[k - 1][2] is not None else " "
        text += piece
    return text


def split_long_sentence(sentence: str, max_tokens: int) -> list[str]:
    """MAX_TOKENS'tan uzun tek cümleyi token sınırında parçalara keser (nadir durum)."""
    tokens = ENCODING.encode(sentence)
    return [ENCODING.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


def overlap_tail(units: list[Unit], limit: int) -> list[Unit]:
    """Listenin sonundan, toplamı `limit` token'ı geçmeyecek kadar cümle alır."""
    tail: list[Unit] = []
    total = 0
    for unit in reversed(units):
        if total + unit[1] > limit:
            break
        tail.insert(0, unit)
        total += unit[1]
    return tail


def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    """Metni cümle sınırlarına saygı göstererek örtüşmeli chunk'lara böler."""
    # Her birimi token sayısıyla birlikte tut; uzun cümleleri önce parçala
    units: list[Unit] = []
    for piece, header in split_units(text):
        n = count_tokens(piece)
        parts = [piece] if n <= max_tokens else split_long_sentence(piece, max_tokens)
        units.extend((p, count_tokens(p), header) for p in parts)

    chunks: list[str] = []
    current: list[Unit] = []
    current_tokens = 0
    for unit in units:
        piece, n, header = unit
        if current and current_tokens + n > max_tokens:
            chunks.append(join_units(current))
            if header is not None and header != piece:
                # Tablo ortasında bölündü: yeni chunk başlıkla başlar (satır örtüşmesi yok)
                current = [(header, count_tokens(header), header)]
            else:
                current = overlap_tail(current, overlap)  # yeni chunk önceki chunk'ın sonuyla başlar
                if any(u[2] is not None for u in current):
                    current = []  # tablo satırlarıyla (başlıksız) başlamasın
            if sum(u[1] for u in current) + n > max_tokens:
                current = []  # örtüşme + yeni birim sığmıyorsa örtüşmeden vazgeç
            current_tokens = sum(u[1] for u in current)
        current.append(unit)
        current_tokens += n
    if current:
        chunks.append(join_units(current))
    return chunks


def main() -> None:
    pages = [json.loads(line) for line in PAGES_PATH.open(encoding="utf-8")]
    for path in sorted(Path().glob(PDF_PAGES_GLOB)):  # PDF sayfaları HTML'den SONRA: eski chunk sırası korunur
        pages += [json.loads(line) for line in path.open(encoding="utf-8")]
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
                "kaynak_tur": page.get("kaynak_tur", "html"),
            })
            if len(text.encode()) > MAX_BYTES:
                raise SystemExit(f"Chunk Milvus sınırını aşıyor ({len(text.encode())} bayt): {page['url']}")

    with CHUNKS_PATH.open("w", encoding="utf-8") as out:
        for chunk in all_chunks:
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    sizes = sorted(c["token"] for c in all_chunks)
    per_page = [sum(1 for c in all_chunks if c["url"] == p["url"]) for p in pages]
    print(f"{len(pages)} sayfa → {len(all_chunks)} chunk  ({CHUNKS_PATH})")
    for kind in ("html", "pdf"):
        n_kind = sum(c["kaynak_tur"] == kind for c in all_chunks)
        n_table = sum(c["kaynak_tur"] == kind and "\n|" in c["metin"] for c in all_chunks)
        print(f"  {kind}: {n_kind} chunk ({n_table} tanesi tablo içeriyor)")
    print(f"Chunk token: min {sizes[0]} | medyan {int(statistics.median(sizes))} | ortalama {statistics.mean(sizes):.0f} | max {sizes[-1]}")
    print(f"Tek chunk'lık sayfa: {per_page.count(1)} | en çok chunk'lı sayfa: {max(per_page)}")
    print(f"Embedding'e gidecek toplam token (örtüşme dahil): {sum(sizes)}")


if __name__ == "__main__":
    main()
