"""BM25 anahtar kelime araması: kendi yazdığımız, bellekte duran bir indeks.

Vektör araması anlamı karşılaştırır; BM25 ise ortak kelimeleri sayar ve nadir
kelimelere ("Çimenoğlu") yaygın olanlardan ("Koç") çok daha fazla puan verir.

Çalıştırma:
    python -m retrieval.bm25 "Koç Holding Ekonomik Araştırmalar Koordinatörü kimdir?"
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

CHUNKS_PATH = Path("data/chunks.jsonl")
WORD = re.compile(r"\w+")  # Unicode harf/rakam dizileri; ' ve ’ ayırıcı olur: Suna'nın → suna, nın


def turkish_lower(text: str) -> str:
    """Türkçe kurallarıyla küçük harf: İ → i, I → ı.

    Python'un lower()'ı "İ"yi "i" + görünmez birleşik nokta (U+0307) yapar,
    "I"yı da "i" yapar; ikisi de Türkçe için yanlış.
    """
    return text.replace("İ", "i").replace("I", "ı").lower()


def tokenize(text: str, stem: bool = False) -> list[str]:
    """Metni küçük harfli kelimelere böler; stem=True ise her kelimeyi ilk 5 harfe keser (F5)."""
    tokens = WORD.findall(turkish_lower(text))
    if stem:
        # F5: Türkçe sondan eklemeli; kökün çoğu ilk 5 harfte. koordinatörü → koord
        tokens = [t[:5] for t in tokens]
    return tokens


class BM25Index:
    """Ters indeks (kelime → hangi chunk'ta kaç kez) ve BM25 puanlaması."""

    def __init__(self, chunks: list[dict], stem: bool = False, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.stem = stem
        self.k1 = k1  # TF doygunluğu: büyüdükçe tekrar eden kelime daha çok puan alır
        self.b = b  # uzunluk normalizasyonu: 0 = yok, 1 = tam

        # Başlık da indekste: özgeçmiş sayfalarında kişinin adı yalnızca başlıkta
        docs = [tokenize(f"{c['baslik']}\n{c['metin']}", stem) for c in chunks]
        self.doc_len = [len(d) for d in docs]
        self.avg_len = sum(self.doc_len) / len(docs)

        # Ters indeks: her kelime için {chunk sırası: o chunk'ta kaç kez geçtiği}
        self.postings: dict[str, dict[int, int]] = {}
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings.setdefault(term, {})[i] = tf

    def idf(self, term: str) -> float:
        """Nadirlik puanı: kelime ne kadar az chunk'ta geçiyorsa o kadar yüksek."""
        n = len(self.chunks)
        df = len(self.postings.get(term, {}))  # kelimenin geçtiği chunk sayısı
        # +1 içeride: çok yaygın kelimede bile negatif olmaz (Lucene'in kullandığı biçim)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, question: str, k: int = 5) -> list[dict]:
        """Soruya en yüksek BM25 puanlı k chunk'ı döndürür (vektör aramasıyla aynı alanlar)."""
        scores: Counter[int] = Counter()
        for term in set(tokenize(question, self.stem)):  # sorudaki tekrarları saymıyoruz
            idf = self.idf(term)
            for i, tf in self.postings.get(term, {}).items():
                # Uzun chunk'ın tf'si biraz kırpılır; tf arttıkça puan k1+1'e doğru doyar
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm)

        return [{**self.chunks[i], "skor": score} for i, score in scores.most_common(k)]


def load_chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Koç Holding Ekonomik Araştırmalar Koordinatörü kimdir?"
    chunks = load_chunks()
    print(f"Soru: {question}")

    for stem in (False, True):
        index = BM25Index(chunks, stem=stem)
        tokens = tokenize(question, stem)
        print(f"\n--- {'F5 (ilk 5 harf)' if stem else 'Ek işlemi yok'} · {len(index.postings):,} farklı kelime")
        # Her soru kelimesinin nadirliği: hangi kelimenin sonucu belirlediğini görelim
        print("    idf: " + ", ".join(f"{t}={index.idf(t):.1f}" for t in dict.fromkeys(tokens)))
        for rank, r in enumerate(index.search(question), start=1):
            print(f"{rank}. [{r['skor']:.1f}] {r['baslik'][:60]}  ({r['tarih'] or 'tarihsiz'})")


if __name__ == "__main__":
    main()
