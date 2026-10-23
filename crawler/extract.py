"""Önbellekteki ham HTML'lerden ana metni çıkarır ve data/pages.jsonl'a yazar.

Siteye hiç istek atmaz; sadece data/raw/ altındaki dosyaları okur.
Bu yüzden ayarları değiştirip istediğin kadar yeniden çalıştırabilirsin.

Çalıştırma:
    python -m crawler.extract
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import trafilatura

from crawler.fetcher import URLS_PATH, cache_path

OUTPUT_PATH = Path("data/pages.jsonl")
TITLE_PREFIX = "Koç Holding | "
MIN_CHARS = 200  # bundan kısa metinler (ör. sadece JS menüsü olan sayfalar) atlanır

# Bazı sayfalarda trafilatura ana metni bulamayıp site menüsünü de alıyor.
# Menü "EN" satırıyla başlıyor ve bu satırla bitiyor; ardından "EN" ve "Ana Sayfa" geliyor.
MENU_END = "\nKoç Topluluğu 100 Yaşında\nEN\n"
# Sayfa sonuna eklenen arayüz artıkları (tam satır olarak eşleşir)
NOISE_LINES = {
    "Ana Sayfa", "Select...", "Bul", "İlgili sayfalar",
    "Please ensure Javascript is enabled for purposes of", "website accessibility",
}


def clean_title(title: str | None) -> str:
    """Tüm sayfalarda tekrar eden 'Koç Holding | ' önekini atar."""
    title = (title or "").strip()
    return title.removeprefix(TITLE_PREFIX).strip()


def pick_date(extracted_date: str | None, url: str) -> str | None:
    """Sayfanın yayın tarihini seçer.

    Haber URL'lerinde yıl var (/haberler/2013/...). trafilatura'nın tarihi o yılla
    tutuyorsa güveniyoruz. Sabit sayfalarda trafilatura sitenin son güncelleme
    tarihini (ör. 2026-09-24) buluyor; yanlış tarih yerine None döndürüyoruz.
    """
    match = re.search(r"/((?:19|20)\d{2})/", url)
    if match is None:
        return None  # URL'de yıl yok → sabit sayfa
    url_year = match.group(1)
    if extracted_date and extracted_date.startswith(url_year):
        return extracted_date  # ör. "2013-12-16"
    return url_year  # tutarsızsa en azından yıl doğru olsun


def strip_site_menu(text: str) -> str:
    """Metne karışmış site menüsünü ve arayüz artıklarını atar.

    Genel "çok tekrar eden satırı sil" yöntemi kullanmıyoruz: yönetim kurulu
    listesi gibi gerçek bilgiler de birçok sayfada tekrar ediyor.
    """
    if text.startswith("EN\n") and MENU_END in text:
        text = text.split(MENU_END, 1)[1]  # menü bitişinden sonrasını al
    lines = [line for line in text.split("\n") if line.strip() not in NOISE_LINES]
    return "\n".join(lines).strip()


def content_hash(text: str) -> str:
    """Metnin parmak izi: aynı metin → aynı hash. Boşluk farkları önemsiz olsun diye normalize ediyoruz."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def extract_page(html: str, url: str) -> dict | None:
    """Tek sayfadan başlık, tarih ve ana metni çıkarır. Metin yoksa None döner."""
    doc = trafilatura.bare_extraction(
        html,
        url=url,
        with_metadata=True,  # başlık, tarih vb.
        include_tables=True,  # tablolardaki bilgiyi (ör. rakamlar) kaybetmeyelim
        include_comments=False,
    )
    if doc is None or not doc.text:
        return None
    text = strip_site_menu(doc.text)
    if not text:
        return None
    return {
        "url": url,
        "baslik": clean_title(doc.title),
        "tarih": pick_date(doc.date, url),
        "metin": text,
    }


def main() -> None:
    entries = json.loads(URLS_PATH.read_text(encoding="utf-8"))
    stats = {"onbellekte_yok": 0, "metin_yok": 0, "cok_kisa": 0, "tekrar_icerik": 0, "yazildi": 0}
    seen_hashes: dict[str, str] = {}  # hash → bu metni ilk taşıyan URL

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for entry in entries:
            url = entry["url"]
            path = cache_path(url)
            if not path.exists():
                stats["onbellekte_yok"] += 1
                continue

            page = extract_page(path.read_text(encoding="utf-8"), url)
            if page is None:
                stats["metin_yok"] += 1
                continue
            if len(page["metin"]) < MIN_CHARS:
                stats["cok_kisa"] += 1
                continue

            # İkinci tekilleştirme katmanı: URL farklı ama metin aynı mı?
            digest = content_hash(page["metin"])
            if digest in seen_hashes:
                stats["tekrar_icerik"] += 1
                continue
            seen_hashes[digest] = url

            page |= {
                "hash": digest,
                "lastmod": entry["lastmod"],
                "karakter": len(page["metin"]),
                "cikarilma_zamani": datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(page, ensure_ascii=False) + "\n")
            stats["yazildi"] += 1

    print(f"Sonuç: {stats}")
    print(f"{OUTPUT_PATH} dosyasına {stats['yazildi']} sayfa yazıldı.")


if __name__ == "__main__":
    main()
