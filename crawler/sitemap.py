"""Sitemap'i okur, URL'leri normalize eder, kapsama göre filtreler ve tekilleştirir.

Kapsam kuralları: docs/kapsam.md

Çalıştırma:
    python -m crawler.sitemap
"""

import json
from pathlib import Path
from urllib.parse import urlparse, urlunparse

# Sitemap dışarıdan gelen güvenilmez XML; defusedxml zararlı XML hilelerini
# (dış varlık / XXE, "billion laughs") reddeden, ElementTree ile aynı arayüzlü sürüm
import defusedxml.ElementTree as ET
import httpx

from crawler import USER_AGENT

SITEMAP_URL = "https://www.koc.com.tr/sitemap.xml"
CANONICAL_HOST = "www.koc.com.tr"  # koc.com.tr buraya yönlendiriyor; doğrudan buna gidip yönlendirmeyi atlıyoruz
OUTPUT_PATH = Path("data/urls.json")

# Sitemap XML'i bir "namespace" kullanıyor; etiketleri ararken bu önekle aramak gerekiyor
NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# Beyaz liste: URL'nin ilk klasörü bunlardan biri değilse kapsam dışı ("" = anasayfa)
ALLOWED_SECTIONS = {
    "", "hakkinda", "faaliyet-alanlari", "medya-merkezi", "yatirimci-iliskileri",
    "surdurulebilirlik", "kocta-hayat", "koc-toplulugu-100-yasinda", "iletisim",
    "kisisel-verilerin-korunmasi",
}
EXCLUDED_PREFIXES = ("/medya-merkezi/gorsel-ve-videolar",)


def fetch_sitemap() -> str:
    """Sitemap XML'ini indirir."""
    response = httpx.get(SITEMAP_URL, headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True)
    response.raise_for_status()  # 4xx/5xx gelirse sessizce devam etme, hata ver
    return response.text


def parse_sitemap(xml_text: str) -> list[dict]:
    """XML'den her <url> için {url, lastmod} sözlüğü çıkarır."""
    root = ET.fromstring(xml_text)
    entries = []
    for node in root.findall("sm:url", NS):
        loc = node.findtext("sm:loc", default="", namespaces=NS).strip()
        lastmod = node.findtext("sm:lastmod", default="", namespaces=NS).strip()
        if loc:
            entries.append({"url": loc, "lastmod": lastmod})
    return entries


def normalize_url(url: str) -> str:
    """Aynı sayfaya giden farklı yazımları tek bir biçime indirger."""
    parts = urlparse(url.strip())
    path = parts.path

    # Bilinen site hatası: /haberler/haberler/X ile /haberler/X aynı sayfa
    while "/haberler/haberler/" in path:
        path = path.replace("/haberler/haberler/", "/haberler/")

    path = path.rstrip("/") or "/"  # sondaki / farkı yaratmasın; anasayfa "/" kalsın
    # https'e ve kanonik host'a zorla; query (?a=b) ve fragment (#bolum) at
    return urlunparse(("https", CANONICAL_HOST, path, "", "", ""))


def is_in_scope(url: str) -> bool:
    """docs/kapsam.md kurallarına göre URL taranacak mı?"""
    path = urlparse(url).path
    section = path.strip("/").split("/")[0]
    if section not in ALLOWED_SECTIONS:
        return False
    if "/404" in path:  # sitemap'teki bozuk kayıtlar
        return False
    return not path.startswith(EXCLUDED_PREFIXES)


def build_url_list(entries: list[dict]) -> list[dict]:
    """Normalize eder, filtreler, tekilleştirir. Aynı URL birden fazlaysa en yeni lastmod kalır."""
    unique: dict[str, dict] = {}
    for entry in entries:
        url = normalize_url(entry["url"])
        if not is_in_scope(url):
            continue
        previous = unique.get(url)
        # ISO tarih metinleri ("2026-09-24T15:59...") alfabetik sıralanınca kronolojik sıralanır
        if previous is None or entry["lastmod"] > previous["lastmod"]:
            unique[url] = {"url": url, "lastmod": entry["lastmod"]}
    return sorted(unique.values(), key=lambda e: e["url"])


def main() -> None:
    entries = parse_sitemap(fetch_sitemap())
    raw_urls = [e["url"] for e in entries]
    normalized = {normalize_url(u) for u in raw_urls}
    urls = build_url_list(entries)

    print(f"Sitemap'teki kayıt          : {len(raw_urls)}")
    print(f"Birebir tekil URL           : {len(set(raw_urls))}")
    print(f"Normalizasyon sonrası tekil : {len(normalized)}")
    print(f"Kapsam filtresi sonrası     : {len(urls)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(urls, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{OUTPUT_PATH} dosyasına yazıldı. İlk 3 kayıt:")
    for e in urls[:3]:
        print(f"  {e['url']}  (lastmod {e['lastmod'][:10]})")


if __name__ == "__main__":
    main()
