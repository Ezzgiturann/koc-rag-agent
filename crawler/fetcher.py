"""data/urls.json'daki sayfaları kibarca indirir ve ham HTML'i data/raw/ altına kaydeder.

Kurallar (CLAUDE.md): robots.txt'ye uy, saniyede en fazla 1 istek, kendini tanıtan User-Agent.
Diskte olan sayfa tekrar indirilmez; script yarıda kesilirse kaldığı yerden devam eder.

Çalıştırma:
    python -m crawler.fetcher --limit 20     # deneme
    python -m crawler.fetcher                # hepsi
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from crawler import USER_AGENT

URLS_PATH = Path("data/urls.json")
RAW_DIR = Path("data/raw")
LOG_PATH = Path("data/fetch_log.jsonl")
ROBOTS_URL = "https://www.koc.com.tr/robots.txt"

MIN_INTERVAL = 1.0  # iki istek arası en az 1 saniye
MAX_CONSECUTIVE_ERRORS = 5  # üst üste bu kadar hata → dur (site bizi engelliyor olabilir)


class RateLimiter:
    """İki istek arasında en az `interval` saniye geçmesini sağlar."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.last = 0.0

    def wait(self) -> None:
        # monotonic: bilgisayarın saati değişse bile geriye gitmeyen sayaç
        elapsed = time.monotonic() - self.last
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last = time.monotonic()


def cache_path(url: str) -> Path:
    """URL'den sabit bir dosya adı üretir (URL'ler dosya adı için fazla uzun/özel karakterli)."""
    return RAW_DIR / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.html"


def load_robots(client: httpx.Client, limiter: RateLimiter) -> RobotFileParser:
    """robots.txt'yi kendi User-Agent'ımızla indirip ayrıştırır."""
    limiter.wait()
    response = client.get(ROBOTS_URL)
    robots = RobotFileParser()
    robots.parse(response.text.splitlines() if response.status_code == 200 else [])
    return robots


def fetch_page(client: httpx.Client, url: str) -> dict:
    """Tek sayfayı indirir; başarılıysa HTML'i diske yazar. Sonuç kaydını döndürür."""
    record = {"url": url, "file": None, "fetched_at": datetime.now(timezone.utc).isoformat()}
    try:
        response = client.get(url)
    except httpx.HTTPError as exc:  # timeout, bağlantı hatası vb.
        return record | {"status": "error", "error": type(exc).__name__}

    record |= {"status": response.status_code, "final_url": str(response.url)}
    content_type = response.headers.get("content-type", "")

    # Yönlendirme bizi başka bir siteye götürdüyse kaydetme
    if urlparse(str(response.url)).netloc != urlparse(url).netloc:
        return record | {"error": "baska_siteye_yonlendirme"}
    if response.status_code == 200 and "text/html" in content_type:
        path = cache_path(url)
        path.write_text(response.text, encoding="utf-8")
        record["file"] = str(path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="en fazla kaç YENİ sayfa indirilsin")
    args = parser.parse_args()

    urls = [e["url"] for e in json.loads(URLS_PATH.read_text(encoding="utf-8"))]
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    limiter = RateLimiter(MIN_INTERVAL)
    headers = {"User-Agent": USER_AGENT}
    stats = {"onbellekte": 0, "indirildi": 0, "robots_engeli": 0, "hata": 0}
    consecutive_errors = 0

    with httpx.Client(headers=headers, timeout=30, follow_redirects=True) as client, \
            LOG_PATH.open("a", encoding="utf-8") as log:
        robots = load_robots(client, limiter)

        for url in urls:
            if cache_path(url).exists():
                stats["onbellekte"] += 1
                continue
            if args.limit is not None and stats["indirildi"] + stats["hata"] >= args.limit:
                break
            if not robots.can_fetch(USER_AGENT, url):
                stats["robots_engeli"] += 1
                continue

            limiter.wait()
            record = fetch_page(client, url)
            log.write(json.dumps(record, ensure_ascii=False) + "\n")

            if record["file"]:
                stats["indirildi"] += 1
                consecutive_errors = 0
            else:
                stats["hata"] += 1
                consecutive_errors += 1
                print(f"  ! {record['status']} {url}")
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"Üst üste {MAX_CONSECUTIVE_ERRORS} hata, durduruluyor.")
                    break

            done = stats["indirildi"] + stats["hata"]
            if done % 10 == 0:
                print(f"  {done} sayfa denendi...")

    print(f"\nSonuç: {stats}")
    print(f"Diskteki toplam sayfa: {len(list(RAW_DIR.glob('*.html')))} / {len(urls)}")


if __name__ == "__main__":
    main()
