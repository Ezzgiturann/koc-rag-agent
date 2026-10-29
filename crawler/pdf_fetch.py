"""Seçilmiş 3 faaliyet raporu PDF'ini cdn.koc.com.tr'den indirir (data/pdf/).

Liste koddadır: rapor sayfasındaki (Yatırımcı İlişkileri → Faaliyet Raporları) başlık ve
yayın tarihleriyle. Tarih metadata'ya girer: yıl filtresi ve "en yeni kaynağı esas al"
kuralı PDF chunk'larında da çalışsın.

Kurallar fetcher.py ile aynı: robots.txt, saniyede en fazla 1 istek, tanıtıcı User-Agent.
Diskte olan PDF tekrar indirilmez.

Çalıştırma:
    python -m crawler.pdf_fetch
"""

from pathlib import Path
from urllib.robotparser import RobotFileParser

import httpx

from crawler import USER_AGENT
from crawler.fetcher import MIN_INTERVAL, RateLimiter

PDF_DIR = Path("data/pdf")
ROBOTS_URL = "https://cdn.koc.com.tr/robots.txt"
MAX_BYTES = 30 * 1024 * 1024  # 30 MB; en büyüğü ~8 MB. Beklenmedik dev dosyayı yarıda kes

REPORTS = [
    {"dosya": "2024-yillik.pdf", "baslik": "2024 Yıllık Faaliyet Raporu", "tarih": "2025-03-11",
     "url": "https://cdn.koc.com.tr/cmscontainer/kocholding/media/koc/03yatirimci-iliskileri/kocholding-fr24tr_.pdf?ext=.pdf"},
    {"dosya": "2025-yillik.pdf", "baslik": "2025 Yıllık Faaliyet Raporu", "tarih": "2026-02-25",
     "url": "https://cdn.koc.com.tr/cmscontainer/kocholding/media/koc/03yatirimci-iliskileri/kocholding-fr25tr_1.pdf?ext=.pdf"},
    {"dosya": "2026-2-ceyrek.pdf", "baslik": "2026 2. Çeyrek Ara Dönem Faaliyet Raporu", "tarih": "2026-08-05",
     "url": "https://cdn.koc.com.tr/cmscontainer/kocholding/media/koc/03yatirimci-iliskileri/finansal-bilgiler/faaliyet-raporlari-tr/koc-holding-ara-donem-faaliyet-raporu.pdf?ext=.pdf"},
]


def load_robots(client: httpx.Client, limiter: RateLimiter) -> RobotFileParser:
    """CDN'in robots.txt'si. Dosya yoksa (cdn.koc.com.tr 400 dönüyor) kural da yoktur:
    boş liste ayrıştırılır ve her şeye izin verilir — urllib.robotparser'ın 4xx davranışıyla aynı."""
    limiter.wait()
    response = client.get(ROBOTS_URL)
    robots = RobotFileParser()
    robots.parse(response.text.splitlines() if response.status_code == 200 else [])
    print(f"robots.txt: HTTP {response.status_code}" + ("" if response.status_code == 200 else " → kural yok, izinli"))
    return robots


def download_pdf(client: httpx.Client, url: str, path: Path) -> str:
    """PDF'i parça parça indirir ve doğrular; sonucu anlatan kısa bir metin döndürür."""
    # stream: dosyanın tamamı belleğe alınmadan önce başlıklara bakıp vazgeçebiliriz
    with client.stream("GET", url) as response:
        if response.url.host != "cdn.koc.com.tr":  # yönlendirme bizi başka bir siteye götürdüyse
            return f"HATA: başka siteye yönlendirme ({response.url.host})"
        if response.status_code != 200:
            return f"HATA: HTTP {response.status_code}"
        content_type = response.headers.get("content-type", "")
        if "application/pdf" not in content_type:
            return f"HATA: PDF değil ({content_type})"
        if int(response.headers.get("content-length", 0)) > MAX_BYTES:
            return "HATA: dosya çok büyük"

        data = bytearray()
        for chunk in response.iter_bytes():
            data += chunk
            if len(data) > MAX_BYTES:  # content-length yalan söylemiş veya hiç yoksa
                return "HATA: dosya çok büyük (indirme sırasında)"

    # Sihirli byte'lar: her PDF "%PDF" ile başlar. Başlık "pdf" dese de içerik hata sayfası olabilir
    if not data.startswith(b"%PDF"):
        return "HATA: içerik PDF değil"
    path.write_bytes(data)
    return f"indirildi ({len(data) / 1e6:.1f} MB)"


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(MIN_INTERVAL)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as client:
        robots = load_robots(client, limiter)
        for report in REPORTS:
            path = PDF_DIR / report["dosya"]
            if path.exists():
                print(f"  önbellekte: {path}")
                continue
            if not robots.can_fetch(USER_AGENT, report["url"]):
                print(f"  robots engeli: {report['url']}")
                continue
            limiter.wait()
            print(f"  {report['baslik']}: {download_pdf(client, report['url'], path)}")


if __name__ == "__main__":
    main()
