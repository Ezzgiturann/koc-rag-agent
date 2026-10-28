"""Soru-cevap logu: her soru SQLite'a bir satır olarak yazılır.

SQLite sunucusuz bir veritabanıdır: tüm tablo tek bir dosyada (data/log.db) durur,
Python'la birlikte gelir (sqlite3). Log hem hata ayıklama hem de eval kaynağı:
kullanıcının 👎 verdiği sorular test setine aday.

Çalıştırma (geçici bir dosyada yaz-oku denemesi; gerçek log'a dokunmaz):
    python -m app.db
"""

import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data/log.db")

# IF NOT EXISTS: tablo zaten varsa dokunma → her bağlantıda güvenle çalıştırılabilir
SCHEMA = """
CREATE TABLE IF NOT EXISTS sorgular (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    zaman         TEXT NOT NULL,     -- ISO 8601, saat dilimiyle: 2026-10-28T10:15:00+03:00
    soru          TEXT NOT NULL,
    cevap         TEXT,              -- hata olduysa boş
    kaynaklar     TEXT,              -- JSON liste: [{"no", "baslik", "url", "tarih", "atif"}]
    aramalar      TEXT,              -- JSON liste: modelin gönderdiği ara(...) parametreleri
    model         TEXT,
    girdi         INTEGER,
    cikti         INTEGER,
    dusunme       INTEGER,
    maliyet       REAL,              -- USD, ücretli katman fiyatıyla tahmin
    sure          REAL,              -- saniye
    hata          TEXT,              -- hata türü ve mesajı; başarılıysa boş
    geri_bildirim INTEGER CHECK (geri_bildirim IN (-1, 1))  -- 👎 = -1, 👍 = 1, verilmediyse boş
)
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    """Bağlantı açar ve tablonun var olduğundan emin olur.

    Her işlemde yeni bağlantı açıyoruz: SQLite'ta bu çok ucuz ve FastAPI istekleri
    farklı thread'lerde çalıştığı için bağlantıyı thread'ler arasında paylaşmamış oluruz.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row  # satırlara sütun adıyla erişim: row["soru"]
    conn.execute(SCHEMA)
    return conn


def log_query(
    question: str, answer: str | None, sources: list[dict], searches: list[str], model: str,
    usage: dict[str, int], cost: float, duration: float, error: str | None = None, path: Path = DB_PATH,
) -> int:
    """Bir soruyu loglar; satırın id'sini döndürür (geri bildirim bu id'ye bağlanır)."""
    row = (
        datetime.now().astimezone().isoformat(timespec="seconds"),
        question, answer,
        json.dumps(sources, ensure_ascii=False), json.dumps(searches, ensure_ascii=False),
        model, usage.get("girdi", 0), usage.get("cikti", 0), usage.get("dusunme", 0),
        cost, round(duration, 2), error,
    )
    conn = connect(path)
    try:
        with conn:  # blok hatasız biterse commit, hata olursa rollback
            # ? yer tutucuları: değerler SQL metnine yapıştırılmaz, ayrı gönderilir → SQL injection olmaz
            cursor = conn.execute(
                "INSERT INTO sorgular (zaman, soru, cevap, kaynaklar, aramalar, model, "
                "girdi, cikti, dusunme, maliyet, sure, hata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        return cursor.lastrowid
    finally:
        conn.close()  # `with conn` commit eder ama bağlantıyı KAPATMAZ


def set_feedback(query_id: int, score: int, path: Path = DB_PATH) -> bool:
    """Geri bildirimi kaydeder (1 veya -1). Böyle bir id yoksa False döner."""
    if score not in (-1, 1):
        raise ValueError("score 1 veya -1 olmalı")
    conn = connect(path)
    try:
        with conn:
            cursor = conn.execute("UPDATE sorgular SET geri_bildirim = ? WHERE id = ?", (score, query_id))
        return cursor.rowcount == 1  # güncellenen satır sayısı: 0 ise id yok
    finally:
        conn.close()


def recent(limit: int = 10, path: Path = DB_PATH) -> list[dict]:
    """Son `limit` kaydı yeniden eskiye döndürür; JSON sütunları listeye çevrilir."""
    conn = connect(path)
    try:
        rows = conn.execute("SELECT * FROM sorgular ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    finally:
        conn.close()
    return [{**dict(r), "kaynaklar": json.loads(r["kaynaklar"]), "aramalar": json.loads(r["aramalar"])}
            for r in rows]


def main() -> None:
    # Geçici dosya: deneme kayıtları gerçek log'u kirletmesin
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "deneme.db"
        ok_id = log_query(
            "Koç Holding'in CEO'su kim?", "Levent Çakıroğlu'dur [1].",
            [{"no": 1, "baslik": "Üst Yönetimde Görev Değişiklikleri", "url": "https://www.koc.com.tr/...",
              "tarih": "2026-02-13", "atif": True}],
            ['{"sorgu": "Koç Holding CEO"}'], "gemini-3.5-flash-lite",
            {"girdi": 4232, "cikti": 44, "dusunme": 0}, 0.00138, 4.81, path=path,
        )
        err_id = log_query("Merhaba", None, [], [], "gemini-3.5-flash-lite", {}, 0.0, 60.3,
                           error="APITimeoutError: Request timed out.", path=path)
        # Kötü niyetli görünen soru: parametreli sorgu sayesinde sadece METİN olarak saklanır
        evil_id = log_query("x'); DROP TABLE sorgular; --", "Bu bilgi kaynaklarda yok.", [], [],
                            "gemini-3.5-flash-lite", {}, 0.0, 1.0, path=path)

        print("geri bildirim (var olan id):", set_feedback(ok_id, 1, path=path))
        print("geri bildirim (olmayan id):", set_feedback(999, -1, path=path))

        for r in recent(path=path):
            print(f"\n#{r['id']} {r['zaman']}  soru={r['soru']!r}")
            print(f"   cevap={r['cevap']!r}  hata={r['hata']!r}  geri_bildirim={r['geri_bildirim']}")
            print(f"   kaynaklar={len(r['kaynaklar'])}  token={r['girdi']}/{r['cikti']}  süre={r['sure']} sn")
        print(f"\nKayıt sayısı: {len(recent(path=path))} (id'ler: {ok_id}, {err_id}, {evil_id})")


if __name__ == "__main__":
    main()
