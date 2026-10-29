"""Soru-cevap logunun özeti: kullanım, süre, arama davranışı, "yok" cevapları, 👎 alan sorular.

LLM çağrısı yok; yalnızca data/log.db okunur. API açıkken de çalışır (SQLite aynı anda
okumaya izin verir; kilit yalnızca Milvus Lite dosyasında).

Çalıştırma:
    python -m app.log_rapor
"""

import json
import statistics

from app.db import DB_PATH, connect

# "Yok" diyen cevapların (tam veya kısmi) tipik kalıpları; LIKE büyük/küçük harfe yalnızca ASCII'de duyarsız
NOT_FOUND_PATTERNS = ["%kaynaklarda yok%", "%yer alm%", "%bulunmamaktadır%"]


def query_texts(searches_json: str) -> list[str]:
    """Log'daki aramalar sütunundan sorgu metinlerini çıkarır; modelin bozuk JSON'u olduğu gibi kalır."""
    texts = []
    for args in json.loads(searches_json):
        try:
            texts.append(json.loads(args).get("sorgu", args))
        except (json.JSONDecodeError, AttributeError):
            texts.append(args)
    return texts


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * (60 - len(title)))


def main() -> None:
    if not DB_PATH.exists():
        print(f"Log yok: {DB_PATH}. Önce arayüzden veya API'den soru sor.")
        return
    conn = connect()

    section("Genel")
    # Karşılaştırma 1/0 döner → SUM ile sayılır. Hiç satır yoksa SUM NULL döner; COALESCE 0'a çevirir.
    g = conn.execute("""
        SELECT COUNT(*)                                   AS toplam,
               COALESCE(SUM(hata IS NOT NULL), 0)         AS hatali,
               COALESCE(SUM(geri_bildirim = 1), 0)        AS iyi,
               COALESCE(SUM(geri_bildirim = -1), 0)       AS kotu,
               COALESCE(SUM(maliyet), 0)                  AS maliyet,
               MIN(zaman) AS ilk, MAX(zaman) AS son
        FROM sorgular
    """).fetchone()
    print(f"{g['toplam']} soru ({g['ilk'][:16]} → {g['son'][:16]}) | hata {g['hatali']} | "
          f"👍 {g['iyi']}  👎 {g['kotu']} | tahmini maliyet ${g['maliyet']:.4f}")
    for row in conn.execute("SELECT hata, COUNT(*) AS n FROM sorgular WHERE hata IS NOT NULL "
                            "GROUP BY substr(hata, 1, instr(hata, ':')) ORDER BY n DESC"):
        print(f"   {row['n']} × {row['hata'][:90]}")  # GROUP BY hata türü ("APITimeoutError:" kısmı)

    section("Süre (başarılı sorular)")
    durations = [r["sure"] for r in conn.execute("SELECT sure FROM sorgular WHERE hata IS NULL")]
    if durations:
        # Medyan: tek bir 60 sn'lik takılma ortalamayı bozar, ortadaki değeri bozmaz
        print(f"ortalama {statistics.mean(durations):.1f} sn | medyan {statistics.median(durations):.1f} sn")
        for r in conn.execute("SELECT sure, soru FROM sorgular WHERE hata IS NULL ORDER BY sure DESC LIMIT 3"):
            print(f"   {r['sure']:5.1f} sn  {r['soru'][:70]}")

    section("Arama davranışı")
    # json_array_length: JSON listesinin eleman sayısı → soru başına arama
    a = conn.execute("""
        SELECT AVG(json_array_length(aramalar))         AS ortalama,
               SUM(json_array_length(aramalar) = 0)     AS aramasiz,
               MAX(json_array_length(aramalar))         AS en_cok
        FROM sorgular WHERE hata IS NULL
    """).fetchone()
    # json_each: listeyi satırlara açar (her arama bir satır). Elemanlar modelin gönderdiği JSON metni;
    # bozuk olabilir → json_valid ile koru, yoksa json_extract tüm sorguyu hataya düşürür.
    y = conn.execute("""
        SELECT COUNT(*) AS toplam,
               SUM(CASE WHEN json_valid(j.value)
                         AND (json_extract(j.value, '$.yil_baslangic') IS NOT NULL
                              OR json_extract(j.value, '$.yil_bitis') IS NOT NULL)
                        THEN 1 ELSE 0 END) AS yilli
        FROM sorgular, json_each(sorgular.aramalar) AS j
    """).fetchone()
    if a["ortalama"] is not None:
        print(f"soru başına {a['ortalama']:.1f} arama (en çok {a['en_cok']}) | aramasız cevap {a['aramasiz']}")
    if y["toplam"]:
        print(f"yıl filtreli arama: {y['yilli']}/{y['toplam']} (%{100 * y['yilli'] / y['toplam']:.0f})")

    section("'Yok' diyen cevaplar (gerçekten kapsam dışı mı, arama hatası mı?)")
    # f-string yalnızca "cevap LIKE ? OR ..." iskeletini kurar; kalıplar yine ? ile ayrı gönderilir
    where = " OR ".join("cevap LIKE ?" for _ in NOT_FOUND_PATTERNS)
    for r in conn.execute(f"SELECT id, soru, aramalar FROM sorgular WHERE {where} ORDER BY id", NOT_FOUND_PATTERNS):
        print(f"   #{r['id']} {r['soru'][:60]!r}  aramalar: {query_texts(r['aramalar'])}")

    section("👎 alan sorular (test setine aday)")
    rows = conn.execute("SELECT id, zaman, soru, cevap, aramalar FROM sorgular WHERE geri_bildirim = -1 ORDER BY id").fetchall()
    if not rows:
        print("   (yok)")
    for r in rows:
        print(f"   #{r['id']} {r['zaman'][:16]}  {r['soru']!r}")
        print(f"      cevap: {(r['cevap'] or '').replace(chr(10), ' ')[:150]}")
        print(f"      aramalar: {query_texts(r['aramalar'])}")
    conn.close()


if __name__ == "__main__":
    main()
