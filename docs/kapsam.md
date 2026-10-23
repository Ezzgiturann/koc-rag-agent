# Kapsam

Agent'ın hangi verilere dayanarak cevap verdiği ve neyin bilerek dışarıda
bırakıldığı. Eval'de "bu soru kapsam dışı mı?" sorusunun cevabı burada.

## Kaynak
- Site: https://koc.com.tr (sitemap: https://www.koc.com.tr/sitemap.xml)
- robots.txt: `User-agent: * / Allow: /` — her şeye izin var, crawl-delay yok.
  Yine de saniyede en fazla 1 istek atıyoruz.
- User-Agent: `koc-rag-agent/0.1 (ogrenme projesi)`

## Dil
**Sadece Türkçe.** Sorular Türkçe olacak; İngilizce sayfalar büyük ölçüde
Türkçelerin çevirisi ve aramada tekrar yaratır. (Karar: 2026-10-23)

## Dahil edilen bölümler (beyaz liste)
URL'nin ilk klasörü bu listede değilse taranmaz. Beyaz liste kullanıyoruz,
çünkü sitede yeni bir İngilizce bölüm açılırsa kendiliğinden içeri sızmaz.

| Bölüm | Yaklaşık URL | İçerik |
|---|---|---|
| (anasayfa) | 1 | |
| `hakkinda` | 66 | Biz kimiz, yönetim kurulu, organizasyon şeması |
| `faaliyet-alanlari` | 11 | Sektörler ve şirketler |
| `medya-merkezi` | 652 | Haberler, bizden haberler, basın bültenleri, kılavuzlar |
| `yatirimci-iliskileri` | 16 | |
| `surdurulebilirlik` | 16 | |
| `kocta-hayat` | 13 | İnsan kaynakları |
| `koc-toplulugu-100-yasinda` | 1 | |
| `iletisim` | 1 | |
| `kisisel-verilerin-korunmasi` | 1 | KVKK metni |
| **Toplam** | **~778** | 2026-10-23 sitemap'ine göre, tekilleştirme sonrası |

## Hariç tutulanlar
| Ne | Neden |
|---|---|
| İngilizce bölümler (`about-us`, `media-center`, `en`, ...) | Dil kararı |
| `medya-merkezi/gorsel-ve-videolar` | Görsel/video galerisi, metin yok (Karar: 2026-10-23) |
| `arama`, `ihbar-bildirim`, `onemli-uyari`, `Newsletter` | Arama sayfası, form ve yasal uyarı; bilgi içeriği yok |
| URL'sinde `/404` geçenler | Sitemap'teki bozuk kayıtlar (ör. `hakkinda/organizasyon-semasi/404`) |
| PDF'ler (faaliyet raporları) | Ayrı bir gün Docling ile eklenecek |
| KAP bildirimleri | Şimdilik kapsam dışı |

## Bilinen veri sorunları
- `/medya-merkezi/haberler/haberler/...` ve `/medya-merkezi/haberler/...` aynı
  sayfayı veriyor (113 tekrar) → URL normalizasyonunda `/haberler/haberler/` → `/haberler/`.
- Sitemap'te `koc.com.tr`, site ise `www.koc.com.tr`'ye yönlendiriyor → normalizasyonda host
  `www.koc.com.tr` yapılır; böylece her sayfa için fazladan bir yönlendirme isteği atılmaz.
- İstisna: 2 haberde (`2015/izin-silinmez`, `2016/bazi-borclar-vardir-odeyemezsin`)
  sadece `/haberler/haberler/` hali çalışıyor, temiz hali 404 → şu an bu 2 sayfa eksik.
- "Bizden Haberler Dergisi" sayfalarının (~275) HTML'i sadece ortak tanıtım metni içeriyor;
  asıl içerik `cdn.koc.com.tr` üzerindeki PDF'lerde. İçerik hash'i kopyaları tek sayfaya indiriyor.
  PDF günü için aday arşiv.
- Sabit sayfalarda trafilatura'nın bulduğu tarih (2026-09-24) sitenin güncelleme tarihi;
  bu sayfalarda `tarih` boş bırakılır. Haberlerde tarih URL'deki yılla doğrulanır.
- Taranan web metni güvenilmez veridir; prompt'ta talimat değil veri olarak işaretlenir.

## Tarama sonucu (2026-10-23)
- 778 URL → 776 indirildi (2 × 404), 243 MB ham HTML
- Metin çıkarma: 235 içerik tekrarı + 5 çok kısa (<200 karakter) atıldı → **536 sayfa**
- 363 haber (1999–2026; 361'i tam tarihli), 133 tarihsiz sabit sayfa
- Toplam ~2 milyon karakter; sayfa başına medyan 2.449 karakter
