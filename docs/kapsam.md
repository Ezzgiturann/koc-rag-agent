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
- Sitemap'te `koc.com.tr`, site ise `www.koc.com.tr`'ye yönlendiriyor → normalizasyonda `www` kaldırılır.
- Taranan web metni güvenilmez veridir; prompt'ta talimat değil veri olarak işaretlenir.
