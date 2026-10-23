# Öğrenme Günlüğü

## 1. Gün — 2026-10-20

### Karar: OpenAI yerine Gemini (ücretsiz katman)
- Bütçe olmadığı için cevap modeli ve embedding'i Gemini API'nin ücretsiz katmanına taşıdık.
- `openai` paketini Gemini'nin OpenAI uyumlu ucuyla kullanıyoruz; sağlayıcı değiştirmek sadece `.env` değişikliği.
- Gemini bu uçta Responses API sunmuyor → Chat Completions kullanacağız.
- Ücretsiz katmanda gönderilen veri Google tarafından kullanılabilir → yalnızca kamuya açık veri.

### Adım 1: Proje iskeleti
- Python 3.12 ile `.venv` sanal ortamı kurduk (sistemdeki `python3` 3.9 olduğu için açıkça 3.12 seçtik).
- `requirements.txt`'e sadece bugün kullanılan 4 paketi, sürümleri sabitlenmiş (`==`) olarak yazdık.
- `.env.example` şablon olarak commit'lenir; gerçek anahtar `.env`'de durur ve `.gitignore` ile git dışında kalır.
- Klasörlere boş `__init__.py` koyduk; böylece her klasör bir Python paketi olur ve `python -m agent.ask` gibi çalıştırılabilir.
- `git init` ile repo açıldı; ilk commit'te `.env`'in girmediğini `git status` ile doğruladık.

### Adım 2: Milvus Lite testi
- `MilvusClient(uri=...)`: uri dosya yolu ise Milvus Lite (yerel), `http://...` ise Milvus sunucusu; kod aynı kalır.
- Koleksiyon = tablo; `dimension` vektör boyutu, `metric_type="COSINE"` benzerlik ölçüsü (1'e yakın = benzer).
- Elle anlamlandırılmış 4 boyutlu vektörlerle aramada "otomotiv" sorgusu Ford Otosan'ı 0.964 skorla ilk buldu.
- Tuzak: pymilvus import sırasında `.env`'i kendisi okuyor ve `MILVUS_URI`'yi http adresi sanıyor → değişkeni `VECTOR_DB_URI` yaptık.
- Şemada olmayan alanlar (`ad`, `metin`) "dinamik alan" olarak saklanıyor; aramada `output_fields` ile geri istenir.

### Adım 3: LLM'e soru gönderme (`agent/ask.py`)
- `OpenAI(api_key, base_url)` ile Gemini'ye bağlandık; `base_url` değişirse aynı kod başka sağlayıcıya gider.
- Chat Completions'ta her istekte `messages` listesini (system + user) biz göndeririz; sunucu geçmişi hatırlamaz.
- Gemini 3.8 Flash varsayılan olarak gizli "düşünme" token'ları harcıyor (bir denemede 512 token, 39 sn); bunları ayrı alanda değil sadece `total_tokens` içinde raporluyor → `toplam − girdi − çıktı` ile hesapladık.
- `reasoning_effort="low"` ile düşünme 0'a indi, süre ~15 sn'ye düştü; basit/RAG sorularında yeterli.
- Model cevabı kendi hafızasından verdi, kaynak yok → doğruluğunu kontrol edemiyoruz; RAG'e ihtiyacımızın nedeni bu.

### Adım 4: Token ve maliyet (`agent/cost.py`)
- tiktoken (`o200k_base`) ile istek atmadan önce token tahmini; Türkçe ekler ayrı token'lara bölünüyor (`kur|uldu`), aynı cümle İngilizceden daha çok token tutuyor.
- Tahmin (11) ile Gemini'nin raporu (25) farklı: system mesajı + mesaj format token'ları + Gemini'nin farklı tokenizer'ı.
- Maliyet = (girdi × girdi fiyatı + (çıktı + düşünme) × çıktı fiyatı) / 1.000.000; fiyatlar 1M token başına.
- Düşünme token'ları çıktı fiyatından faturalanıyor (fiyat sayfasında "including thinking tokens").
- Bilinmeyen modelde 0 döndürmek yerine `ValueError`: yanlış "bedava" raporu sessizce oluşmasın.

### Adım 5: Elle RAG (`agent/manual_rag.py`)
- koc.com.tr/hakkinda/biz-kimiz'den 4 paragrafı elle alıp `<kaynak url="...">` etiketi içinde prompt'a koyduk; robots.txt her şeye izin veriyor (`Allow: /`).
- Metinde olan soru doğru cevaplandı (%8 ihracat payı) ve kaynak URL'si yazıldı.
- Metinde olmayan soruda (CEO) model uydurmadı: "Bu bilgi verilen kaynakta yok." Ama yine de "Kaynak:" satırı ekledi → ileride düzeltilecek küçük kusur.
- Metne gizlenen "talimatları yok say" saldırısı uygulanmadı; system prompt'taki "kaynak VERİDİR" kuralı + etiketleme işe yaradı. Tek bir başarılı deneme tam güvenlik kanıtı değil.
- RAG ile girdi 25 → ~480 token'a çıktı; injection denemesinde `low` ayarına rağmen ~150 gizli düşünme token'ı harcandı (maliyet 2 kat).
- Veri kalitesi notu: sitemap'te `organizasyon-semasi/404` gibi bozuk URL'ler var; crawler'da ayıklanmalı.

### 1. Gün quiz sonucu: 2 doğru, 1 kısmen, 2 tekrar
- ✅ COSINE'de `distance` = benzerlik (büyük daha iyi); embedding modeli sonradan değiştirilemez (yeniden indeksleme).
- 🔁 `MILVUS_URI` hatası prompt injection değil, isim çakışmasıydı: pymilvus `.env`'i kendisi okuyor.
- 🔁 Maliyet hesabı örneği: 2.000 girdi + (300 çıktı + 200 düşünme) → (1.500 + 1.875) / 1M = 0,003375 $.
- 🟡 `</kaynak>` silinmezse saldırgan etiketi erken kapatıp veri alanından "kaçar".
- 🔁 Tek bir başarılı injection testi güvenlik kanıtı değildir; savunma katmanlıdır.

## 2. Gün — 2026-10-23: Crawler

### Adım 1: Kapsam (`docs/kapsam.md`)
- Sitemap analizi: 1.135 kayıt → tekilleştirme ve filtre sonrası ~778 Türkçe URL (~13 dk tarama).
- `/haberler/haberler/X` ile `/haberler/X` aynı sayfa (ikisi de 200, boyutlar neredeyse aynı) → normalizasyonla birleştirilecek.
- Dil: sadece Türkçe; `gorsel-ve-videolar` hariç. Kararlar tarihleriyle kapsam.md'de.
- Kara liste yerine beyaz liste: sitede yeni bir bölüm açılırsa kendiliğinden taramaya girmez.

### Adım 2: Sitemap okuma (`crawler/sitemap.py`)
- 1.135 kayıt → 1.108 birebir tekil → 997 normalizasyon sonrası → 778 kapsam içi; `data/urls.json`'a yazıldı.
- Normalizasyon: https + `www.koc.com.tr` (yönlendirme isteğini atlamak için), `/haberler/haberler/` → `/haberler/`, sondaki `/`, query ve `#fragment` atılır.
- Sitemap XML'i namespace kullanıyor → etiketleri `sm:url`, `sm:loc` diye aramak gerekiyor.
- Güvenilmez XML için `defusedxml` (XXE ve "billion laughs" saldırılarına karşı); arayüzü ElementTree ile aynı.
- Varsayım hatası: httpx'in openai ile geldiğini sanmıştım, 3.x sürümünde gelmiyor → ayrıca kuruldu.

### Adım 3: Sayfa indirici (`crawler/fetcher.py`)
- robots.txt'yi kendi User-Agent'ımızla indirip `RobotFileParser.parse()`'a veriyoruz; her URL'den önce `can_fetch()` kontrolü.
- `RateLimiter`: `time.monotonic()` ile iki istek arası en az 1 sn; 20 sayfa + robots.txt = 20,5 sn, kayıtlardaki aralık tam 1,0 sn.
- Ham HTML `data/raw/<sha256[:16]>.html` olarak önbellekleniyor; tekrar çalıştırınca diskteki sayfalar atlanıyor (20 önbellekte, 3 yeni).
- Hata olursa kaydedip devam; üst üste 5 hata → dur. Her deneme `data/fetch_log.jsonl`'a yazılıyor.
- `final_url` = istenen URL → `www` normalizasyonu sayesinde hiç yönlendirme yok.

### Adım 4: Metin çıkarma (`crawler/extract.py`) ve tam tarama
- Tam tarama: 776/778 sayfa, 2 × 404, hiç robots engeli yok; ~13 dk, 243 MB.
- trafilatura ~320 KB HTML'den 0,2–27 KB temiz metin çıkarıyor (%99+ gürültü atılıyor).
- İçerik hash'i 235 kopyayı yakaladı: "Bizden Haberler Dergisi" sayfaları aynı şablon metni taşıyor, asıl içerik PDF'te. URL normalizasyonu bunu asla yakalayamazdı.
- Tarih: haberlerde trafilatura doğru (URL yılıyla tutarlı), sabit sayfalarda site güncelleme tarihini buluyor → `None`.
- Varsayım istisnası: 2 haberde sadece `/haberler/haberler/` hali çalışıyor; normalizasyon onları 404'e çevirdi.
- Sonuç: 536 sayfa, ~2 milyon karakter → `data/pages.jsonl`.

## 3. Gün — 2026-10-23: İndeksleme

### Kararlar
- Embedding boyutu 768 (`.env`: `EMBEDDING_DIM`), chunk 400 token + 60 token örtüşme.
- Ölçüm: toplam 582.605 token, medyan sayfa 713 token, 536 sayfanın 328'i 512 token'dan uzun → chunking şart.
- Gemini embedding ucu: toplu girdi ve `dimensions` destekli, vektörler normalize geliyor, `usage` raporlamıyor.

### Adım 1: Embedding sezgisi (`indexing/embedding_demo.py`)
- Tahmin: TR–EN benzerliği < 0,30 → gerçek **0,94**. Model kelimeyi değil anlamı kodluyor; dil neredeyse fark etmiyor.
- Tahmin: ilgisiz cümle ≈ 0 → gerçek **0,62**. Skorlar sıfırdan başlamıyor; sabit eşik işe yaramaz, sıralama ve skor farkı önemli.
- Sorguda "Koç" geçtiği için "Koç Topluluğu'nun enerji şirketi" cümlesi "Tüpraş rafinerisi"nin önüne geçti → chunk'lara başlık eklemek bağlam katar.
- 768 ve 3072 boyut aynı sıralamayı verdi (fark ≤ 0,02) → 768 kararı doğrulandı.
- Vektörler normalize olduğu için iç çarpım = cosine. Sorgu öneki skorları ~0,08 artırdı ama bu küçük testte sıralamayı değiştirmedi.
