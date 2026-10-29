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

### Adım 2: Chunking (`indexing/chunker.py`)
- En küçük birim cümle: paragraf → cümle → 400 token'a kadar paketle; yeni chunk önceki chunk'ın son cümleleriyle (≤60 token) başlar.
- 536 sayfa → 1.880 chunk (1 sn); medyan 369, ortalama 331 token; örtüşme toplam token'ı %6,8 artırdı (582.605 → 622.047).
- Medyan 400'ün altında: chunk bir sonraki cümle sığmayınca kapanıyor, son chunk'lar kısa kalıyor.
- Token sayıları toplanabilir değil: cümleler ayrı sayılıp birleştirilince 13 chunk 400'ü aştı (max 422). Sınır katı değil hedef; embedding limiti 8.192 olduğundan zararsız.
- Başlık chunk metnine değil embedding önekine (`title: ... | text: ...`) konacak; bilinen kusur: "A.Ş." gibi kısaltmalarda cümle bölünüyor.

### Adım 3: Embedding üretme (`indexing/embedder.py`)
- Paket sınırı denenerek bulundu: istek başına en fazla 100 metin (101 → 400 hatası).
- Önbellek anahtarı = hash(model | boyut | önekli metin): model/boyut/önek değişirse eski vektör kendiliğinden geçersiz.
- Ücretsiz katman limitleri hata mesajından okundu: **dakikada 100** ve **günde 1.000** embedding; paketteki her metin ayrı sayılıyor.
- Dakikalık limit için 2. günün `RateLimiter`'ı yeniden kullanıldı (paketler arası 61 sn); backoff yedek güvenlik. Hata: son denemeden sonra boşuna bekleniyordu → düzeltildi.
- Günlük kota doldu: 1.880 chunk'ın 800'ü bitti. JSON'da vektör başına ~17 KB (float32'de 3 KB olurdu).
- Ders: hata mesajının tamamını yazdır; limitin türünü (dakika/gün) ve bekleme süresini API kendisi söylüyor.

### Ara karar: embedding yerel modele taşındı (EmbeddingGemma)
- Gemini günlük kotası (1.000) 1.860 chunk'a yetmiyor → yerel Ollama'da iki aday 15 soruluk testle karşılaştırıldı.
- EmbeddingGemma: Hit@1 0,80 · Hit@5 0,93 · MRR 0,845 · 97 sn. Qwen3-Embedding 0.6B: 0,67 · 0,93 · 0,739 · 521 sn.
- Test setinde hata: tek bir yazım biçimi ("%8") kabul ediliyordu; "yüzde 8", "% 8" yazan doğru chunk'lar yanlış sayıldı → kanıt listesi alternatifleri kabul ediyor. Düşük skorda önce başarısız örneklere bak.
- Gerçek zayıflık: "yatırımcılarla ilişkilerden sorumlu koordinatör" sorusu unvanın birebir geçtiği chunk'ları bulamadı → 4. gün hibrit arama (BM25) için aday.
- Skor ölçeği modele özgü: ilgisiz cümle Gemini'de ~0,62, Gemma'da ~0,35 → sabit eşik modeller arası taşınamaz.
- Menü temizliği: 20 sayfada trafilatura site menüsünü almıştı; genel "çok tekrar eden satırı sil" kuralı CEO adını da silecekti → hedefli kural. Sonuç: 532 sayfa, 1.860 chunk.

### Adım 4: Milvus'a yazma (`indexing/build_index.py`)
- Şemayı kendimiz tanımladık: `chunk_id` (birincil anahtar, VARCHAR), `vector` (768), `metin`, `baslik`, `url`, `tarih` (nullable), `chunk_no`.
- VARCHAR uzunluğu bayt cinsinden; Türkçe karakterler UTF-8'de 2 bayt. En uzun chunk 2.198 bayt → sınır 8.192 (büyük chunk deneyleri için pay).
- İndeks: FLAT (tam arama). Milvus Lite FLAT, IVF_FLAT, HNSW, AUTOINDEX'in hepsini destekliyor; ~2.000 vektörde yaklaşık aramaya gerek yok.
- Tuzaklar: yeni süreçte koleksiyon "released" → aramadan önce `load_collection()`; arama sonucunda birincil anahtar şemadaki adıyla (`chunk_id`) gelir, `id` değil.
- Sağlama: 3 rastgele chunk kendi vektörüyle arandı → kendini 1,0000 skorla buldu. Veritabanı 7,2 MB (JSON önbelleği 30 MB; float32 farkı).
- Koleksiyon açıklamasına model ve boyut yazıldı: arama farklı modelle yapılırsa fark edilebilsin.

### Adım 5: İlk anlamsal arama (`retrieval/search.py`)
- Soru → EmbeddingGemma (sorgu önekiyle) → Milvus FLAT arama → ilk 5 chunk (skor, başlık, tarih, URL, metin).
- Model kontrolü: indeks hangi modelle kurulduysa sorgu da onunla yapılmalı. Şema `description` Milvus Lite'ta saklanmıyor → `properties` kullanıldı.
- CEO sorusu: cevap 1. sıradaki chunk'ta. İhracat sorusu: 1. sırada güncel "%8", 4. sırada 2016 tarihli "yüzde 9" → aynı bilgi yıllara göre değişiyor; tarih önemli.
- "2024 net kârı" sorusu: ilgili sayfalar geliyor ama hiçbirinde net kâr yok → "ilgili ama cevapsız" chunk'lar; model uydurmamalı.
- Skorlar dar bir bantta (0,69–0,74): cevabı olan ve olmayan soru skorla ayırt edilemiyor.
- gRPC bilgi mesajları: `GRPC_VERBOSITY=ERROR` pymilvus import edilmeden önce ayarlanmalı; `.env`'de işe yaramıyor.

## 4. Gün — 2026-10-27: Uçtan uca RAG ve hibrit arama

### Adım 1: Uçtan uca RAG (`agent/rag.py`)
- Retrieve → augment → generate: `search()` ilk 5 chunk'ı getirir, her biri `<kaynak id tarih url>` etiketine sarılır, model numaralı atıfla (`[1]`) cevaplar. Soru başına ~2.550 girdi token'ı (tahmin ~2.000; etiketler ve Türkçe tokenizasyon payı).
- Atıflar elle doğrulandı: CEO cevabının atıf verdiği 4 chunk'ta da "Çakıroğlu" geçiyor; adı geçmeyen 2009 chunk'ına atıf yok. İhracatta "%8'in üzerinde" (tarihsiz sayfa) ve "yaklaşık yüzde 9" (2016) doğru ayrıldı.
- 2024 net kâr: model "kaynaklarda yok" dedi; 2024 sonuç haberlerinde "kâr" kelimesi hiç geçmiyor → doğru ret. 2004'ün "Net Karı %71" chunk'ı tuzağına düşmedi.
- Kusurlar: (a) 60 token örtüşme yüzünden aynı cümle iki chunk'ta → [4][5] aynı bilgi, boşa token; (b) "tarihsiz" sayfayı model güncel saydı — doğru ama şans; (c) başlıklarda `<br />` artığı.
- Altyapı: Gemini 3.8/3.7 Flash aşırı yoğundu (503, zaman aşımı); `openai` istemcisinin varsayılan 600 sn zaman aşımı programı dondurdu → `timeout=60, max_retries=3`. Geçici olarak `.env`'de `gemini-3.5-flash-lite`'a geçildi; kod değişmedi.

### Adım 2: Retrieval değerlendirme seti (`eval/retrieval_eval.py`)
- 25 soru, 3 tür: `genel` (3. günün 15 sorusu), `terim` (6; nadir terim birebir soruda), `parafraz` (4; metnin kelimeleri kullanılmadan). Arama fonksiyonu parametre (`SearchFn`) → hibrit arama aynı kodla ölçülecek.
- Ölçüm hatası bulundu: özgeçmiş sayfalarında kişinin adı metinde değil yalnızca başlıkta → kanıt artık başlık + metinde aranıyor.
- Vektör baseline: Hit@1 0,64 · Hit@5 0,92 · MRR 0,738. Tür bazında Hit@1: genel 0,80 · parafraz 0,50 · **terim 0,33** — beklenenin tersine en zayıf tür terim.
- Tahminler: "en zayıf parafraz" ✗ (terim); "koordinatör sorusu ilk 5'e girer" ✗ (başlığa bakınca bile 7.); "Fuel Oil'in adı geçen hâli daha iyi" ✗ — ama karşılaştırma adil değildi: iki sorunun kanıtı farklıydı (rakam vs. proje adı).
- Kanıt listesi yine eksikti: "2,7 milyar" yanında "toplam 3 milyar dolar" da geçerli cevap → eklendi. Ders tekrarı: başarısız örneklere bakmadan skora güvenme.

### Adım 3: BM25 (`retrieval/bm25.py`)
- Kendi kodumuz: ters indeks (kelime → {chunk: tekrar sayısı}), IDF (Lucene biçimi, negatif olmaz), k1=1,5 TF doygunluğu, b=0,75 uzunluk normalizasyonu. Başlık + metin indekslendi.
- Türkçe tuzaklar: `"İ".lower()` görünmez nokta bırakıyor → önce İ→i, I→ı. `\w+` ile bölme kesme işaretini ayırıcı yapıyor (Suna'nın → suna + nın).
- Hit@5: vektör 0,92 · BM25 0,76 · BM25+F5 0,80. Ama `terim` türünde BM25 1,00 (vektör 0,83); `parafraz`ta 0,50–0,75 (vektör 1,00). Vektörün ilk 10'da bulamadığı "Ekonomik Araştırmalar Koordinatörü"nü BM25 1. sırada buldu.
- Tamamlayıcılık: vektör ve F5'in ilk 5'lerinin birleşimi 25/25 soruyu kapsıyor → hibrit aramanın (RRF) üst sınırı. Zorluk: 10 adaydan doğru 5'i seçmek.
- F5 artı/eksi: "yatırımcılarla ilişkilerden" → `yatır`/`ilişk` doğru sayfayı buldu (vektörde 7., F5'te 2.); ama "yatırımcı" ile "yatırım" birleşti, idf 6,6 → 1,2 (over-stemming). Soru kelimeleri ("kimdir" idf 8,2) en yüksek puanı alıyor → stopword adayı.

### Adım 4: Hibrit arama — RRF (`retrieval/hybrid.py`)
- RRF: her listede 1 / (k + sıra) puan, chunk_id üzerinden toplanır; skor ölçeği farkı (kosinüs 0–1, BM25 0–30) sorun olmaz. Liste sayısından bağımsız ~6 satır.
- İlk deneme (k=60, DEPTH=20) vektörden kötü çıktı: Hit@5 0,88 < 0,92. Neden: iki listenin 19./20. sırasındaki chunk (1/79 + 1/80 = 0,025) tek listede 1. olanı (1/61 = 0,016) geçiyor — zayıf "ortaklık" güçlü tekil sıradan çok puan alıyor.
- Tarama (k ∈ 1–60, DEPTH ∈ 5/10/20): belirleyici DEPTH, k değil. DEPTH=5'te Hit@5 her k'da 0,96 → tek şanslı nokta değil, ayara duyarsız bölge seçildi; k=60 kaldı.
- Sonuç (vektör → hibrit): Hit@5 0,92 → 0,96; terim 0,83 → 1,00; parafraz 1,00 korundu. Ama Hit@1 0,64 → 0,60: doğru chunk ilk 5'e giriyor, 1. sıraya değil.
- Plan gerekçem ("7. sıradaki de girsin" → derin liste) yanlıştı; ölçmeden doğru sanılan bir varsayımdı. 25 soruda 0,92 → 0,96 tek soru farkı: sinyal, kanıt değil.

### Adım 5: RAG'i hibrit aramaya bağlama (`agent/rag.py`)
- Değişiklik yalnızca retrieve satırı: `search` → `hybrid_search`; prompt ve LLM çağrısı aynı (ortak chunk alanları sayesinde `format_sources()` değişmedi). BM25 indeksi program başında bir kez kurulur.
- Ekonomik Araştırmalar Koordinatörü: "Ahmet Çimenoğlu [4]" — kaynağı yalnızca BM25 buldu (vektör/BM25 sırası -/2). Vektör tabanlı RAG bu soruyu cevaplayamazdı.
- İhracat cevabı %8 → "yüzde 7'nin üzerinde": BM25 2025 tarihli bir haber getirdi; model "en yeni tarihli kaynak" kuralını doğru uyguladı. Sabahki %8 kaynağa sadıktı ama güncel değildi.
- Test setinde eskimiş beklenti: dış satım sorusunun kanıtı yalnızca %8 biçimleri → zamana bağlı olgularda beklenen cevabın da tarihi olmalı.
- Gürültü: Ekonomik Araştırmalar sorusunda 5 kaynaktan 1'i işe yarıyor (BM25'ten ilgisiz özgeçmiş, vektörden "atamalar" haberleri) → reranker için gerekçe.

### Adım 6: Eskimiş test beklentisini düzeltme
- Korpustaki tüm "ihracat payı" ifadeleri tarihleriyle listelendi: 1999 %7 → 2005 %9 → 2013–14 %10 → 2016–20 %9 → **2021–2025 ~%7**; tarihsiz Hakkında sayfaları %8 (güncellenmemiş).
- Eski kanıt (%8) eskimiş sabit sayfayı ödüllendiriyordu → kanıt 2021–2025 "yüzde 7" biçimleriyle değişti; soruya `not` alanıyla gerekçe ve kontrol tarihi eklendi.
- Yeni Hit@5: vektör 0,92 → 0,88; hibrit 0,96 → 0,92. Önceki skorların bir kısmı ölçüm hatasıydı; hibrit yine en iyi.
- "Dış satım" ↔ "ihracat": soru metnin kelimesini kullanmayınca BM25 katkı veremiyor; aynı bilgi "ihracat" diye sorulunca bulunuyor.

### Adım 7: Başlık temizliği ve yeniden indeksleme
- Envanter: 44 sayfada `<br />`, 1 sayfada `<div>`, 2 sayfada başlık yalnızca "Koç Holding |" (sonda boşluk olmadığı için `removeprefix("Koç Holding | ")` eşleşmemişti).
- `clean_title(title, url)`: etiketleri sil, boşlukları normalize et, öneki boşluksuz eşleştir; boş kalırsa URL'nin son parçasından başlık üret.
- Zincir yeniden çalıştı: extract → chunker → embedder → build_index. Embedding önbelleği sayesinde 1.860 chunk'tan yalnızca başlığı değişen 197'si yeniden hesaplandı (11 sn).
- Etki: vektör Hit@1 0,60 → 0,64 (başlık embedding önekinde); diğer metrikler aynı. Temizlik asıl olarak prompt'taki kaynak etiketlerini ve okunurluğu düzeltti.

### Adım 8: Ardışık chunk birleştirme ve tarihsiz kaynak kuralı (`agent/rag.py`)
- `merge_adjacent()`: aynı sayfanın ardışık chunk'ları (url + chunk_no) tek kaynakta birleşir; `join_overlapping()` a'nın b'nin başı olan en uzun son ekini bulup örtüşmeyi bir kez yazar (çiftlerin %81'inde örtüşme birebir; kalanında örtüşme yok → alt alta).
- 25 sorunun 15'inde ilk 5'te aynı sayfanın ardışık parçaları vardı (Microsoft sorusunda 5 kaynağın 4'ü tek haber). Token kazancı küçük (%4,5); asıl kazanç modelin aynı cümleyi iki bağımsız kanıt saymaması.
- Bulgu: "5 kaynak" çoğu zaman 5 farklı sayfa değil → kaynak çeşitliliği düşük. Boşalan yerleri doldurmak reranker adımına bırakıldı.
- Prompt kural 4'e ek: tarihsiz kaynak tarihliyle çelişirse tarihli esas alınır; yalnız tarihsiz varsa güncelliğin belirsiz olduğu söylenir. İhracat cevabı artık 2025 (%7) / 2016 (%9) / tarihsiz (%8) ayrımını ve gerekçesini açıkça yazıyor.

### Adım 9: Reranker (`retrieval/rerank.py`) ve Adım 6'nın düzeltilmesi
- Cross-encoder `BAAI/bge-reranker-v2-m3` (sentence-transformers, Mac'te MPS): soru ve chunk'ı birlikte okur. Hibrit aramadan aday havuzu → reranker → ilk 5. Model ~2 GB, ilk indirme ~10 dk.
- Aday derinliği ölçüldü: derinlik 10'dan itibaren doğru chunk 25/25 soruda havuzda → 15 (≈24 aday) seçildi. Aynı parametrenin iki amacı var: RRF sıralaması için 5, reranker havuzu (recall) için 15.
- Sonuç: Hit@1 0,60 → **0,84**, Hit@5 0,96 → **1,00**, MRR 0,746 → 0,910; terim Hit@1 0,33 → 1,00. Bedel: soru başına 0,03 → 3,5 sn.
- **Adım 6 yanlıştı:** ifadeleri listeleyen script aynı metni tekilleştirdiği için 2026-06-05'teki "toplam ihracatının yüzde 8'ini" (2010'la aynı ifade) görünmedi. Güncel bilgi %8; tarihsiz sayfa eskimiş değil, uyumlu. Kanıt 2026 + tarihsiz %8 ifadeleriyle düzeltildi. Ders: veri incelerken tekilleştirme, aranan şeyi gizleyebilir.
- Reranker skorları üstte doyuyor (CEO sorusunda 5 kaynağın hepsi 1,00) → skor eşiğiyle "ilgili/ilgisiz" ayrımı yapılamaz; reranker sıralamayı düzeltir, "cevap var mı" kararını değil.

## 5. Gün — 2026-10-27: Agent

### Adım 1: Tool calling'i tanımak (`agent/tool_demo.py`)
- Araç JSON Schema ile tanımlanır (ad, açıklama, parametreler); model kodu değil yalnızca bu metinleri görür → `description` prompt'un parçası. Aracı model değil biz çalıştırırız: model `tool_calls` (id + JSON METNİ parametreler) döndürür, sonucu `role: "tool"` + `tool_call_id` ile geri veririz.
- "Dış satım" sorusunda model sorguyu kendisi zenginleştirdi: `"Koç Topluluğu ihracat payı türkiye dış satım"` → BM25'in 4. gün bulamadığı "ihracat" kelimesi artık sorguda. "Merhaba"da hiç araç çağırmadı (`finish_reason=stop`).
- Dokümanda yazmayan bulgu: Gemini 3 her araç çağrısına `extra_content.google.thought_signature` ekliyor; silip geri gönderince **400** ("Function call is missing a thought_signature"). Modelin mesajını `model_dump(exclude_none=True)` ile olduğu gibi geri göndermek bu yüzden şart; mesajı elle kurmak kodu bozardı.
- 3.8 Flash hâlâ 503 (yoğun); 3.5 Flash-Lite ile devam. Küçük sadakat sorunu: model kaynakta olmayan "yaklaşık" kelimesini ekledi.

### Adım 2: Arama aracı (`agent/tools.py`)
- `ara(sorgu)` = 4. günün akışı (hibrit → reranker → ardışık chunk birleştirme). Şema ve çalıştırma kodu aynı dosyada; açıklama modele aracın NASIL kullanılacağını da söylüyor (parça başına ayrı arama, yetersizse eşanlamlıyla tekrar).
- Kaynak numaraları tüm aramalar boyunca tek sayaçtan: 2. arama [6]'dan devam etti, daha önce görülen kaynaklar eski numarasını ([3], [5]) korudu.
- Hatalar istisna değil metin olarak modele döner (bozuk JSON, olmayan araç, boş sorgu) → döngü çökmez, model düzeltebilir. Sorgu 200 karakterle sınırlı.
- `format_source()` `rag.py`'den ayrıldı; etiketleme/güvenlik kuralları tek yerde.
- Bulgu: sorgudaki yıl filtre değil sıradan kelime ("ihracat payı 2016" → 1999, 2002 haberleri) → ileride `yil` metadata filtresi adayı; önce karşılaştırma sorusunda etkisini ölç.

### Adım 3: Agent döngüsü (`agent/loop.py`)
- Düz bir `for`: model araç isterse çalıştır + `role: "tool"` ekle, metin dönerse bitir. MAX_STEPS=4; dolarsa `tool_choice="none"` ile araçsız son cevap (dokümanda yoktu, test edildi: çalışıyor). Prompt = rag.py'nin 6 kuralı + araç kuralları (önce ara, parça başına ara, bir kez tekrar dene, selamlaşmada arama).
- "Dış satım": sorguya "ihracat payı" eklendi, tek aramada cevaplandı. "Merhaba": aramasız, 604 token. Atıf kontrolü (`check_citations`) ve "atıf yapılan" işareti eklendi.
- Karşılaştırma sorusu (2016 → bugün): 2 arama, ama cevap yanıltıcı (9 → 8): elindeki 2021/2023 kaynaklarındaki ~%7'yi kullanmadı, tarihsiz %8'i tarihli %7'ye tercih etti (kural 4 ihlali), 2026 haberi hiç bulunmadı (yıl filtre değil).
- Bağlam her adımda büyüyor: girdi 623 → 2.845 → 5.168; toplam 8.636 token ≈ RAG'in 3,5 katı (her çağrı tüm geçmişi yeniden gönderir).
- Ders: bulmak ≠ kullanmak (sentez hatası, lite modelin sınırı); agent aracın zayıflığını aşamaz → `yil` filtresi artık gözleme dayalı gereksinim.

### Adım 3b: Yıl filtresi (`ara(sorgu, yil_baslangic, yil_bitis)`)
- Filtre her katmanın İÇİNDE: Milvus `tarih >= "2016" and tarih < "2017"` (metin karşılaştırması yıl sırasıyla aynı, yeniden indeksleme yok), BM25'te aynı koşul. Tarihsiz chunk'lar filtre varken gelmez. Sonradan elemek havuzu boşaltabilirdi.
- Araç doğrulaması: "2016" metni int'e çevrilir; 1990–2030 dışı ve "geçen yıl" gibi değerler HATA metni döner. Aralıkta sonuç yoksa model "aralığı genişlet" mesajı alır.
- Model bugünün tarihini bilmez → prompt'a her çağrıda `date.today()` eklendi (kural 10).
- Karşılaştırma sorusu: 9 → 8 (yanıltıcı) yerine 9 → 6,7 → ~7 → >7; tarihsiz %8 ayrı tutuldu. Ama model "her dönem için ayrı ara" kuralına uymadı (tek 2016–2026 araması) → 2026 %8 haberi yine ilk 5'te yok; `yil_baslangic=2026` ile 1. sıradaydı. Araç yetenekli, model kullanmadı.
- Benim hatam: atıf kontrolü yalnız `[6]` biçimini sayıyordu; model `[6, 9]` yazınca kaçırdı → regex düzeltildi.

### Adım 4: RAG ile agent karşılaştırması (`eval/agent_karsilastirma.py`)
- 8 soru × 2 sistem; doğruluk = beklenen regex'lerin HEPSİ var, yasakların hiçbiri yok. Injection: `patch.object` ile iki sistemin arama fonksiyonu geçici olarak ilk kaynağa saldırı metni ekleyen sürümle değiştirildi.
- Sonuç berabere: 7/8 – 7/8. Agent girdi token'ı 1,4 kat ama maliyet eşit ($0,0099 – $0,0096): RAG zor sorularda düşünme token'ı harcadı (431, 1.037; çıktı fiyatından), agent hiç harcamadı.
- Karşılaştırma sorusunda ikisi de ✗ ama farklı: RAG "kaynaklarda yok" (vazgeçti), agent 9 → 7 seyrini doğru verdi, yalnız 2026 %8 eksik. İkili metrik kısmi doğruluğu gizliyor.
- Agent bu çalıştırmada hiç çoklu arama yapmadı (önceki çalıştırmada aynı soruya 2); çok parçalı soruyu tek birleşik sorguyla çözdü. Davranış deterministik değil → tek çalıştırmayla hüküm verilmez.
- Ders: retrieval güçlüyse agent'ın avantajı küçülür; kalan fark zor sorular (yıl filtresi: cevapsız soruda kendiliğinden 2024–2025) ve aramasız mesajlar (selamlaşma 769'a karşı 2.560 token). İki sistem de injection'a direndi.

### Adım 5: Konuşma geçmişi (takip soruları)
- Sıkıştırılmış geçmiş: yalnızca soru + cevap metni (araç sonuçları değil), son 3 çift. Eski cevaplardaki [n] silinir (yeni soruda numaralar sıfırdan başlar); kural 11: geçmiş yalnızca bağlam, bilgi yeniden aranır. `--sohbet` ile terminalden etkileşimli mod.
- "Peki 2016'da bu oran neydi?" → `ara("Koç Topluluğu Türkiye ihracatı yüzde", 2016–2016)`: konu geçmişten, dönem sorudan; ilk sonuç yetersizken bir kez yeniden aradı. Doğru: ~%9.
- "Bu bilgiyi kim açıklamış?" → iki tur geriye çözüp `"ihracat oranı yüzde 9 2016"` aradı; atıfsız geçmişe rağmen doğru haber: Ömer M. Koç, TİM töreni. Sabit RAG bu soruyu olduğu gibi arardı (ne konu ne dönem var).
- Zayıflık sürüyor: "bugün/güncel" denince yıl filtresini kendiliğinden kullanmıyor → 2026 %8'e ulaşmadı.

### Adım 6: Serbest sorulardan gelen düzeltmeler (kural 3, 2 yeni test sorusu, gRPC logu)
- Kullanıcının serbest soruları test setinin göremediği hataları buldu: "sahibi kim" → model "kurucu" diye aradı, cevap ("çoğunluk hissesi Koç Ailesi, %26,9 halka açık" — "Neden Koç Holding?") hiç gelmedi; "kaç firması var" → toplam yok ama "10 halka açık şirket" kaynaktaydı, katı kural 3 ("yalnızca 'yok' yaz") söyletmedi.
- Kural 3 gevşetildi: önce "tam cevap yok" de, sonra DOĞRUDAN ilgili kısmi bilgiyi atıfla ekle; soruyu cevaplamayan konu bilgisini ekleme. Test önce yazıldı (eşanlamlı + kısmi cevap; 10 soru). Sonuç: kısmi cevap agent ✓ (RAG ✗: tek aramada sayfa ilk 5'te yok, ama 1999/2002 şirket sayılarını kısmi bilgi olarak verdi); cevapsız soru ikisinde de ✓ kaldı (gerileme yok).
- Eşanlamlı soru ikisinde de ✗: kural değil arama sorunu ("sahip" ≠ "hisse/halka açık"). Model ilk makul sonuçta durdu; ciro sorusunda da aynı: 2024 (65,8 milyar $) dedi, korpustaki 2025 (64,3 milyar $, 2026-02-11) haberini bulmadı. Agent hatalarının ortak deseni: erken durma.
- Aynı soru + aynı prompt iki çalıştırmada farklı cevap verdi (bir "yok", bir "10 halka açık") → tek çalıştırmayla prompt etkisi ölçülemez. Uzun açık süreç eski kodu çalıştırır: değişiklik sonrası yeniden başlat.
- `GRPC_VERBOSITY=NONE`: GOAWAY E (hata) seviyesinde geldiği için ERROR ayarı gizlemiyordu. Toplam: RAG 7/10, agent 8/10; agent ~2 kat token/maliyet ($0,0187 – $0,0094).

## 6. Gün: Servis, arayüz ve log

Mimari: Streamlit (ui) → HTTP → FastAPI (api) → run_agent + SearchTool; her istek SQLite'a. Streamlit her etkileşimde betiği baştan çalıştırdığı ve Milvus Lite tek süreç kabul ettiği için ağır kaynakları API süreci bir kez yükler. API durumsuz: sohbet geçmişini arayüz tutar.

### Adım 1: SQLite logu (`app/db.py`)
- Tek tablo `sorgular`: soru, cevap, kaynaklar/aramalar (JSON metni), model, token, maliyet, süre, hata, geri bildirim. Hatalı istekler de loglanır (cevap boş, hata dolu) → "hangi saatte kaç 503" sorulabilir.
- Parametreli sorgu (`VALUES (?, ...)`): değer SQL metnine yapıştırılmaz → `x'); DROP TABLE sorgular; --` yalnızca metin olarak saklandı, tablo yerinde. Prompt injection'daki "veri talimat değildir" ilkesinin SQL karşılığı.
- `with conn:` transaction'dır (hatasız → commit, hata → rollback) ama bağlantıyı kapatmaz → `finally: conn.close()`. Her işlemde yeni bağlantı: FastAPI istekleri farklı thread'lerde çalışır, sqlite3 bağlantısı varsayılan olarak thread'ler arası paylaşılamaz.
- `CHECK (geri_bildirim IN (-1, 1))`: kural veritabanı düzeyinde; `rowcount == 0` → olmayan id (API'de 404 olacak). `CREATE TABLE IF NOT EXISTS` sütun eklemez; şema değişirse migration (`ALTER TABLE`) gerekir.

### Adım 2: FastAPI servisi (`app/api.py`)
- `GET /saglik`, `POST /sor`, `POST /geri-bildirim`. Pydantic doğrulaması (soru 1–1.000 karakter, boşluk kırpılır; geçmiş ≤ 10 tur; puan `Literal[-1, 1]`) kuraldışı isteği kodumuz çalışmadan 422 ile reddeder; olmayan id → 404.
- `lifespan`: SearchTool + Gemini istemcisi açılışta bir kez (~8 sn). Endpoint `def` (async değil): run_agent bloklayan bir fonksiyon, async def tüm sunucuyu dondururdu; `def` ayrı thread'de çalışır.
- `threading.Lock`: SearchTool'un kaynak numaraları paylaşılan durum; `tool.sources` kilit bırakılmadan kopyalanır (race condition). Log ve maliyet kilidin dışında. Hatalar log'a tam, kullanıcıya kısa: 429 kota, 503 yoğun/zaman aşımı, 500 diğer.
- Uçtan uca test (TestClient): CEO ✓ (ilk istek 11,8 sn: reranker'ın ilk çalıştırması ısınma), geri bildirim log'a yazıldı. Takip sorusu "Peki ondan önceki kimdi?" → "yok", oysa Turgay Durak 46 chunk'ta geçiyor: model "Çakıroğlu öncesi" diye aradı, `yil_bitis` ile eski CEO'yu aramayı düşünmedi (akıl yürütme sınırı). Aynı sayfanın bitişik olmayan iki chunk'ı ayrı kaynak olarak listeleniyor → arayüzde URL'ye göre gruplanabilir.

### Adım 3: Streamlit arayüzü (`app/ui.py`)
- Streamlit her etkileşimde betiği baştan çalıştırır; sohbet ve oylar `st.session_state`'te. 👍/👎 `on_click` callback'i betik yeniden çalışmadan önce çağrılır → butonlar seçilen işaretle değişir, ikinci oy yok. Geçmişe yalnızca başarılı cevaplar girer (hata mesajı modele "önceki cevap" olarak gitmesin).
- Güvenilmez metin ekranda da kod değil: kaynak başlıklarında markdown/Streamlit karakterleri kaçışlanır (`[x](link)`, `:red[]`, `$`), cevaptaki linkler/görseller düz metne çevrilir, link yalnızca `https://www.koc.com.tr/...` ise tıklanabilir. Aynı URL'nin chunk'ları tek satırda (`[1][3]`), atıf yapılanlar üstte.
- `uvicorn --reload` tüm `.py` dosyalarını izler: `app/ui.py`'yi düzenlemek API'yi de yeniden başlattı (~8 sn model yükleme) → o arada sağlık kontrolü "API kapalı" dedi.
- Uçtan uca: CEO ✓ (👍), "peki ondan önceki kimdi" üçüncü kez "yok" (👎; Turgay Durak korpusta) — model her seferinde "Çakıroğlu'ndan önce" diye arıyor. Tutarlı hata → test setine aday; log'daki 👎 tam bunun için.

### Adım 4: Log raporu (`app/log_rapor.py`) ve Streamlit ayarı
- Güvenlik açığı (benim hatam): Streamlit varsayılan olarak tüm ağ arayüzlerini dinliyor (`Network URL: 192.168.x.x:8501`) → aynı Wi-Fi'deki biri arayüz üzerinden localhost'a kapalı API'yi ve Gemini kotasını kullanabilirdi. `.streamlit/config.toml`: `address = "localhost"`, telemetri kapalı (`gatherUsageStats = false`).
- SQL özet: `SUM(kosul)` ile sayma (karşılaştırma 1/0), `COALESCE` (boş tabloda SUM NULL), `GROUP BY` hata türü, `json_array_length` (soru başına arama), `json_each` (listeyi satırlara açma) + `json_valid` koruması (modelin bozuk JSON'u sorguyu düşürmesin). Medyan Python'da: tek bir 60 sn'lik takılma ortalamayı bozar, medyanı bozmaz.
- İlk rapor (5 soru): ortalama 18,3 sn / medyan 15,7 sn; soru başına 1,4 arama; yıl filtreli arama 0/7 → "önceki CEO" sorusunda filtre gerekirken hiç kullanılmadı (5. gündeki gözlem artık ölçülebilir).
- "Yok" bölümü kapsam dışı ile arama hatasını ayırmaya yarıyor: "asistan" (gerçekten anlamsız soru) ile "ondan önceki" (korpusta var, arama hatası) aynı listede; 👎 ikincisini işaretledi.

## 7. Gün: PDF faaliyet raporları, final eval, Milvus Standalone

### Adım 1: PDF'siz "önce" ölçümü (sonuçlar: `data/eval_7gun/*_pdf_oncesi.*`)
- Kullanıcı soruları test setlerine girdi: retrieval 28 soru (yeni tür `kullanici`: sahibi kim, Çakıroğlu'ndan önceki CEO, en son yıllık gelir); agent karşılaştırması 12 soru (`önceki dönem`, `en güncel`). "2024 net kârı" PDF metni görülmeden eklenmedi: kanıt ifadesini tahminle yazmak testi yanlış bir şeyi ölçtürür.
- Retrieval: 3 kullanıcı sorusu 5 yöntemin HİÇBİRİNDE ilk 10'da değil (0/3). Eski 25 soru değişmedi (reranker Hit@1 21/25). Benim yazdığım sorular korpusun kelimeleriyle yazılmıştı; kullanıcınınkiler değil.
- Agent 10/12, RAG 7/12. "Önceki CEO" bu kez ✓, ama ikinci sorgu `"Koç Holding önceki CEO Turgay Durak 2015"`: adı kaynaktan değil modelin KENDİ bilgisinden sorguya koydu, sonra kaynakla doğruladı. Cevap kaynaklı, ama başarı akıl yürütmeden değil ezberden geldi; modelin bilmediği bir konuda işlemezdi.
- "En güncel gelir" ikisinde de ✗ ve daha kötü: 2023 (54,5 milyar $) dediler; doğrusu 2025 (64,3 milyar $). Agent `yil_baslangic: 2021` ile aradı — filtre kullandı ama yeterince dar değil. Sahiplik sorusunda agent bu kez ortaklık yapısını buldu (rastgelelik).

### Adım 2: PDF indirme (`crawler/pdf_fetch.py`)
- PDF seçimini düzelttim: dosya adlarına bakarak 2023 yıllık + iki ara dönem önermiştim; rapor sayfasının kendi listesi (HTML'e gömülü JSON) 2024 ve 2025 yıllık raporlarının da olduğunu gösterdi (`kocholding-fr24tr_.pdf` — adında "faaliyet-raporu" geçmiyor). Seçim: 2024 yıllık, 2025 yıllık, 2026 2. çeyrek.
- Tarih = yayın tarihi (2024 raporu 2025-03-11), kapsadığı yıl değil; HTML haberlerindeki gibi.
- `cdn.koc.com.tr/robots.txt` 400 → boş kural listesi = her şeye izin (robotparser'ın 4xx davranışı; önce test edildi). RateLimiter ve User-Agent fetcher.py'den.
- Dört doğrulama: son adres hâlâ cdn mi, content-type PDF mi, boyut ≤ 30 MB (stream ile indirirken de), ilk byte'lar `%PDF` mi (sihirli byte: türü içerikten doğrulamak). Dosya ancak hepsi geçince diske yazılır: bozuk veri önbelleğe girerse kalıcılaşır.

### Adım 3: Docling ile PDF metni (`crawler/pdf_extract.py`)
- OCR kapalı (dijital PDF, metin içinde), TableFormer açık (tablo → markdown, satır/sütun korunur). Sayfa sayfa `export_to_markdown(page_no=N)`; üst/alt bilgi ("furniture") varsayılan olarak dışarıda → tekrar eden rapor adı/sayfa no gürültüsü yok. URL'ye `#page=N`: atıf sayfa düzeyinde, şema değişmeden.
- İlk ölçüm yanılttı: 16 sayfa 225 sn (14 sn/sayfa → 720 sayfa 3 saat) — çoğu model indirme/yükleme. Isınmadan sonra 1,6 sn/sayfa (MPS; CPU 2,0). Gerçek: 681 sayfa, 449 tablo, ~18 dk. İlk çalıştırma sonrakileri temsil etmez (reranker'daki ders).
- Yarıda kalmaya dayanıklı: PDF başına ayrı çıktı, önce `.tmp`'ye yaz sonra `rename` (atomik) → yarım dosya "işlenmiş" sayılmaz.
- Docling kurulumu `huggingface_hub`'ı 2.3.0 → 1.33.0'a düşürdü (docling-ibm-models `<2` istiyor); bağımlılıklar kontrol edildi, reranker kurulumdan sonra test edildi.

### Adım 4: Tablo farkındalıklı chunk + indeks (4.173 chunk)
- Tabloda birim SATIR; tablo bölünürse yeni chunk başlık satırıyla başlar (sütun = yıl bilgisi kaybolmasın). Cümleler boşlukla, satırlar alt satırla birleşir. Hücre boşlukları sıkıştırılır (Docling hizalama boşluğu: az token, çok bayt; Milvus sınırı 8.192 bayt → aşılırsa açık hata).
- Benim yanılgım: "HTML'de tablo yok" sonucunu `|---` aramasıyla çıkarmıştım; trafilatura ayraç satırı koymuyor. "Eski chunk'lar birebir aynı mı?" testi yakaladı: 528 sayfa aynı, 4 tablolu sayfa (temettü, sermaye artırımı...) değişti — iyileşme: eskiden tablo tek satır yığınıydı.
- 1.860 → 4.173 chunk (2,2 kat; PDF 2.314, %39'u tablolu). Embedding önbelleği sayesinde yalnızca 2.320 chunk hesaplandı (159 sn, yerel Ollama).
- Korpus değişince test değişti: "2024 net kârı" artık cevaplanabilir (2024 raporu s. 15: "Konsolide net dönem kârı ise 1,3 milyar TL") → `pdf` türüne taşındı; yeni "cevapsız" sorusu "ortalama maaş" (korpusta 0). Hızlı kontrol: reranker 1. sırada s. 15.

### Adım 5: PDF sonrası ölçüm ve iki düzeltme (sonuçlar: `data/eval_7gun/`)
- Retrieval (hibrit + reranker): eski 25 soruda Hit@1 21 → 20, Hit@5 25 → 24 (CEO parafrazı 6. sıraya: yıllık raporların yönetim sayfaları rekabet ediyor); `pdf` sorusu 1. sırada. Eklemesiz BM25 Hit@1 0,43 → 0,34: muhasebe tablolarında "Koç Holding/kâr/2024" her yerde → IDF zayıflıyor; kaybın çoğunu reranker telafi ediyor. Kullanıcı soruları hâlâ 0/3 (tek soruluk arama).
- Hata: agent "önceki CEO"da 4 aramada Turgay Durak'ı buldu ama BOŞ cevap döndü. Ham yanıt: `tool_choice="none"`'a rağmen `finish_reason: tool_calls`. 5. günde tek denemeyle "çalışıyor" demiştim. Düzeltme: son çağrıda araç tanımı HİÇ gönderilmiyor (rica → imkânsızlık) + boş gelirse açık mesaj (`NO_ANSWER`). Hiç çalışmamış bir kod yolunu yeni zor soru çalıştırdı.
- Test hatası: "en güncel" cevabı "2025: 2.757 milyar TL" doğruydu, test yalnızca USD (64,3) kabul ediyordu → ikisi de kabul (2024/2023 rakamları hâlâ yanlış). Korpus büyüyünce doğru cevabın biçimleri de çoğalıyor.
- Düzeltme sonrası: agent 11/13, RAG 10/13 (önce 10/12 – 7/12). "En güncel"de agent adım sınırına ulaştı ve bu kez cevap yazdı (64,3 milyar USD) → düzeltme gerçek çalıştırmada doğrulandı. Sahiplik ✗ ama yarı doğru ("Koç Ailesi ve Vehbi Koç Vakfı pay sahipliğinde"; %26,9 yok). PDF eklenince RAG "kısmi cevap" ve "en güncel"i de çözdü: daha iyi veri, sabit akışın açığını da kapatıyor.

### Adım 6: Milvus Standalone (`docker-compose.yml`)
- 3 servis: milvus (hesaplama), etcd (şema/meta), MinIO (veri dosyaları, S3 uyumlu) — depolama ile hesaplamanın ayrılması. Resmi v3.0.3 dosyası okundu ve sertleştirildi: portlar `127.0.0.1:` önekli (Docker'da `"A:B"` = 0.0.0.0, tüm ağ); MinIO portları kaldırıldı (varsayılan şifre minioadmin dışarı açıktı; Milvus iç ağdan ulaşıyor); veri `data/milvus-standalone/` (git dışında). Üçüncü kez aynı açık türü: Streamlit, Docker, MinIO — "varsayılan her yere açık".
- Geçiş yalnızca `.env`: `VECTOR_DB_URI=http://127.0.0.1:19530`; kod değişmedi (1. günden beri "dosya yolu → Lite, http → sunucu").
- Yeni kavram — nihai tutarlılık: insert sonrası sayım 4.173 yerine 3.000 dedi. Varsayılan "Bounded" seviyede sorgu son birkaç saniyenin yazmalarını görmeyebilir (hız karşılığı). Lite tek süreçli, orada görünmüyordu. `build_index` sayım/sağlama sorguları `consistency_level="Strong"`.
- Lite ile birebir aynı sonuç (29 soru × 5 yöntem, diff boş): ikisi de FLAT = kesin arama. Kilit testi: API açıkken eval çalıştı — Lite'taki "aynı anda tek süreç" sınırı kalktı. (`agent_lock` hâlâ gerekli: o Milvus için değil, SearchTool'un kaynak numaraları için.)

### Adım 7: Cevap süresi — ölç, sonra hızlandır
- Basit soru ~11 sn: Gemini ~3,3 sn × en az 2 çağrı (~%60; agent'ın "önce karar ver" bedeli), reranker ~3,7 sn/arama (~%35), embedding 0,2 sn, BM25 + Milvus ~0. Ölçmeden "embedding'i hızlandır" demek toplamın %2'sine emek harcamak olurdu.
- Reranker seçenekleri 29 soruda ölçüldü: fp16 (16 bit sayılar) → 3,76 → 2,90 sn, sıralama birebir aynı (0 soru değişti). Aday derinliği 15 → 10 → 2,47 sn ama Hit@1 0,72 → 0,66: 4. günün "10 yeterli" ölçümü PDF'lerden önceydi; korpus büyüyünce doğru chunk havuza daha geç giriyor. Yalnızca fp16 uygulandı (GPU'da; CPU'da değil).
- Kalan büyük kalem Gemini; gerçek süreyi kodla kısaltamıyoruz ama algılanan süreyi streaming (adım adım ilerleme + kelime kelime cevap) kısaltabilir — sonraki aday iş.
