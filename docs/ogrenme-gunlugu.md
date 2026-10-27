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
