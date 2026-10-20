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
