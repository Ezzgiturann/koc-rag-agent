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
