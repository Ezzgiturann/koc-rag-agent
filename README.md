# koc-rag-agent

Koç Holding'in **kamuya açık** verileri (koc.com.tr sayfaları ve faaliyet raporu
PDF'leri) üzerinden Türkçe soruları **kaynak göstererek** cevaplayan bir RAG agent.

> ⚠️ Bu bir **öğrenme projesidir**; Koç Holding ile bir bağlantısı yoktur ve resmî
> bir ürün değildir. Cevaplar hatalı olabilir; doğru bilgi için kaynak bağlantılarına bakın.

Projenin amacı bir RAG sistemini parça parça, **framework kullanmadan** (LangChain,
LlamaIndex vb. yok) kurarak her adımı anlamaktı. Gün gün alınan kararlar, ölçümler
ve hatalar [`docs/ogrenme-gunlugu.md`](docs/ogrenme-gunlugu.md) dosyasında.

## Nasıl çalışıyor?

```
koc.com.tr ──► crawler ──► chunking ──► embedding ──► Milvus
 (sitemap,      (trafilatura,  (tablo        (Ollama,
  robots.txt,    Docling PDF)   farkındalıklı) embeddinggemma 768 boyut)
  1 istek/sn)
                                                       │
Soru ──► Agent döngüsü (Gemini, tool calling) ──► ara() aracı
                                                       │
                     hibrit arama: vektör + BM25 ─► RRF ─► reranker (bge-reranker-v2-m3)
                                                       │
         Cevap + numaralı kaynaklar ◄──────────────────┘
```

| Klasör | İçerik |
|---|---|
| `crawler/` | Sitemap okuma, robots.txt'ye uyan indirici, HTML/PDF'ten metin |
| `indexing/` | Chunking, embedding, Milvus'a yazma |
| `retrieval/` | Vektör arama, BM25, RRF ile hibrit arama, cross-encoder reranker |
| `agent/` | Arama aracı ve agent döngüsü (ön arama, streaming, yedekli istek) |
| `eval/` | Retrieval ve agent test setleri, karşılaştırma betikleri |
| `app/` | FastAPI servisi, Streamlit arayüzü, SQLite log ve rapor |
| `docs/` | Kapsam ve öğrenme günlüğü |

**Teknolojiler:** Python 3.12+, `openai` paketi (Gemini'nin OpenAI uyumlu ucu),
Ollama (yerel embedding), Milvus / Milvus Lite (`pymilvus`), httpx, trafilatura,
Docling, sentence-transformers, FastAPI, Streamlit, SQLite.

## Kurulum

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Yerel embedding modeli (https://ollama.com)
ollama pull embeddinggemma:300m

# Ortam değişkenleri: şablonu kopyala, GEMINI_API_KEY'i doldur
cp .env.example .env
```

Gemini API anahtarı: https://aistudio.google.com → *Get API key*. Ücretsiz katmanda
gönderilen içerik Google tarafından kullanılabilir; yalnızca kamuya açık veri gönderin.

## İndeksi oluşturma

Taranan veri ve indeks depoda **yok** (`data/` git dışında); kendiniz üretirsiniz.
Crawler saniyede en fazla 1 istek attığı için tam tarama zaman alır.

```bash
python -m crawler.sitemap          # URL listesi (tekilleştirilmiş)
python -m crawler.fetcher          # sayfaları indir (--limit 20 ile deneyin)
python -m crawler.extract          # HTML → metin
python -m crawler.pdf_fetch        # faaliyet raporu PDF'leri
python -m crawler.pdf_extract      # PDF → metin ve tablo (Docling)
python -m indexing.embedder        # chunk + embedding
python -m indexing.build_index     # Milvus'a yaz
```

## Çalıştırma

```bash
uvicorn app.api:app          # 1. terminal: API (http://127.0.0.1:8000)
streamlit run app/ui.py      # 2. terminal: arayüz (http://localhost:8501)
```

Değerlendirme: `python -m eval.retrieval_eval` ve `python -m eval.agent_karsilastirma`.

Sağlayıcı yalnızca `.env` ile değişir: `LLM_BASE_URL` + `CHAT_MODEL` ile OpenAI'ye
veya Ollama'daki yerel bir modele geçilebilir; `VECTOR_DB_URI=http://127.0.0.1:19530`
ile Milvus Lite yerine `docker compose up -d` ile başlatılan Milvus Standalone kullanılır.

## Lisans

Kod [MIT](LICENSE) lisanslıdır. Taranan içerik (koc.com.tr metinleri ve PDF'ler)
bu depoda yer almaz ve sahiplerine aittir.
