"""HTTP servisi: agent'ı bir API arkasına koyar, her soruyu SQLite'a loglar.

Ağır kaynaklar (Milvus, BM25, reranker, Gemini istemcisi) sunucu açılırken BİR KEZ yüklenir.
API durumsuzdur: sohbet geçmişini istemci (Streamlit) tutar ve her istekte gönderir.

Çalıştırma (yalnızca bu bilgisayardan erişilir; tek worker: Milvus Lite tek süreç kabul eder):
    uvicorn app.api:app --reload
Deneme sayfası: http://127.0.0.1:8000/docs
"""

import json
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from agent.loop import Event, add_turn, check_citations, run_agent  # ilk import: GRPC ayarı pymilvus'tan önce

import openai
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StringConstraints

from agent.ask import get_client
from agent.cost import estimate_cost
from agent.tools import SearchTool
from app.db import log_query, set_feedback

# Baştaki/sondaki boşluk silinir; "   " boş sayılır. Üst sınır ücretsiz kotayı korur.
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class Turn(BaseModel):
    soru: str = Field(max_length=1000)
    cevap: str = Field(max_length=5000)


class AskRequest(BaseModel):
    soru: Question
    gecmis: list[Turn] = Field(default=[], max_length=10)  # add_turn zaten son 3 turu tutar


class Source(BaseModel):
    no: int
    baslik: str
    url: str
    tarih: str | None
    atif: bool  # cevapta [no] olarak geçiyor mu


class AskResponse(BaseModel):
    id: int  # log satırı; geri bildirim bu id'ye verilir
    cevap: str
    kaynaklar: list[Source]
    aramalar: list[str]
    token: dict[str, int]
    maliyet: float
    sure: float


class FeedbackRequest(BaseModel):
    id: int
    puan: Literal[-1, 1]  # 👎 = -1, 👍 = 1; başka değer 422


resources: dict = {}  # lifespan doldurur: "llm", "tool"
# SearchTool kaynak numaralarını kendi içinde tutar (tool.sources); iki istek aynı anda
# çalışırsa numaralar karışır. Kilit istekleri sıraya sokar.
agent_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """yield'den önce: sunucu açılırken; sonra: kapanırken çalışır."""
    resources["llm"] = get_client()  # .env'i de yükler
    resources["tool"] = SearchTool()  # ~10 sn: reranker modeli belleğe
    yield
    resources["tool"].close()


app = FastAPI(title="Koç RAG Agent", lifespan=lifespan)


@app.get("/saglik")
def health() -> dict:
    return {"durum": "hazir" if "tool" in resources else "yukleniyor", "model": os.environ.get("CHAT_MODEL")}


def answer_question(req: AskRequest, on_event: Event | None = None) -> AskResponse:
    """Soruyu agent'la cevaplar ve loglar. on_event: aramalar ve cevap metni oluştukça bildirilir.
    Hatalar kullanıcıya gösterilecek kısa mesajlı HTTPException'a çevrilir."""
    history: list[dict] = []
    for turn in req.gecmis:
        history = add_turn(history, turn.soru, turn.cevap)  # atıflar silinir, son 3 tur kalır

    model = os.environ["CHAT_MODEL"]
    start = time.perf_counter()
    with agent_lock:
        tool: SearchTool = resources["tool"]
        try:
            answer, trace = run_agent(resources["llm"], tool, req.soru, verbose=False, history=history,
                                      on_event=on_event)
        except Exception as e:
            # Ayrıntı yalnızca log'a; kullanıcıya iç hata mesajı gösterilmez
            log_query(req.soru, None, [], [], model, {}, 0.0, time.perf_counter() - start,
                      error=f"{type(e).__name__}: {e}"[:1000])
            if isinstance(e, openai.RateLimitError):
                raise HTTPException(429, "Ücretsiz katmanın istek sınırına ulaşıldı; bir dakika sonra tekrar dene.")
            if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)):
                raise HTTPException(503, "Model şu an yoğun veya erişilemiyor; biraz sonra tekrar dene.")
            raise HTTPException(500, "Bir hata oluştu.")
        chunks = list(tool.sources)  # kilit bırakılmadan kopyala: sonraki istek reset() yapacak
    duration = time.perf_counter() - start

    valid, _ = check_citations(answer, len(chunks))
    sources = [Source(no=i, baslik=c["baslik"], url=c["url"], tarih=c["tarih"], atif=i in valid)
               for i, c in enumerate(chunks, start=1)]
    searches = [args for step in trace for args in step["aramalar"]]
    usage = {k: sum(step[k] for step in trace) for k in ("girdi", "cikti", "dusunme")}
    cost = estimate_cost(model, usage["girdi"], usage["cikti"], usage["dusunme"])

    query_id = log_query(req.soru, answer, [s.model_dump() for s in sources], searches, model,
                         usage, cost, duration)
    return AskResponse(id=query_id, cevap=answer, kaynaklar=sources, aramalar=searches,
                       token=usage, maliyet=cost, sure=round(duration, 2))


# async def DEĞİL: run_agent bloklayan bir fonksiyon (Gemini'yi ve reranker'ı bekler). async def
# içinde tüm sunucuyu dondururdu; def ile FastAPI isteği ayrı bir thread'de çalıştırır.
@app.post("/sor")
def ask(req: AskRequest) -> AskResponse:
    return answer_question(req)


@app.post("/sor-akis")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Aynı cevap, olaylar oluştukça: her satır bir JSON (NDJSON).
    {"olay": "arama"|"metin"|"sifirla"} ... ve sonda {"olay": "bitti", ...AskResponse} veya {"olay": "hata"}."""
    events: queue.Queue[dict | None] = queue.Queue()

    def worker() -> None:
        # Agent "itme" (callback) ile olay üretir, HTTP akışı "çekme" ile okur → arada kuyruk
        try:
            events.put({"olay": "bitti", **answer_question(req, events.put).model_dump()})
        except HTTPException as e:
            events.put({"olay": "hata", "kod": e.status_code, "mesaj": e.detail})
        except Exception:
            events.put({"olay": "hata", "kod": 500, "mesaj": "Bir hata oluştu."})
        finally:
            events.put(None)  # akış sonu işareti

    threading.Thread(target=worker, daemon=True).start()

    def lines():
        while (event := events.get()) is not None:
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.post("/geri-bildirim")
def feedback(req: FeedbackRequest) -> dict:
    if not set_feedback(req.id, req.puan):
        raise HTTPException(404, f"{req.id} numaralı kayıt yok.")
    return {"ok": True}
