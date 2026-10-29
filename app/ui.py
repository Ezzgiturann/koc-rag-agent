"""Sohbet arayüzü: soruları FastAPI servisine (app/api.py) gönderir, cevabı ve kaynakları gösterir.

Streamlit her etkileşimde (mesaj, buton) bu dosyayı BAŞTAN çalıştırır. Kalıcı olması gereken
her şey (sohbet, geri bildirimler) st.session_state'te tutulur. Ağır modeller burada değil,
API sürecinde: arayüz yalnızca HTTP isteği atar.

Çalıştırma (önce başka bir terminalde: uvicorn app.api:app --reload):
    streamlit run app/ui.py
"""

import json
import os
import re

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
TIMEOUT = 180  # sn; agent en kötü durumda 5 model çağrısı + aramalar yapar
HISTORY_TURNS = 3  # API'ye gönderilen son soru-cevap çifti (add_turn da 3 tutar)
# Yalnızca bu biçimdeki linkler tıklanabilir; taranan veride başka bir adres olsa düz metin kalır
SAFE_URL = re.compile(r"https://www\.koc\.com\.tr/[\w\-./%?=&#]*")


def escape_md(text: str) -> str:
    """Markdown/Streamlit özel karakterlerini kaçışlar: başlıktaki '[tıkla](kötü-link)' link olmasın.
    ':' Streamlit'in :red[...] renk ve $ LaTeX sözdizimi için."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$:])", r"\\\1", text)


def safe_answer(text: str) -> str:
    """Cevaptaki markdown linklerini/görsellerini düz metne çevirir. Cevap taranan (güvenilmez)
    metinden üretiliyor; kalın yazı ve madde işaretleri kalsın, linkler kalmasın."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return text.replace("$", r"\$")  # "65,8 milyar $ ... 47,1 milyar $" LaTeX sanılmasın


def group_sources(sources: list[dict]) -> list[dict]:
    """Aynı URL'nin chunk'larını tek satırda toplar; atıf yapılan sayfalar üstte."""
    groups: dict[str, dict] = {}
    for s in sources:
        g = groups.setdefault(s["url"], {**s, "nolar": [], "atif": False})
        g["nolar"].append(s["no"])
        g["atif"] = g["atif"] or s["atif"]
    return sorted(groups.values(), key=lambda g: (not g["atif"], g["nolar"][0]))


def source_line(g: dict) -> str:
    numbers = "".join(f"\\[{n}\\]" for n in g["nolar"])  # [1] link sözdizimine karışmasın
    title = escape_md(g["baslik"][:100])
    link = f"[{title}]({g['url']})" if SAFE_URL.fullmatch(g["url"]) else title
    mark = "**·**" if g["atif"] else "&nbsp;"  # &nbsp;: HTML etiketi değil karakter varlığı; hizalama için
    return f"{mark} {numbers} {link} — {g['tarih'] or 'tarihsiz'}"


def api_error(response: httpx.Response) -> str:
    """API'nin hata mesajını çıkarır; 422'de detail bir liste olur."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, str):
        return detail
    if response.status_code == 422:
        return "Soru geçersiz: 1–1.000 karakter olmalı."
    return f"Beklenmeyen hata (HTTP {response.status_code})."


def ask_api(question: str, status, placeholder) -> dict:
    """Soruyu geçmişle birlikte /sor-akis'a gönderir; olaylar geldikçe ekranı günceller.

    status: aramaların listelendiği durum kutusu; placeholder: cevabın kelime kelime yazıldığı alan.
    Döndürür: {"ok": True, ...cevap} veya {"ok": False, "hata": ...}.
    """
    msgs = st.session_state.messages
    # Yalnızca başarılı cevaplar geçmişe girer: hata mesajı modele "önceki cevap" gibi gitmesin
    pairs = [{"soru": q["content"], "cevap": a["content"]}
             for q, a in zip(msgs, msgs[1:]) if q["role"] == "user" and a.get("meta")]
    text = ""
    n_search = 0
    try:
        with httpx.stream("POST", f"{API_URL}/sor-akis", json={"soru": question, "gecmis": pairs[-HISTORY_TURNS:]},
                          timeout=TIMEOUT) as r:
            if r.status_code != 200:  # ör. 422: doğrulama hatası akış başlamadan döner
                r.read()
                return {"ok": False, "hata": api_error(r)}
            for line in r.iter_lines():  # her satır bir olay (NDJSON)
                event = json.loads(line)
                if event["olay"] == "arama":
                    n_search += 1
                    status.update(label=f"🔍 Aranıyor ({n_search})…")
                    status.write(f"🔍 {escape_md(event['sorgu'][:100])} → {event['kaynak']} kaynak")
                elif event["olay"] == "metin":
                    text += event["parca"]
                    placeholder.markdown(safe_answer(text) + " ▌")  # ▌: yazıyor imleci
                elif event["olay"] == "sifirla":  # yazılan metin cevap değilmiş, model yine arıyor
                    text = ""
                    placeholder.empty()
                elif event["olay"] == "bitti":
                    status.update(label=f"✓ {n_search} arama", state="complete")
                    return {"ok": True, **{k: v for k, v in event.items() if k != "olay"}}
                elif event["olay"] == "hata":
                    status.update(label="Hata", state="error")
                    return {"ok": False, "hata": event["mesaj"]}
    except httpx.TimeoutException:
        return {"ok": False, "hata": "Cevap çok uzun sürdü; tekrar dene."}
    except httpx.ConnectError:
        return {"ok": False, "hata": "API'ye bağlanılamadı. Çalışıyor mu? (uvicorn app.api:app --reload)"}
    return {"ok": False, "hata": "Bağlantı cevap tamamlanmadan kesildi."}


def send_feedback(index: int, score: int) -> None:
    """Buton callback'i: Streamlit bunu betiği yeniden çalıştırmadan ÖNCE çağırır."""
    msg = st.session_state.messages[index]
    try:
        r = httpx.post(f"{API_URL}/geri-bildirim", json={"id": msg["meta"]["id"], "puan": score}, timeout=10)
        ok = r.status_code == 200
    except httpx.HTTPError:
        ok = False
    if ok:
        msg["feedback"] = score
    else:
        st.session_state.notice = "Geri bildirim kaydedilemedi."


def render_assistant(index: int, msg: dict) -> None:
    if "meta" not in msg:  # hata mesajı
        st.error(msg["content"])
        return
    meta = msg["meta"]
    st.markdown(safe_answer(msg["content"]))
    groups = group_sources(meta["kaynaklar"])
    if groups:
        n_cited = sum(g["atif"] for g in groups)
        with st.expander(f"Kaynaklar ({n_cited} atıf yapılan / {len(groups)} sayfa)"):
            st.markdown("  \n".join(source_line(g) for g in groups))  # "  \n" = markdown satır sonu
    t = meta["token"]
    info = f"{len(meta['aramalar'])} arama · {t['girdi'] + t['cikti'] + t['dusunme']:,} token · {meta['sure']:.1f} sn"
    left, up, down = st.columns([8, 1, 1])
    left.caption(info.replace(",", "."))
    if msg["feedback"] is None:
        up.button("👍", key=f"up{index}", on_click=send_feedback, args=(index, 1))
        down.button("👎", key=f"down{index}", on_click=send_feedback, args=(index, -1))
    else:
        up.caption("👍" if msg["feedback"] == 1 else "👎")


# --- Sayfa (her etkileşimde yukarıdan aşağı çalışır) ---
st.set_page_config(page_title="Koç Holding Asistanı", page_icon="🏢")
st.session_state.setdefault("messages", [])  # ilk açılışta boş sohbet; sonra korunur

title, clear = st.columns([4, 1])
title.title("Koç Holding Asistanı")
if clear.button("Sohbeti temizle"):
    st.session_state.messages = []

try:
    health = httpx.get(f"{API_URL}/saglik", timeout=3).json()
    st.caption(f"🟢 API {health['durum']} · {health['model']} · cevaplar yalnızca koc.com.tr içeriğine dayanır")
except httpx.HTTPError:
    st.warning("API kapalı. Başka bir terminalde çalıştır: `uvicorn app.api:app --reload`")

if notice := st.session_state.pop("notice", None):
    st.toast(notice)

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.text(msg["content"])  # kullanıcı metni markdown olarak yorumlanmaz
        else:
            render_assistant(i, msg)

if question := st.chat_input("Sorunu yaz..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.text(question)
    with st.chat_message("assistant"):
        # Spinner yerine canlı ilerleme: aramalar durum kutusunda, cevap geldikçe yazılır
        status = st.status("Düşünüyor…", expanded=False)
        placeholder = st.empty()
        result = ask_api(question, status, placeholder)
    if result.pop("ok"):
        answer = result.pop("cevap")
        st.session_state.messages.append({"role": "assistant", "content": answer, "meta": result, "feedback": None})
    else:
        st.session_state.messages.append({"role": "assistant", "content": result["hata"]})
    st.rerun()  # yeni mesajı butonlarıyla birlikte yukarıdaki döngü çizsin
