"""LLM'e tek bir soru gönderip cevabı ve token kullanımını yazdıran script.

Çalıştırma:
    python -m agent.ask "Koç Holding ne zaman kuruldu?"
"""

import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI


def get_client() -> OpenAI:
    """.env'deki ayarlarla OpenAI uyumlu bir istemci oluşturur (şu an Gemini)."""
    load_dotenv()
    return OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],  # yoksa KeyError: anahtar eksik demektir
        base_url=os.environ["LLM_BASE_URL"],  # isteklerin gideceği sunucu
    )


def ask(client: OpenAI, question: str, reasoning_effort: str | None = "low") -> tuple[str, dict]:
    """Soruyu modele gönderir; (cevap metni, token kullanımı) döndürür.

    reasoning_effort: modelin cevaptan önce ne kadar "düşüneceği".
    None = model kendi seçer (Gemini'de yüzlerce gizli token, daha yavaş).
    """
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    response = client.chat.completions.create(
        model=os.environ["CHAT_MODEL"],
        **extra,
        # Her mesajın bir rolü var: system = kurallar, user = kullanıcının sorusu
        messages=[
            {"role": "system", "content": "Kısa ve net cevap ver. Türkçe yaz."},
            {"role": "user", "content": question},
        ],
    )
    answer = response.choices[0].message.content or ""  # choices: model birden fazla cevap üretebilir; ilkini alıyoruz
    usage = response.usage.model_dump() if response.usage else {}
    return answer, usage


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Koç Holding ne zaman ve kim tarafından kuruldu?"
    client = get_client()

    start = time.perf_counter()
    answer, usage = ask(client, question)
    elapsed = time.perf_counter() - start

    print(f"Soru : {question}")
    print(f"Cevap: {answer}\n")
    print(f"Süre : {elapsed:.1f} sn")
    print(f"Girdi token (prompt)    : {usage.get('prompt_tokens')}")
    print(f"Çıktı token (completion): {usage.get('completion_tokens')}")
    # Gemini gizli "düşünme" token'larını ayrı raporlamıyor, sadece toplama ekliyor;
    # bu yüzden farkı kendimiz hesaplıyoruz
    thinking = usage["total_tokens"] - usage["prompt_tokens"] - usage["completion_tokens"]
    print(f"Düşünme token (gizli)   : {thinking}")
    print(f"Toplam                  : {usage.get('total_tokens')}")


if __name__ == "__main__":
    main()
