"""Token sayımı ve maliyet hesabı.

Çalıştırma:
    python -m agent.cost "Koç Holding'in otomotiv şirketleri hangileri?"
"""

import os
import sys

import tiktoken

from agent.ask import ask, get_client

# 1M token başına USD fiyatlar (ücretli katman).
# Kaynak: https://ai.google.dev/gemini-api/docs/pricing — kontrol tarihi 2026-10-20
# Not: 3.8 Flash fiyatı 1 Ocak 2027'de iki katına çıkıyor (girdi 1.50, çıktı 7.50).
# Not: Gemini'de "düşünme" token'ları çıktı fiyatından faturalanır.
PRICES: dict[str, dict[str, float]] = {
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    # 3.8 Flash aşırı yoğun olduğunda yedek model (kontrol tarihi 2026-10-27)
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    # Yerel model (Ollama): API ücreti yok; karşılaştırmada maliyet sütunu 0 görünsün
    "gemma4-koc": {"input": 0.0, "output": 0.0},
}


def count_tokens(text: str) -> int:
    """Metnin token sayısını tiktoken ile tahmin eder.

    o200k_base OpenAI'nin güncel tokenizer'ı. Gemini kendi tokenizer'ını
    kullandığı için bu sayı Gemini'nin raporladığından biraz farklı çıkar;
    istek atmadan önce kaba tahmin için yeterli.
    """
    encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(text))


def estimate_cost(model: str, input_tokens: int, output_tokens: int, thinking_tokens: int = 0) -> float:
    """Bir isteğin ücretli katmandaki maliyetini USD olarak döndürür.

    PRICES sözlüğündeki fiyatlar 1 MİLYON token başınadır.
    """
    # Bilinmeyen model → sessizce 0 döndürmek yerine hata: yanlış "bedava" raporu olmasın
    if model not in PRICES:
        raise ValueError(f"'{model}' için fiyat yok; PRICES sözlüğüne ekle.")
    price = PRICES[model]

    # Düşünme token'ları çıktı fiyatından faturalanır
    billed_output = output_tokens + thinking_tokens
    # Fiyat 1M token başına olduğu için 1_000_000'a bölüyoruz
    return (input_tokens * price["input"] + billed_output * price["output"]) / 1_000_000


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Koç Holding'in otomotiv şirketleri hangileri?"
    model = os.environ.get("CHAT_MODEL", "gemini-3.8-flash")

    # İstek atmadan ÖNCE tahmin (sadece soru metni; system mesajı ve format token'ları hariç)
    estimated = count_tokens(question)

    answer, usage = ask(get_client(), question)
    thinking = usage["total_tokens"] - usage["prompt_tokens"] - usage["completion_tokens"]

    print(f"Soru: {question}\n")
    print(f"tiktoken tahmini (sadece soru) : {estimated} token")
    print(f"Gemini'nin raporladığı girdi   : {usage['prompt_tokens']} token")
    print(f"Çıktı / düşünme                : {usage['completion_tokens']} / {thinking} token")

    cost = estimate_cost(model, usage["prompt_tokens"], usage["completion_tokens"], thinking)
    print(f"\nBu isteğin ücretli katmanda maliyeti: ${cost:.6f}")
    print(f"Aynı isteğin 1.000 kez tekrarı      : ${cost * 1000:.4f}")
    print("Ücretsiz katmanda gerçek maliyet    : $0")


if __name__ == "__main__":
    main()
