"""Agent döngüsü: model arama yapıp yapmayacağına, neyi ve kaç kez arayacağına kendisi karar verir.

Döngü (framework yok, düz bir for):
  1) Model mesaj geçmişini ve araç tanımlarını görür.
  2) tool_calls döndürürse araçları çalıştırıp sonuçları role="tool" ile ekleriz, başa döneriz.
  3) Metin döndürürse cevap budur.
  MAX_STEPS dolarsa araçsız son bir çağrıyla elindekiyle cevaplatırız.

Çalıştırma:
    python -m agent.loop "Koç Topluluğu'nun ihracat payı 2016'dan bugüne nasıl değişti?"
    python -m agent.loop            # soru verilmezse takip sorulu 3 soruluk sohbet
    python -m agent.loop --sohbet   # terminalden etkileşimli sohbet
"""

import os
import re
import sys
from datetime import date

from agent.tools import TOOLS, SearchTool, execute_tool  # ilk import: GRPC ayarı (search.py) önce yüklensin

from openai import OpenAI

from agent.ask import get_client
from agent.cost import estimate_cost
from agent.rag import SYSTEM_PROMPT

MAX_STEPS = 4  # en fazla kaç model çağrısı araç kullanabilir; sonsuz döngüye ve kota tüketimine karşı
NO_ANSWER = "Arama sınırına ulaşıldı; kaynaklardan kesin bir cevap oluşturulamadı."

# rag.py'nin 6 kuralı aynen + araç kullanımına dair kurallar
AGENT_PROMPT = SYSTEM_PROMPT + """
7. Koç Holding hakkındaki her bilgi sorusunda önce 'ara' aracını kullan; cevabı yalnızca
   araç sonuçlarındaki <kaynak> metinlerine dayandır. Atıf numaraları kaynakların id'leridir.
8. Sorunun birden çok parçası varsa her parça için ayrı ara. Sonuçlar cevabı içermiyorsa
   bir kez farklı kelimelerle tekrar ara; yine yoksa "Bu bilgi kaynaklarda yok." de.
9. Selamlaşma gibi bilgi gerektirmeyen mesajlarda arama yapma.
10. Bugünün tarihi: {today}. "Güncel", "bugün", "son" gibi ifadeleri buna göre yorumla;
    dönem karşılaştırmalarında her dönem için yıl parametreleriyle ayrı ara.
11. Önceki konuşma yalnızca yeni sorunun neyi kastettiğini anlamak içindir ("peki ya 2016'da?" →
    konu aynı, dönem farklı). Önceki cevaplardaki bilgileri kaynak sayma; yeniden ara."""

MAX_HISTORY_TURNS = 3  # geçmişte tutulan soru-cevap çifti; bağlam büyümesin


def add_turn(history: list[dict], question: str, answer: str) -> list[dict]:
    """Soru-cevabı geçmişe ekler. Atıflar silinir: yeni soruda numaralar sıfırdan başlar,
    eski [3] başka bir kaynağı gösterirdi. Yalnızca son MAX_HISTORY_TURNS çift tutulur."""
    clean = re.sub(r"\s*\[[\d,\s]+\]", "", answer)
    history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": clean}]
    return history[-2 * MAX_HISTORY_TURNS:]


def usage_of(response) -> dict[str, int]:
    """Gemini düşünme token'larını ayrı raporlamıyor; farkı kendimiz hesaplıyoruz."""
    u = response.usage
    return {"girdi": u.prompt_tokens, "cikti": u.completion_tokens,
            "dusunme": u.total_tokens - u.prompt_tokens - u.completion_tokens}


def run_agent(
    llm: OpenAI, tool: SearchTool, question: str, max_steps: int = MAX_STEPS, verbose: bool = True,
    history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """Soruyu agent döngüsüyle cevaplar; (cevap, adım izi) döndürür. Kaynaklar tool.sources'ta.

    history: önceki soru-cevaplar (add_turn ile); takip sorularını anlamak için.
    """
    tool.reset()
    messages: list[dict] = [
        # Model bugünün tarihini bilmez (eğitim verisi geçmişte kalır); her çağrıda biz söylüyoruz
        {"role": "system", "content": AGENT_PROMPT.replace("{today}", date.today().isoformat())},
        *(history or []),
        {"role": "user", "content": question},
    ]
    trace: list[dict] = []
    model = os.environ["CHAT_MODEL"]

    for step in range(1, max_steps + 1):
        response = llm.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, tool_choice="auto", reasoning_effort="low",
        )
        message = response.choices[0].message
        entry = {"adim": step, **usage_of(response), "aramalar": []}
        trace.append(entry)

        if not message.tool_calls:  # model araç istemedi → cevap hazır
            if verbose:
                print(f"  adım {step}: cevap yazıldı  [girdi {entry['girdi']} token]")
            return message.content or "", trace

        # Modelin mesajını OLDUĞU GİBİ ekle: thought_signature olmadan sonraki çağrı 400 verir
        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            result = execute_tool(tool, call.function.name, call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            entry["aramalar"].append(call.function.arguments)
            if verbose:
                found = len(re.findall(r"<kaynak id=", result)) or result[:60]
                print(f"  adım {step}: {call.function.name}({call.function.arguments}) → {found} kaynak"
                      f"  [girdi {entry['girdi']} token]")

    # Adım sınırı doldu: elindekiyle cevaplat. tool_choice="none" yetmiyor: Gemini bazen yine
    # tool_calls döndürüyor (7. gün, boş cevap). Araç tanımı hiç gönderilmezse çağıramaz.
    response = llm.chat.completions.create(model=model, messages=messages, reasoning_effort="low")
    trace.append({"adim": max_steps + 1, **usage_of(response), "aramalar": [], "not": "adım sınırı"})
    if verbose:
        print(f"  adım {max_steps + 1}: adım sınırı doldu, araçsız cevap istendi")
    # Son güvence: kullanıcı hiçbir durumda boş cevap görmesin
    return response.choices[0].message.content or NO_ANSWER, trace


def check_citations(answer: str, n_sources: int) -> tuple[set[int], set[int]]:
    """Cevaptaki [n] atıflarını bulur; (geçerli, geçersiz) numara kümelerini döndürür."""
    # Model [6][9] yerine bazen [6, 9] yazıyor; köşeli parantez içindeki tüm sayıları al
    cited = {int(n) for group in re.findall(r"\[([\d,\s]+)\]", answer) for n in re.findall(r"\d+", group)}
    return {n for n in cited if 1 <= n <= n_sources}, {n for n in cited if not 1 <= n <= n_sources}


def print_result(answer: str, trace: list[dict], tool: SearchTool) -> None:
    """Cevabı, kaynak listesini (atıf yapılanlar işaretli) ve token/maliyet özetini yazdırır."""
    print(f"\n{answer}\n")
    valid, invalid = check_citations(answer, len(tool.sources))
    if tool.sources:
        print("Kaynaklar (* = cevapta atıf yapılan):")
        for i, c in enumerate(tool.sources, start=1):
            mark = "*" if i in valid else " "
            print(f" {mark}[{i}] {c['baslik'][:70]} — {c['tarih'] or 'tarihsiz'}")
    if invalid:
        print(f"UYARI: var olmayan kaynaklara atıf: {sorted(invalid)}")

    total = {k: sum(t[k] for t in trace) for k in ("girdi", "cikti", "dusunme")}
    cost = estimate_cost(os.environ["CHAT_MODEL"], total["girdi"], total["cikti"], total["dusunme"])
    n_search = sum(len(t["aramalar"]) for t in trace)
    print(f"[{len(trace)} model çağrısı, {n_search} arama | girdi {total['girdi']} / çıktı {total['cikti']} / "
          f"düşünme {total['dusunme']} token, ${cost:.6f}]")


def main() -> None:
    llm = get_client()
    tool = SearchTool()
    history: list[dict] = []

    if sys.argv[1:] == ["--sohbet"]:
        # Etkileşimli mod: boş satır veya Ctrl+D ile çıkılır
        print("Sohbet modu (çıkmak için boş satır). Takip soruları önceki cevapları hatırlar.")
        while question := input("\nSen: ").strip():
            answer, trace = run_agent(llm, tool, question, history=history)
            print_result(answer, trace, tool)
            history = add_turn(history, question, answer)
    else:
        # Varsayılan: takip soruları içeren kısa bir sohbet; 2. ve 3. soru tek başına anlamsız
        questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else [
            "Koç Topluluğu Türkiye ihracatının yüzde kaçını gerçekleştiriyor?",
            "Peki 2016'da bu oran neydi?",  # konu: ihracat payı; dönem: 2016
            "Bu bilgiyi kim açıklamış?",  # "bu bilgi" = 2016 oranı
        ]
        for question in questions:
            print(f"\n=== Soru: {question}")
            answer, trace = run_agent(llm, tool, question, history=history)
            print_result(answer, trace, tool)
            history = add_turn(history, question, answer)
    tool.close()


if __name__ == "__main__":
    main()
