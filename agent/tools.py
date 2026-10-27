"""Agent'ın araçları: tanım (JSON Schema) ve çalıştırma kodu bir arada.

Şu an tek araç var: `ara` — 4. günün akışı (vektör + BM25 → RRF → reranker →
ardışık chunk birleştirme). Agent bir soruda birden çok kez arayabildiği için
kaynak numaraları tüm aramalar boyunca TEK sayaçtan verilir: 2. aramanın
sonuçları [6]'dan başlar, aynı chunk tekrar gelirse ilk numarasını korur.

Çalıştırma (LLM'siz deneme):
    python -m agent.tools
"""

import json

from retrieval.search import check_index_model  # GRPC ayarı için pymilvus'tan önce

from agent.rag import format_source, merge_adjacent
from indexing.build_index import COLLECTION, get_milvus_client
from indexing.embedder import get_embedding_client
from retrieval.bm25 import BM25Index, load_chunks
from retrieval.rerank import load_reranker, search_and_rerank

MAX_QUERY_CHARS = 200  # modelin göndereceği sorgu için makul üst sınır

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ara",
            "description": (
                "koc.com.tr içeriğinde (haberler, kurumsal sayfalar, yönetici özgeçmişleri) arama yapar; "
                "en ilgili 5 metin parçasını numaralı <kaynak> etiketleriyle döndürür. "
                "Sorunun birden çok parçası varsa (ör. iki yılı karşılaştırma) her parça için ayrı ara. "
                "Sonuç yetersizse farklı kelimelerle (eşanlamlı, resmi terim) tekrar ara."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sorgu": {
                        "type": "string",
                        "description": "Kısa arama ifadesi; sitede geçmesi muhtemel kelimelerle, ör. 'Koç Topluluğu ihracat payı'",
                    },
                },
                "required": ["sorgu"],
            },
        },
    }
]


class SearchTool:
    """Arama için ağır kaynakları (Milvus, BM25, reranker) bir kez yükler ve kaynak numaralarını tutar."""

    def __init__(self) -> None:
        self.milvus = get_milvus_client()
        check_index_model(self.milvus)
        self.milvus.load_collection(COLLECTION)
        self.emb_client = get_embedding_client()
        self.bm25 = BM25Index(load_chunks(), stem=True)
        self.reranker = load_reranker()
        self.reset()

    def reset(self) -> None:
        """Yeni bir soru için numaralandırmayı sıfırlar."""
        self.numbers: dict[str, int] = {}  # chunk_id → kaynak numarası
        self.sources: list[dict] = []  # numara sırasıyla görülen kaynaklar (cevabın altına liste için)

    def search(self, query: str) -> str:
        """Aramayı yapar; sonuçları tek sayaçla numaralanmış <kaynak> blokları olarak döndürür."""
        chunks = merge_adjacent(search_and_rerank(self.milvus, self.emb_client, self.bm25, self.reranker, query))
        blocks = []
        for c in chunks:
            if c["chunk_id"] not in self.numbers:  # daha önce görülmediyse yeni numara ver
                self.sources.append(c)
                self.numbers[c["chunk_id"]] = len(self.sources)
            blocks.append(format_source(self.numbers[c["chunk_id"]], c))
        return "\n\n".join(blocks)

    def close(self) -> None:
        self.milvus.close()


def execute_tool(tool: SearchTool, name: str, arguments: str) -> str:
    """Modelin istediği aracı çalıştırır. Hatalar istisna olarak değil METİN olarak döner:
    program çökmez, model hatayı görüp isteğini düzeltebilir."""
    if name != "ara":
        return f"HATA: '{name}' adında bir araç yok. Kullanılabilir araç: ara"
    try:
        args = json.loads(arguments)  # parametreler JSON metni olarak gelir; bozuk olabilir
    except json.JSONDecodeError:
        return "HATA: Parametreler geçerli JSON değil."
    query = str(args.get("sorgu", "")).strip()
    if not query:
        return "HATA: 'sorgu' parametresi boş olamaz."
    return tool.search(query[:MAX_QUERY_CHARS])


def main() -> None:
    tool = SearchTool()
    for name, arguments in [
        ("ara", json.dumps({"sorgu": "Koç Topluluğu ihracat payı 2026"})),
        ("ara", json.dumps({"sorgu": "Koç Topluluğu ihracat payı 2016"})),  # numaralar 6'dan devam etmeli
        ("ara", json.dumps({"sorgu": "Koç Topluluğu ihracat payı 2026"})),  # aynı sonuçlar: aynı numaralar
        ("ara", "{bozuk json"),
        ("sil", "{}"),
    ]:
        result = execute_tool(tool, name, arguments)
        print(f"\n>>> {name}({arguments})")
        # Sadece etiket satırlarını göster: hangi numara hangi kaynağa verildi?
        for line in result.splitlines():
            if line.startswith("<kaynak") or line.startswith("HATA"):
                print("   ", line[:130])
    print(f"\nToplam farklı kaynak: {len(tool.sources)}")
    tool.close()


if __name__ == "__main__":
    main()
