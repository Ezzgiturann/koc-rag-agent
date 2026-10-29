"""İndirilen PDF'lerden Docling ile sayfa sayfa metin (tablolar markdown olarak) çıkarır.

Her PDF için data/pdf/<ad>.pages.jsonl yazılır; biçim data/pages.jsonl ile aynı
(url, baslik, tarih, metin), böylece chunker iki kaynağı aynı şekilde işler.
URL'ye #page=N eklenir: tarayıcı PDF'i doğrudan o sayfada açar → sayfa düzeyinde atıf.

Docling ilk çalıştırmada yerleşim ve tablo modellerini indirir (Hugging Face önbelleği);
dönüştürme tamamen yereldir.

Çalıştırma:
    python -m crawler.pdf_extract 2026-2-ceyrek.pdf   # tek dosya (deneme)
    python -m crawler.pdf_extract                     # hepsi; işlenmiş olanlar atlanır
"""

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from crawler.pdf_fetch import PDF_DIR, REPORTS

MIN_CHARS = 200  # kapak, bölüm ayracı, boş sayfa; HTML'deki eşikle aynı
TABLE_SEPARATOR = re.compile(r"^\|[-:| ]+\|$", re.M)


def get_converter() -> DocumentConverter:
    options = PdfPipelineOptions()
    # Bu PDF'ler dijital üretilmiş: metin zaten içinde. OCR (görüntüden yazı okuma) taranmış
    # kâğıtlar içindir; burada yavaşlatır ve hata ekler.
    options.do_ocr = False
    options.do_table_structure = True  # TableFormer: tabloyu satır/sütun yapısıyla çıkar
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def extract_pdf(converter: DocumentConverter, report: dict) -> list[dict]:
    """Bir PDF'i dönüştürür; sayfa başına bir kayıt döndürür (kısa sayfalar hariç)."""
    doc = converter.convert(PDF_DIR / report["dosya"]).document
    now = datetime.now(timezone.utc).isoformat()
    records = []
    for page in range(1, doc.num_pages() + 1):
        # Varsayılan içerik katmanı sayfa üst/alt bilgisini ("furniture") dışarıda bırakır:
        # her sayfada tekrar eden "Koç Holding 2024 Faaliyet Raporu | 45" satırları gelmez
        text = doc.export_to_markdown(page_no=page, image_placeholder="").strip()
        if len(text) < MIN_CHARS:
            continue
        records.append({
            "url": f"{report['url']}#page={page}",
            "baslik": f"{report['baslik']}, s. {page}",
            "tarih": report["tarih"],
            "metin": text,
            "karakter": len(text),
            "kaynak_tur": "pdf",
            "cikarilma_zamani": now,
        })
    return records


def main() -> None:
    wanted = sys.argv[1:]  # dosya adı verilmezse hepsi
    converter = get_converter()
    for report in REPORTS:
        if wanted and report["dosya"] not in wanted:
            continue
        out_path = PDF_DIR / report["dosya"].replace(".pdf", ".pages.jsonl")
        if out_path.exists():
            print(f"  zaten işlenmiş: {out_path}")
            continue

        start = time.perf_counter()
        records = extract_pdf(converter, report)
        elapsed = time.perf_counter() - start
        # Önce geçici dosyaya yaz, sonra adını değiştir: yarıda kesilirse yarım dosya "işlenmiş" sayılmasın
        tmp_path = out_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp_path.rename(out_path)

        # Markdown tablosunun başlık ayıracı satırı (|---|---|) her tabloda bir kez geçer
        n_tables = sum(len(TABLE_SEPARATOR.findall(r["metin"])) for r in records)
        print(f"  {report['baslik']}: {len(records)} sayfa, {n_tables} tablo, "
              f"{sum(r['karakter'] for r in records):,} karakter, {elapsed:.0f} sn → {out_path}")


if __name__ == "__main__":
    main()
