import html
import json
from pathlib import Path

import pandas as pd

from config import PRELOADED_DIR
from document_parser import (
    STRUCTURED_TABLE_ALLOWLIST,
    OCR_FUTURE_TABLES,
    detect_captions,
    get_page_size,
    get_page_lines,
    find_object_end_y,
    extract_words_in_bbox,
    reconstruct_2021_table2,
    reconstruct_2021_table5,
    reconstruct_2025_table3,
    extract_generic_table_with_pdfplumber,
)


OUTPUT_DIR = Path("verification_output_html")
OUTPUT_DIR.mkdir(exist_ok=True)


def safe_name(text: str) -> str:
    text = str(text or "")
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', " "]:
        text = text.replace(ch, "_")
    return text[:160]


def dataframe_to_html_table(df: pd.DataFrame) -> str:
    """
    Converts DataFrame to HTML manually so multiline text is preserved with <br>.
    """
    html_parts = []
    html_parts.append("<div class='table-scroll'>")
    html_parts.append("<table>")

    # Header
    html_parts.append("<thead><tr>")
    for col in df.columns:
        html_parts.append(f"<th>{html.escape(str(col))}</th>")
    html_parts.append("</tr></thead>")

    # Body
    html_parts.append("<tbody>")
    for _, row in df.iterrows():
        html_parts.append("<tr>")
        for col in df.columns:
            value = "" if pd.isna(row[col]) else str(row[col])
            value = html.escape(value).replace("\n", "<br>")
            html_parts.append(f"<td>{value}</td>")
        html_parts.append("</tr>")
    html_parts.append("</tbody>")

    html_parts.append("</table>")
    html_parts.append("</div>")

    return "\n".join(html_parts)


def full_html_page(title: str, body: str) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
    body {{
        font-family: Arial, sans-serif;
        margin: 28px;
        background: #ffffff;
        color: #1f1f1f;
    }}

    h1 {{
        color: #17324d;
        margin-bottom: 6px;
    }}

    h2 {{
        color: #17324d;
        margin-top: 34px;
        border-bottom: 1px solid #dddddd;
        padding-bottom: 8px;
    }}

    .meta {{
        background: #f7f7f7;
        border: 1px solid #dddddd;
        border-radius: 10px;
        padding: 12px 14px;
        margin: 12px 0 18px 0;
        line-height: 1.55;
        font-size: 14px;
    }}

    .status-ok {{
        color: #176b36;
        font-weight: bold;
    }}

    .status-warn {{
        color: #9a4d00;
        font-weight: bold;
    }}

    .table-scroll {{
        overflow-x: auto;
        border: 1px solid #d9d9d9;
        border-radius: 10px;
        margin-top: 14px;
    }}

    table {{
        border-collapse: collapse;
        width: 100%;
        min-width: 900px;
        font-size: 14px;
    }}

    th {{
        background: #17324d;
        color: white;
        padding: 10px;
        border: 1px solid #c9c9c9;
        text-align: left;
        vertical-align: top;
        position: sticky;
        top: 0;
    }}

    td {{
        padding: 10px;
        border: 1px solid #d5d5d5;
        vertical-align: top;
        line-height: 1.45;
        white-space: normal;
    }}

    tr:nth-child(even) {{
        background: #fafafa;
    }}

    .link-card {{
        border: 1px solid #dddddd;
        border-radius: 10px;
        padding: 12px;
        margin: 10px 0;
        background: #fbfbfb;
    }}

    a {{
        color: #0b5cad;
        text-decoration: none;
        font-weight: bold;
    }}

    a:hover {{
        text-decoration: underline;
    }}

    pre {{
        white-space: pre-wrap;
        background: #f6f6f6;
        border: 1px solid #dddddd;
        padding: 12px;
        border-radius: 8px;
    }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def save_table_outputs(
    df: pd.DataFrame,
    file_name: str,
    table_number: str,
    page_number: int,
    caption: str,
    method: str,
):
    base = safe_name(f"{Path(file_name).stem}_TABLE_{table_number}")
    html_file = OUTPUT_DIR / f"{base}.html"
    excel_file = OUTPUT_DIR / f"{base}.xlsx"

    # Save Excel also, useful for row/column checking
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Extracted_Table")

    table_html = dataframe_to_html_table(df)

    body = f"""
<h1>{html.escape(file_name)} - Table {html.escape(str(table_number))}</h1>

<div class="meta">
    <b>PDF:</b> {html.escape(file_name)}<br>
    <b>Page:</b> {html.escape(str(page_number))}<br>
    <b>Table:</b> {html.escape(str(table_number))}<br>
    <b>Caption:</b> {html.escape(str(caption))}<br>
    <b>Extraction method:</b> {html.escape(str(method))}<br>
    <b>Rows:</b> {df.shape[0]}<br>
    <b>Columns:</b> {df.shape[1]}<br>
    <b>Excel:</b> <a href="{excel_file.name}">{excel_file.name}</a>
</div>

{table_html}
"""

    html_file.write_text(
        full_html_page(f"{file_name} Table {table_number}", body),
        encoding="utf-8",
    )

    return {
        "file_name": file_name,
        "table_number": table_number,
        "page_number": page_number,
        "caption": caption,
        "method": method,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "html_file": html_file.name,
        "excel_file": excel_file.name,
    }


def extract_structured_tables_for_pdf(pdf_path: Path):
    file_name = pdf_path.name
    allow = STRUCTURED_TABLE_ALLOWLIST.get(file_name, set())
    ocr_table_rules = OCR_FUTURE_TABLES.get(file_name, {})

    table_captions, figure_captions = detect_captions(str(pdf_path))

    extracted = []
    skipped = []

    for cap in table_captions:
        table_number = cap["number"]
        page_number = cap["page"]
        caption = cap["caption"]

        if table_number in ocr_table_rules:
            skipped.append({
                "file_name": file_name,
                "table_number": table_number,
                "page_number": page_number,
                "caption": caption,
                "reason": ocr_table_rules[table_number],
            })
            continue

        if allow and table_number not in allow:
            skipped.append({
                "file_name": file_name,
                "table_number": table_number,
                "page_number": page_number,
                "caption": caption,
                "reason": "Not in structured extraction allowlist.",
            })
            continue

        page_width, page_height = get_page_size(str(pdf_path), page_number)
        lines = get_page_lines(str(pdf_path), page_number)
        caption_y = cap["bbox"][1]
        end_y = find_object_end_y(lines, caption_y, page_height)

        structured_bbox = [
            15,
            max(caption_y + 15, 0),
            page_width - 15,
            min(end_y, page_height),
        ]

        df = None
        method = ""

        if file_name == "2021_pdf.pdf" and table_number == "2":
            words = extract_words_in_bbox(str(pdf_path), page_number, structured_bbox)
            df = reconstruct_2021_table2(words, structured_bbox)
            method = "custom_2021_table2_word_position"


        elif file_name == "2021_pdf.pdf" and table_number == "5":
            words = extract_words_in_bbox(str(pdf_path), page_number, structured_bbox)
            df = reconstruct_2021_table5(words, structured_bbox)
            method = "custom_2021_table5_section_band"

        elif file_name == "2025_pdf.pdf" and table_number == "3":
            words = extract_words_in_bbox(str(pdf_path), page_number, structured_bbox)
            df = reconstruct_2025_table3(words, structured_bbox)
            method = "custom_2025_table3_section_row_column"

        else:
            df, score = extract_generic_table_with_pdfplumber(
                str(pdf_path),
                page_number,
                structured_bbox,
            )
            method = f"pdfplumber_generic_score_{score}"





        

        if df is None or df.empty:
            skipped.append({
                "file_name": file_name,
                "table_number": table_number,
                "page_number": page_number,
                "caption": caption,
                "reason": "Structured extraction returned empty table.",
            })
            continue

        result = save_table_outputs(
            df=df,
            file_name=file_name,
            table_number=table_number,
            page_number=page_number,
            caption=caption,
            method=method,
        )

        extracted.append(result)

    return extracted, skipped


def build_index(all_extracted, all_skipped):
    cards = []

    cards.append("<h1>Structured Table Extraction Verification</h1>")
    cards.append("""
<div class="meta">
This report shows only tables that were extracted as structured table data.
Open each table HTML to verify row/column structure visually.
OCR-heavy tables, figures, charts, maps and flowcharts are intentionally skipped in this text-table-only version.
</div>
""")

    cards.append("<h2>Extracted structured tables</h2>")

    if not all_extracted:
        cards.append("<div class='meta status-warn'>No structured tables were extracted.</div>")
    else:
        for item in all_extracted:
            cards.append(f"""
<div class="link-card">
    <div class="status-ok">EXTRACTED</div>
    <b>PDF:</b> {html.escape(item["file_name"])}<br>
    <b>Page:</b> {html.escape(str(item["page_number"]))}<br>
    <b>Table:</b> {html.escape(str(item["table_number"]))}<br>
    <b>Caption:</b> {html.escape(str(item["caption"]))}<br>
    <b>Rows:</b> {item["rows"]} |
    <b>Columns:</b> {item["columns"]}<br>
    <b>Method:</b> {html.escape(item["method"])}<br>
    <a href="{html.escape(item["html_file"])}">Open HTML table</a>
    &nbsp; | &nbsp;
    <a href="{html.escape(item["excel_file"])}">Open Excel</a>
</div>
""")

    cards.append("<h2>Skipped tables / future OCR items</h2>")

    if not all_skipped:
        cards.append("<div class='meta'>No skipped table items.</div>")
    else:
        for item in all_skipped:
            cards.append(f"""
<div class="link-card">
    <div class="status-warn">SKIPPED FOR FUTURE OCR</div>
    <b>PDF:</b> {html.escape(item["file_name"])}<br>
    <b>Page:</b> {html.escape(str(item["page_number"]))}<br>
    <b>Table:</b> {html.escape(str(item["table_number"]))}<br>
    <b>Caption:</b> {html.escape(str(item["caption"]))}<br>
    <b>Reason:</b> {html.escape(str(item["reason"]))}
</div>
""")

    index_html = full_html_page(
        "Structured Table Verification",
        "\n".join(cards),
    )

    index_path = OUTPUT_DIR / "index.html"
    index_path.write_text(index_html, encoding="utf-8")

    summary_path = OUTPUT_DIR / "structured_table_summary.json"
    summary = {
        "extracted": all_extracted,
        "skipped": all_skipped,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return index_path


def main():
    pdfs = sorted(PRELOADED_DIR.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in: {PRELOADED_DIR}")
        return

    all_extracted = []
    all_skipped = []

    for pdf in pdfs:
        print()
        print("=" * 80)
        print(f"Checking structured tables in: {pdf.name}")
        print("=" * 80)

        extracted, skipped = extract_structured_tables_for_pdf(pdf)

        all_extracted.extend(extracted)
        all_skipped.extend(skipped)

        print(f"Extracted structured tables: {len(extracted)}")
        print(f"Skipped table items: {len(skipped)}")

        for item in extracted:
            print(
                f"  OK: Table {item['table_number']} | "
                f"Page {item['page_number']} | "
                f"{item['rows']} rows x {item['columns']} cols | "
                f"{item['html_file']}"
            )

        for item in skipped:
            print(
                f"  SKIP: Table {item['table_number']} | "
                f"Page {item['page_number']} | "
                f"{item['reason']}"
            )

    index_path = build_index(all_extracted, all_skipped)

    print()
    print("=" * 80)
    print("HTML verification created.")
    print("=" * 80)
    print(f"Open this file:")
    print(index_path.resolve())


if __name__ == "__main__":
    main()