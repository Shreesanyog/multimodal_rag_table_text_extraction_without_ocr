import json
import re
from pathlib import Path
from typing import List

import fitz
import pdfplumber
import pandas as pd
from langchain_core.documents import Document

from config import PARSED_CACHE_DIR, EXTRACTION_REPORT_DIR, PAGE_IMAGES_DIR
from utils import (
    clean_text,
    normalize_inline,
    stable_doc_id,
    looks_like_contents_page,
)



# KNOWN HANDLING FOR 3 PDFs


STRUCTURED_TABLE_ALLOWLIST = {
    "2021_pdf.pdf": {"2", "5"},
    "2025_pdf.pdf": {"3"},
}

OCR_FUTURE_TABLES = {
    "2021_pdf.pdf": {
        "1": "Color/legend-style table. Normal text extraction can miss visual category cells."
    }
}

OCR_FUTURE_FIGURES = {
    "2021_pdf.pdf": {"BOX 1", "1", "2", "3", "4", "6", "8", "9"},
    "2023_pdf.pdf": {"1", "3", "5", "6", "7", "8", "9", "11", "13", "15"},
    "2025_pdf.pdf": {"3", "5", "10", "9", "11", "13", "15", "18", "24"},
}



# BASIC HELPERS


def get_page_text_pymupdf(pdf_path: str, page_index: int) -> str:
    pdf = fitz.open(pdf_path)
    text = pdf[page_index].get_text("text") or ""
    pdf.close()
    return clean_text(text)


def get_page_text_pdfplumber(pdf_path: str, page_index: int) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[page_index].extract_text() or ""
    return clean_text(text)


def best_page_text(pdf_path: str, page_index: int) -> str:
    pymu = get_page_text_pymupdf(pdf_path, page_index)
    plumber = get_page_text_pdfplumber(pdf_path, page_index)

    if len(plumber.split()) > len(pymu.split()) * 1.05:
        return plumber

    return pymu


def get_page_lines(pdf_path: str, page_number: int):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    data = page.get_text("dict")
    lines = []

    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])

            if not spans:
                continue

            text = normalize_inline("".join(span.get("text", "") for span in spans))

            if not text:
                continue

            x0 = min(span["bbox"][0] for span in spans)
            y0 = min(span["bbox"][1] for span in spans)
            x1 = max(span["bbox"][2] for span in spans)
            y1 = max(span["bbox"][3] for span in spans)

            lines.append({
                "text": text,
                "bbox": [x0, y0, x1, y1],
            })

    doc.close()
    return lines


def get_page_size(pdf_path: str, page_number: int):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    width = page.rect.width
    height = page.rect.height
    doc.close()
    return width, height


def render_page_image(pdf_path: str, page_index: int, zoom: float = 1.8) -> str:
    """
    Render PDF page as image for citation preview only.

    Important:
    - This is NOT OCR.
    - This image is NOT sent to Gemini.
    - This image is NOT embedded.
    - This is only shown in Streamlit citation card.
    """
    doc_id = stable_doc_id(pdf_path)
    out_dir = PAGE_IMAGES_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    page_number = page_index + 1
    img_path = out_dir / f"page_{page_number}.png"

    if img_path.exists():
        return str(img_path)

    pdf = fitz.open(pdf_path)
    page = pdf[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    pix.save(str(img_path))
    pdf.close()

    return str(img_path)


def detect_captions(pdf_path: str):
    file_name = Path(pdf_path).name

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    tables = []
    figures = []

    for page_number in range(1, total_pages + 1):
        lines = get_page_lines(pdf_path, page_number)
        page_text = "\n".join(line["text"] for line in lines)

        if looks_like_contents_page(page_text):
            continue

        for line in lines:
            text = line["text"].strip()

            table_match = re.match(
                r"^\s*TABLE\s+(\d+)\s+(.+)$",
                text,
                flags=re.IGNORECASE,
            )

            figure_match = re.match(
                r"^\s*FIGURE\s+(\d+)\s+(.+)$",
                text,
                flags=re.IGNORECASE,
            )

            box_figure_match = re.match(
                r"^\s*FIGURE\s+FROM\s+BOX\s+(\d+)\s+(.+)$",
                text,
                flags=re.IGNORECASE,
            )

            if table_match:
                tables.append({
                    "type": "table",
                    "number": table_match.group(1),
                    "caption": text,
                    "page": page_number,
                    "bbox": line["bbox"],
                    "file_name": file_name,
                })

            if figure_match:
                figures.append({
                    "type": "figure",
                    "number": figure_match.group(1),
                    "caption": text,
                    "page": page_number,
                    "bbox": line["bbox"],
                    "file_name": file_name,
                })

            if box_figure_match:
                figures.append({
                    "type": "figure",
                    "number": f"BOX {box_figure_match.group(1)}",
                    "caption": text,
                    "page": page_number,
                    "bbox": line["bbox"],
                    "file_name": file_name,
                })

    def unique(items):
        seen = set()
        output = []

        for item in items:
            key = (
                item.get("type"),
                item.get("number"),
                item.get("page"),
                item.get("caption"),
            )

            if key in seen:
                continue

            seen.add(key)
            output.append(item)

        return output

    return unique(tables), unique(figures)


def find_object_end_y(lines, caption_y, page_height):
    candidates = []

    for line in lines:
        y = line["bbox"][1]
        upper = line["text"].upper()

        if y > caption_y and (
            upper.startswith("NOTE:")
            or upper.startswith("NOTES:")
            or upper.startswith("SOURCE:")
        ):
            candidates.append(y)

    if candidates:
        return min(candidates)

    return page_height - 25


def extract_words_in_bbox(pdf_path: str, page_number: int, bbox):
    x0, y0, x1, y1 = bbox
    words_inside = []

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]

        words = page.extract_words(
            x_tolerance=2,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=False,
        )

        for word in words:
            wx0 = float(word["x0"])
            wx1 = float(word["x1"])
            top = float(word["top"])
            bottom = float(word["bottom"])

            if wx0 >= x0 and wx1 <= x1 and top >= y0 and bottom <= y1:
                words_inside.append(word)

    return words_inside


def group_words_into_rows(words, y_tolerance=5):
    if not words:
        return []

    words = sorted(words, key=lambda w: (float(w["top"]), float(w["x0"])))
    rows = []

    for word in words:
        word_y = float(word["top"])
        placed = False

        for row in rows:
            if abs(word_y - row["y"]) <= y_tolerance:
                row["words"].append(word)
                row["ys"].append(word_y)
                row["y"] = sum(row["ys"]) / len(row["ys"])
                placed = True
                break

        if not placed:
            rows.append({
                "y": word_y,
                "ys": [word_y],
                "words": [word],
            })

    for row in rows:
        row["words"] = sorted(row["words"], key=lambda w: float(w["x0"]))

    return sorted(rows, key=lambda r: r["y"])



# 2021 TABLE 2 RECONSTRUCTION


def clean_table2_rows(rows):
    skip_phrases = [
        "NUMBER OF PEOPLE",
        "UNABLE TO AFFORD",
        "HEALTHY DIET",
        "IF INCOMES ARE REDUCED",
        "BY ONE-THIRD",
        "PERCENT TOTAL NUMBER",
        "TABLE 2",
        "INDICATORS OF UNAFFORDABILITY",
    ]

    temp = []

    for row in rows:
        row = [normalize_inline(cell) for cell in row]
        row_text_upper = " ".join(row).upper()

        if any(phrase in row_text_upper for phrase in skip_phrases):
            continue

        if not any(row):
            continue

        temp.append(row)

    merged = []

    for row in temp:
        category = row[0]
        numeric_cells = row[1:]
        has_numeric = any(any(ch.isdigit() for ch in cell) for cell in numeric_cells)

        if merged and not has_numeric:
            merged[-1][0] = normalize_inline(merged[-1][0] + " " + category)
        else:
            merged.append(row)

    final_rows = []

    for row in merged:
        category = row[0]

        if "COUNTRY INCOME GROUPS" in category.upper():
            before = re.sub(
                r"COUNTRY\s+INCOME\s+GROUPS",
                "",
                category,
                flags=re.IGNORECASE,
            ).strip()

            if before:
                fixed_row = row.copy()
                fixed_row[0] = before
                final_rows.append(fixed_row)

            final_rows.append(["COUNTRY INCOME GROUPS", "", "", "", ""])
            continue

        row_text = " ".join(row)
        has_digit = any(ch.isdigit() for ch in row_text)

        if has_digit or category.upper() == "COUNTRY INCOME GROUPS":
            final_rows.append(row)

    return final_rows


def reconstruct_2021_table2(words, bbox):
    x0, y0, x1, y1 = bbox
    width = x1 - x0

    col_edges = [
        x0,
        x0 + width * 0.35,
        x0 + width * 0.50,
        x0 + width * 0.66,
        x0 + width * 0.82,
        x1,
    ]

    rows = group_words_into_rows(words, y_tolerance=5)
    raw_rows = []

    for row in rows:
        cells = ["", "", "", "", ""]

        for word in row["words"]:
            text = word["text"]
            center_x = (float(word["x0"]) + float(word["x1"])) / 2

            col_idx = None

            for i in range(len(col_edges) - 1):
                if col_edges[i] <= center_x < col_edges[i + 1]:
                    col_idx = i
                    break

            if col_idx is None:
                continue

            if cells[col_idx]:
                cells[col_idx] += " " + text
            else:
                cells[col_idx] = text

        cells = [normalize_inline(cell) for cell in cells]

        if any(cells):
            raw_rows.append(cells)

    cleaned = clean_table2_rows(raw_rows)

    df = pd.DataFrame(
        cleaned,
        columns=[
            "Category / Region",
            "Unable to afford healthy diet in 2019 - Percent",
            "Unable to afford healthy diet in 2019 - Total number (millions)",
            "At risk if incomes reduced by one-third - Percent",
            "At risk if incomes reduced by one-third - Total number (millions)",
        ],
    )

    return df



# 2021 TABLE 5 RECONSTRUCTION


def extract_table5_section_bands(words, bbox):
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    left_col_x1 = x0 + width * 0.22

    left_words = []

    for word in words:
        wx0 = float(word["x0"])
        wx1 = float(word["x1"])

        if wx0 >= x0 and wx1 <= left_col_x1:
            left_words.append(word)

    left_rows = group_words_into_rows(left_words, y_tolerance=7)

    section_order = [
        "CONTEXTUAL FACTORS",
        "NATIONAL AGRIFOOD SYSTEMS",
        "FOOD SUPPLY CHAINS AND ACTORS",
        "HOUSEHOLDS AND LIVELIHOODS",
    ]

    found = {}

    for row in left_rows:
        row_text = normalize_inline(" ".join(w["text"] for w in row["words"]))
        upper = row_text.upper()

        section = None

        if "CONTEXTUAL" in upper:
            section = "CONTEXTUAL FACTORS"
        elif "NATIONAL" in upper:
            section = "NATIONAL AGRIFOOD SYSTEMS"
        elif "FOOD" in upper and "SUPPLY" in upper:
            section = "FOOD SUPPLY CHAINS AND ACTORS"
        elif "HOUSEHOLDS" in upper:
            section = "HOUSEHOLDS AND LIVELIHOODS"

        if section and section not in found:
            found[section] = row["y"]

    ordered = []

    for section in section_order:
        if section in found:
            ordered.append({
                "section": section,
                "start_y": found[section],
            })

    if len(ordered) < 4:
        body_start = y0 + (y1 - y0) * 0.15
        body_end = y1
        band_height = (body_end - body_start) / 4

        ordered = []

        for i, section in enumerate(section_order):
            ordered.append({
                "section": section,
                "start_y": body_start + i * band_height,
            })

    bands = []

    for i, item in enumerate(ordered):
        start_y = item["start_y"]

        if i + 1 < len(ordered):
            end_y = ordered[i + 1]["start_y"] - 2
        else:
            end_y = y1

        bands.append({
            "section": item["section"],
            "start_y": start_y,
            "end_y": end_y,
        })

    return bands


def table5_column_edges(bbox):
    x0, y0, x1, y1 = bbox
    width = x1 - x0

    return [
        x0,
        x0 + width * 0.19,
        x0 + width * 0.46,
        x0 + width * 0.73,
        x1,
    ]


def clean_table5_line(text):
    text = normalize_inline(text)

    text = text.replace("}", "•")
    text = text.replace("›", "•")
    text = text.replace("▸", "•")
    text = text.replace("", "•")
    text = text.replace("", "•")

    return text.strip()


def join_table5_lines(lines):
    cleaned = []

    skip = [
        "TABLE 5",
        "ENTRY POINTS",
        "SHOCKS DIFFICULT",
        "TO FORESEE",
        "MORE PREDICTABLE",
        "SHOCKS",
        "ENSURING DIVERSITY",
        "MANAGING CONNECTIVITY",
        "MANAGING RISKS",
        "NOTE:",
        "SOURCE:",
    ]

    for line in lines:
        line = clean_table5_line(line)
        upper = line.upper()

        if not line:
            continue

        if any(s in upper for s in skip):
            continue

        cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n\s*•\s*", "\n• ", text)
    text = re.sub(r"^\s*•\s*", "• ", text)

    return text.strip()


def reconstruct_2021_table5(words, bbox):
    col_edges = table5_column_edges(bbox)
    bands = extract_table5_section_bands(words, bbox)

    final_rows = []

    for band in bands:
        section = band["section"]
        start_y = band["start_y"]
        end_y = band["end_y"]

        row_cells = [section, "", "", ""]

        for col_idx in [1, 2, 3]:
            col_x0 = col_edges[col_idx]
            col_x1 = col_edges[col_idx + 1]

            cell_words = []

            for word in words:
                wx0 = float(word["x0"])
                wx1 = float(word["x1"])
                top = float(word["top"])
                bottom = float(word["bottom"])

                center_x = (wx0 + wx1) / 2
                center_y = (top + bottom) / 2

                if col_x0 <= center_x < col_x1 and start_y <= center_y < end_y:
                    cell_words.append(word)

            cell_rows = group_words_into_rows(cell_words, y_tolerance=6)
            cell_lines = []

            for cell_row in cell_rows:
                line_text = " ".join(w["text"] for w in cell_row["words"])
                cell_lines.append(line_text)

            row_cells[col_idx] = join_table5_lines(cell_lines)

        final_rows.append(row_cells)

    df = pd.DataFrame(
        final_rows,
        columns=[
            "Agrifood system level",
            "Ensuring diversity",
            "Managing connectivity",
            "Managing risks",
        ],
    )

    return df



# 2025 TABLE 3 RECONSTRUCTION 


def table3_clean_line(text: str) -> str:
    text = normalize_inline(text)
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_table3_column_edges(bbox):
    x0, y0, x1, y1 = bbox
    width = x1 - x0

    return [
        x0,
        x0 + width * 0.17,
        x0 + width * 0.60,
        x1,
    ]


def detect_2025_table3_section_bands(words, bbox):
    x0, y0, x1, y1 = bbox
    rows = group_words_into_rows(words, y_tolerance=6)

    found = []

    for row in rows:
        row_text = table3_clean_line(" ".join(w["text"] for w in row["words"]))
        upper = row_text.upper()

        section = None

        if upper.strip() == "REGULATORY" or "REGULATORY" in upper:
            section = "REGULATORY"
        elif "INCENTIVE" in upper:
            section = "INCENTIVE-BASED"
        elif "CROSS" in upper and "COMPLIANCE" in upper:
            section = "CROSS-COMPLIANCE (CONDITIONALITY)"

        if section:
            found.append({
                "section": section,
                "y": row["y"],
            })

    unique = {}

    for item in found:
        if item["section"] not in unique:
            unique[item["section"]] = item["y"]

    order = [
        "REGULATORY",
        "INCENTIVE-BASED",
        "CROSS-COMPLIANCE (CONDITIONALITY)",
    ]

    ordered = []

    for section in order:
        if section in unique:
            ordered.append({
                "section": section,
                "start_y": unique[section],
            })

    if len(ordered) < 3:
        body_start = y0 + (y1 - y0) * 0.08
        body_end = y1
        band_height = (body_end - body_start) / 3

        ordered = []

        for i, section in enumerate(order):
            ordered.append({
                "section": section,
                "start_y": body_start + i * band_height,
            })

    bands = []

    for i, item in enumerate(ordered):
        start_y = item["start_y"]

        if i + 1 < len(ordered):
            end_y = ordered[i + 1]["start_y"] - 2
        else:
            end_y = y1

        bands.append({
            "section": item["section"],
            "start_y": start_y,
            "end_y": end_y,
        })

    return bands


def detect_2025_table3_row_bands(words, section_band, bbox):
    col_edges = get_table3_column_edges(bbox)

    label_x0 = col_edges[0]
    label_x1 = col_edges[1]

    section_start = section_band["start_y"]
    section_end = section_band["end_y"]

    label_words = []

    for word in words:
        wx0 = float(word["x0"])
        wx1 = float(word["x1"])
        top = float(word["top"])
        bottom = float(word["bottom"])

        center_x = (wx0 + wx1) / 2
        center_y = (top + bottom) / 2

        if label_x0 <= center_x < label_x1 and section_start <= center_y < section_end:
            label_words.append(word)

    label_rows = group_words_into_rows(label_words, y_tolerance=7)

    row_texts = []

    for row in label_rows:
        row_text = table3_clean_line(" ".join(w["text"] for w in row["words"]))

        if row_text:
            row_texts.append({
                "text": row_text,
                "upper": row_text.upper(),
                "y": row["y"],
            })

    starts = []

    for i, row in enumerate(row_texts):
        current = row["upper"]
        next_text = row_texts[i + 1]["upper"] if i + 1 < len(row_texts) else ""
        combined = f"{current} {next_text}"

        label = None

        if "DOES" in combined and "FARM" in combined:
            label = "Does farm size matter?"
        elif "MANAGEMENT" in combined and "BURDEN" in combined:
            label = "Management burden"
        elif "MONITORING" in combined and "REQUIREMENTS" in combined:
            label = "Monitoring requirements"
        elif "FINANCING" in combined and "NEEDS" in combined:
            label = "Financing needs"

        if label:
            starts.append({
                "label": label,
                "start_y": row["y"],
            })

    unique = {}

    for item in starts:
        if item["label"] not in unique:
            unique[item["label"]] = item["start_y"]

    order = [
        "Does farm size matter?",
        "Management burden",
        "Monitoring requirements",
        "Financing needs",
    ]

    ordered = []

    for label in order:
        if label in unique:
            ordered.append({
                "label": label,
                "start_y": unique[label],
            })

    if len(ordered) < 4:
        usable_start = section_start + 12
        usable_end = section_end
        row_height = (usable_end - usable_start) / 4

        ordered = []

        for i, label in enumerate(order):
            ordered.append({
                "label": label,
                "start_y": usable_start + i * row_height,
            })

    row_bands = []

    for i, item in enumerate(ordered):
        start_y = item["start_y"] - 8

        if i + 1 < len(ordered):
            end_y = ordered[i + 1]["start_y"] - 3
        else:
            end_y = section_end - 2

        row_bands.append({
            "label": item["label"],
            "start_y": start_y,
            "end_y": end_y,
        })

    return row_bands


def extract_text_from_table3_cell(words, x0, x1, y0, y1):
    cell_words = []

    y0 = y0 - 2
    y1 = y1 + 2

    for word in words:
        wx0 = float(word["x0"])
        wx1 = float(word["x1"])
        top = float(word["top"])
        bottom = float(word["bottom"])

        center_x = (wx0 + wx1) / 2
        center_y = (top + bottom) / 2

        if x0 <= center_x < x1 and y0 <= center_y < y1:
            cell_words.append(word)

    cell_rows = group_words_into_rows(cell_words, y_tolerance=6)
    lines = []

    for row in cell_rows:
        line_text = table3_clean_line(" ".join(w["text"] for w in row["words"]))

        if not line_text:
            continue

        upper = line_text.upper().strip()

        if upper in {
            "TABLE 3",
            "LAND MANAGEMENT",
            "LAND-USE CHANGE",
            "REGULATORY",
            "INCENTIVE-BASED",
            "CROSS-COMPLIANCE",
            "CROSS-COMPLIANCE (CONDITIONALITY)",
            "CONDITIONALITY",
        }:
            continue

        lines.append(line_text)

    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()

    text = text.replace("land- use", "land-use")
    text = text.replace("high- resolution", "high-resolution")
    text = text.replace("set- aside", "set-aside")
    text = text.replace("cofinanced", "co-financed")

    return text


def reconstruct_2025_table3(words, bbox):
    col_edges = get_table3_column_edges(bbox)
    section_bands = detect_2025_table3_section_bands(words, bbox)

    final_rows = []

    for section_band in section_bands:
        section = section_band["section"]
        row_bands = detect_2025_table3_row_bands(words, section_band, bbox)

        for row_band in row_bands:
            question = row_band["label"]

            y0 = row_band["start_y"]
            y1 = row_band["end_y"]

            land_management = extract_text_from_table3_cell(
                words=words,
                x0=col_edges[1],
                x1=col_edges[2],
                y0=y0,
                y1=y1,
            )

            land_use_change = extract_text_from_table3_cell(
                words=words,
                x0=col_edges[2],
                x1=col_edges[3],
                y0=y0,
                y1=y1,
            )

            final_rows.append([
                section,
                question,
                land_management,
                land_use_change,
            ])

    df = pd.DataFrame(
        final_rows,
        columns=[
            "Policy instrument",
            "Question / aspect",
            "Land management",
            "Land-use change",
        ],
    )

    df = df[
        df.apply(
            lambda row: any(str(cell).strip() for cell in row),
            axis=1,
        )
    ].reset_index(drop=True)

    return df



# GENERIC TABLE EXTRACTION


def score_generic_df(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0

    rows, cols = df.shape
    text = " ".join(df.astype(str).values.flatten()).upper()

    score = 0

    if rows >= 3:
        score += 2

    if cols >= 3:
        score += 2

    if "TABLE" not in text:
        score += 1

    important_terms = [
        "REGULATORY",
        "INCENTIVE",
        "CROSS-COMPLIANCE",
        "CONDITIONALITY",
        "LAND MANAGEMENT",
        "LAND-USE CHANGE",
        "MONITORING",
        "FINANCING",
    ]

    if any(term in text for term in important_terms):
        score += 3

    return score


def extract_generic_table_with_pdfplumber(pdf_path: str, page_number: int, crop_bbox):
    settings_list = [
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 5,
            "snap_tolerance": 4,
            "join_tolerance": 4,
            "edge_min_length": 20,
        },
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "intersection_tolerance": 5,
            "snap_tolerance": 4,
            "join_tolerance": 4,
            "min_words_vertical": 2,
            "min_words_horizontal": 1,
        },
    ]

    best_df = None
    best_score = -1

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        cropped = page.crop(tuple(crop_bbox))

        for settings in settings_list:
            try:
                tables = cropped.extract_tables(table_settings=settings) or []
            except Exception:
                tables = []

            for raw in tables:
                df = pd.DataFrame(raw).fillna("")

                for col in df.columns:
                    df[col] = df[col].apply(normalize_inline)

                df = df[
                    df.apply(
                        lambda row: any(str(cell).strip() for cell in row),
                        axis=1,
                    )
                ]

                df = df.loc[
                    :,
                    df.apply(
                        lambda col: any(str(cell).strip() for cell in col),
                        axis=0,
                    )
                ]

                df = df.reset_index(drop=True)

                score = score_generic_df(df)

                if score > best_score:
                    best_score = score
                    best_df = df

    return best_df, best_score



# TABLE DOCUMENT CREATION


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def dataframe_to_table_documents(
    df: pd.DataFrame,
    pdf_path: str,
    doc_id: str,
    file_name: str,
    page: int,
    table_number: str,
    caption: str,
):
    docs = []
    image_path = render_page_image(pdf_path, page - 1, zoom=1.8)

    md = dataframe_to_markdown(df)

    full_content = clean_text(f"""
[DOCUMENT: {file_name}]
[PAGE: {page}]
[SOURCE TYPE: STRUCTURED TABLE]
[TABLE: {table_number}]
[TABLE CAPTION: {caption}]

{md}
""")

    docs.append(
        Document(
            page_content=full_content,
            metadata={
                "doc_id": doc_id,
                "file_name": file_name,
                "source_pdf": pdf_path,
                "page": page,
                "page_index": page - 1,
                "image_path": image_path,
                "source_kind": "table",
                "table_number": table_number,
                "caption": caption,
                "chunk_type": "full_table",
            },
        )
    )

    for idx, row in df.iterrows():
        row_pairs = []

        for col in df.columns:
            value = normalize_inline(row[col])

            if value:
                row_pairs.append(f"{col}: {value}")

        if not row_pairs:
            continue

        row_content = clean_text(f"""
[DOCUMENT: {file_name}]
[PAGE: {page}]
[SOURCE TYPE: TABLE ROW]
[TABLE: {table_number}]
[TABLE CAPTION: {caption}]
[ROW: {idx + 1}]

""" + "\n".join(row_pairs))

        docs.append(
            Document(
                page_content=row_content,
                metadata={
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "source_pdf": pdf_path,
                    "page": page,
                    "page_index": page - 1,
                    "image_path": image_path,
                    "source_kind": "table_row",
                    "table_number": table_number,
                    "caption": caption,
                    "row_index": int(idx + 1),
                    "chunk_type": "table_row",
                },
            )
        )

    return docs



# STRUCTURED TABLE PROCESSING


def process_structured_tables(pdf_path: str, doc_id: str, file_name: str, table_captions):
    allow = STRUCTURED_TABLE_ALLOWLIST.get(file_name, set())
    ocr_table_rules = OCR_FUTURE_TABLES.get(file_name, {})

    table_docs = []
    structured_tables = []
    ocr_future_items = []

    for cap in table_captions:
        table_number = cap["number"]
        page_number = cap["page"]
        caption = cap["caption"]

        if table_number in ocr_table_rules:
            ocr_future_items.append({
                "item_type": "table",
                "number": table_number,
                "page": page_number,
                "caption": caption,
                "reason": ocr_table_rules[table_number],
            })
            continue

        if allow and table_number not in allow:
            ocr_future_items.append({
                "item_type": "table",
                "number": table_number,
                "page": page_number,
                "caption": caption,
                "reason": "Not in structured extraction allowlist for this PDF.",
            })
            continue

        page_width, page_height = get_page_size(pdf_path, page_number)
        lines = get_page_lines(pdf_path, page_number)
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
            words = extract_words_in_bbox(pdf_path, page_number, structured_bbox)
            df = reconstruct_2021_table2(words, structured_bbox)
            method = "custom_2021_table2_word_position"

        elif file_name == "2021_pdf.pdf" and table_number == "5":
            words = extract_words_in_bbox(pdf_path, page_number, structured_bbox)
            df = reconstruct_2021_table5(words, structured_bbox)
            method = "custom_2021_table5_section_band"

        elif file_name == "2025_pdf.pdf" and table_number == "3":
            words = extract_words_in_bbox(pdf_path, page_number, structured_bbox)
            df = reconstruct_2025_table3(words, structured_bbox)
            method = "custom_2025_table3_section_row_column"

        else:
            df, score = extract_generic_table_with_pdfplumber(
                pdf_path,
                page_number,
                structured_bbox,
            )
            method = f"pdfplumber_generic_score_{score}"

        if df is None or df.empty:
            ocr_future_items.append({
                "item_type": "table",
                "number": table_number,
                "page": page_number,
                "caption": caption,
                "reason": "Structured extraction failed or returned empty table.",
            })
            continue

        docs = dataframe_to_table_documents(
            df=df,
            pdf_path=pdf_path,
            doc_id=doc_id,
            file_name=file_name,
            page=page_number,
            table_number=table_number,
            caption=caption,
        )

        table_docs.extend(docs)

        structured_tables.append({
            "table_number": table_number,
            "page": page_number,
            "caption": caption,
            "method": method,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        })

    return table_docs, structured_tables, ocr_future_items


def process_future_figures(file_name: str, figure_captions):
    future_set = OCR_FUTURE_FIGURES.get(file_name, set())
    items = []

    for fig in figure_captions:
        number = fig["number"]

        if number in future_set or not future_set:
            items.append({
                "item_type": "figure",
                "number": number,
                "page": fig["page"],
                "caption": fig["caption"],
                "reason": "Figure/chart/diagram/map requires OCR or visual understanding. Not indexed in text-table-only mode.",
            })

    return items



# MAIN PARSER


def parse_pdf(pdf_path: str) -> List[Document]:
    pdf_path = str(pdf_path)
    p = Path(pdf_path)
    file_name = p.name
    doc_id = stable_doc_id(pdf_path)

    cache_file = PARSED_CACHE_DIR / f"{doc_id}_text_table_only.json"

    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)

        return [
            Document(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )
            for item in cached["documents"]
        ]

    pdf = fitz.open(pdf_path)
    total_pages = len(pdf)
    pdf.close()

    documents = []

    extraction_report = {
        "file_name": file_name,
        "doc_id": doc_id,
        "mode": "text_table_only_no_ocr",
        "pages": total_pages,
        "structured_tables": [],
        "ocr_future_items": [],
    }

    # 1. Extract normal page text and render citation page preview
    for page_index in range(total_pages):
        page_number = page_index + 1
        text = best_page_text(pdf_path, page_index)
        image_path = render_page_image(pdf_path, page_index, zoom=1.8)

        if len(text.split()) < 10:
            extraction_report["ocr_future_items"].append({
                "item_type": "page",
                "page": page_number,
                "reason": "Page has very little extractable text. May require OCR later.",
            })
            continue

        content = clean_text(f"""
[DOCUMENT: {file_name}]
[PAGE: {page_number}]
[SOURCE TYPE: PDF TEXT]

{text}
""")

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "source_pdf": pdf_path,
                    "page": page_number,
                    "page_index": page_index,
                    "image_path": image_path,
                    "source_kind": "text",
                    "chunk_type": "page_text",
                },
            )
        )

    # 2. Detect captions
    table_captions, figure_captions = detect_captions(pdf_path)

    # 3. Extract allowed structured tables
    table_docs, structured_tables, ocr_table_items = process_structured_tables(
        pdf_path=pdf_path,
        doc_id=doc_id,
        file_name=file_name,
        table_captions=table_captions,
    )

    documents.extend(table_docs)
    extraction_report["structured_tables"].extend(structured_tables)
    extraction_report["ocr_future_items"].extend(ocr_table_items)

    # 4. Put figures into future OCR bucket
    figure_items = process_future_figures(file_name, figure_captions)
    extraction_report["ocr_future_items"].extend(figure_items)

    # 5. Save OCR future manifest
    report_path = EXTRACTION_REPORT_DIR / f"{doc_id}_ocr_future_manifest.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(extraction_report, f, ensure_ascii=False, indent=2)

    # 6. Save parsed cache
    serializable = {
        "documents": [
            {
                "page_content": d.page_content,
                "metadata": d.metadata,
            }
            for d in documents
        ],
        "extraction_report": extraction_report,
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    return documents


def get_extraction_report(pdf_path: str):
    doc_id = stable_doc_id(pdf_path)
    cache_file = PARSED_CACHE_DIR / f"{doc_id}_text_table_only.json"

    if not cache_file.exists():
        return {}

    with open(cache_file, "r", encoding="utf-8") as f:
        cached = json.load(f)

    return cached.get("extraction_report", {})