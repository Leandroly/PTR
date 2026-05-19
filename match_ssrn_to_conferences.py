import argparse
import os
import re
import shutil
import time
from pathlib import Path

import pandas as pd
import pdfplumber
import pytesseract
import requests
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
from rapidfuzz import fuzz, process
from tqdm import tqdm


# ----------------------------
# CONFIG
# ----------------------------

DEFAULT_INPUT_XLSX = r"D:\H\ssrn\afa_papers_feb6.xlsx"
DEFAULT_OUTPUT_XLSX = r"D:\H\ssrn\ssrn_afa95_flags.xlsx"
DEFAULT_SHEET_NAME = 0

TITLE_COL = "title"
ABSTRACT_ID_COL = "abstract_id"

REQUEST_DELAY = 5
HEADERS = {"User-Agent": "Mozilla/5.0 (SSRN-AFA-WFA-matcher/1.0)"}

AFA_2018_2023_CUTOFF = 88
AFA_2024_2026_CUTOFF = 93
WFA_CUTOFF = 92

CACHE_DIR = Path("program_cache")
CACHE_DIR.mkdir(exist_ok=True)

AFA_URLS = {
    "afa2018": "https://afajof.org/2018-preliminary-program/",
    "afa2019": "https://afajof.org/2019-preliminary-program/",
    "afa2020": "https://afajof.org/2020-preliminary-program/",
    "afa2021": "https://afajof.org/2021-preliminary-program/",
    "afa2022": "https://afajof.org/2022-preliminary-program/",
    "afa2023": "https://afajof.org/2023-preliminary-program/",
    "afa2024": "https://afajof.org/management/full-program2024.html",
    "afa2025": "https://afajof.org/management/full-program2025.html",
    "afa2026": "https://afajof.org/management/full-program2026.html",
}

WFA_PDFS = {
    "wfa2018": "http://westernfinance.org/wp-content/uploads/2018.pdf",
    "wfa2019": "https://westernfinance.org/wp-content/uploads/2019.pdf",
    "wfa2020": "https://westernfinance.org/wp-content/uploads/2020.pdf",
    "wfa2021": "https://westernfinance.org/wp-content/uploads/2021.pdf.pdf",
    "wfa2022": "https://westernfinance.org/wp-content/uploads/2022.pdf",
    "wfa2023": "https://westernfinance.org/wp-content/uploads/2023.pdf",
    "wfa2024": "https://westernfinance.org/wp-content/uploads/2024.pdf",
    "wfa2025": "https://westernfinance.org/wp-content/uploads/2025_links.pdf",
}

AFA_BAD_WORDS = [
    "session",
    "chair",
    "discussant",
    "break",
    "lunch",
    "coffee",
    "reception",
    "welcome",
    "annual meeting",
    "preliminary program",
    "american finance association",
    "program committee",
    "poster session",
    "room",
    "hotel",
    "ballroom",
]


# ----------------------------
# TEXT NORMALIZATION
# ----------------------------

def clean_line(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u00a0", " ")
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r":(?=\w)", ": ", s)
    s = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_title(s: str) -> str:
    s = clean_line(s).lower()
    return re.sub(r"\s+", " ", s).strip()


# ----------------------------
# DOWNLOAD HELPERS
# ----------------------------

def fetch_url_text(url: str, cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")

    resp = requests.get(url, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    cache_path.write_text(resp.text, encoding="utf-8", errors="ignore")
    time.sleep(REQUEST_DELAY)
    return resp.text


def download_pdf(url: str, cache_path: Path) -> None:
    if cache_path.exists():
        return

    resp = requests.get(url, headers=HEADERS, timeout=180)
    resp.raise_for_status()
    ctype = (resp.headers.get("Content-Type") or "").lower()
    content = resp.content
    if "pdf" not in ctype and not content.startswith(b"%PDF"):
        raise RuntimeError(f"Did not receive a PDF for {url}. Content-Type={ctype}")

    cache_path.write_bytes(content)
    time.sleep(REQUEST_DELAY)


# ----------------------------
# AFA EXTRACTION
# ----------------------------

def is_noise_line(s: str) -> bool:
    low = s.lower()
    if re.match(r"^\d{1,2}:\d{2}", s):
        return True
    if any(w in low for w in AFA_BAD_WORDS):
        return True
    if len(s) < 12:
        return True
    if len(s.split()) < 4:
        return True
    if s.count(",") >= 3:
        return True
    return False


def extract_afa_titles_2018_2023(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    candidates = []
    for tagname in ["h3", "h4", "h5", "h6"]:
        for h in soup.find_all(tagname):
            t = clean_line(h.get_text(" ", strip=True))
            if t:
                candidates.append(t)

    for st in soup.find_all(["strong", "b"]):
        t = clean_line(st.get_text(" ", strip=True))
        if t:
            candidates.append(t)

    seen = set()
    out = []
    for t in candidates:
        if is_noise_line(t):
            continue
        nt = normalize_title(t)
        if nt and nt not in seen:
            seen.add(nt)
            out.append(t)
    return out


def extract_afa_titles_2024_plus(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    candidates = []
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip().lower()
        txt = clean_line(a.get_text(" ", strip=True))
        if not txt or is_noise_line(txt):
            continue
        if "viewp.php" in href or "paper" in href or "id=" in href:
            candidates.append(txt)

    seen = set()
    out = []
    for t in candidates:
        nt = normalize_title(t)
        if nt and nt not in seen:
            seen.add(nt)
            out.append(t)
    return out


# ----------------------------
# WFA EXTRACTION
# ----------------------------

def looks_like_author_line(s: str) -> bool:
    s = s.strip()
    if len(s) < 6 or len(s) > 180:
        return False

    low = s.lower()
    if any(
        k in low
        for k in [
            "university",
            "college",
            "school",
            "department",
            "institute",
            "federal reserve",
            "bank",
            "nber",
            "centre",
            "center",
        ]
    ):
        return True

    if s.count(",") >= 1 and not any(k in low for k in ["session", "chair", "discussant"]):
        return True

    if " and " in low and len(s.split()) >= 4:
        return True

    return False


def looks_like_title_line_wfa(s: str, year: int) -> bool:
    s = s.strip()
    if len(s) < 12 or len(s) > 180:
        return False

    low = s.lower()
    if any(
        k in low
        for k in [
            "session",
            "chair",
            "discussant",
            "break",
            "lunch",
            "coffee",
            "reception",
            "plenary",
            "welcome",
            "program",
            "agenda",
            "poster session",
            "roundtable",
            "keynote",
            "panel",
            "room",
            "ballroom",
            "meeting",
            "committee",
        ]
    ):
        return False

    if any(
        k in low
        for k in [
            "university",
            "college",
            "school of",
            "department",
            "institute",
            "federal reserve",
            "bank",
            "nber",
            "centre",
            "center",
        ]
    ):
        return False

    if s.count(",") >= 3:
        return False

    tmp = re.sub(r"^[\u2022\-\*\u00b7]\s*", "", s)
    tmp = re.sub(r"^\(?\d+\)?[\.\)]\s*", "", tmp)
    words = tmp.split()

    if year <= 2022:
        return len(words) >= 4

    if len(words) < 5:
        return False

    has_title_punct = any(ch in tmp for ch in [":", "-", "(", ")", "?", "%"])
    has_long_word = any(len(w) >= 9 for w in words)
    return has_title_punct or has_long_word


def extract_wfa_titles_from_pdf(
    pdf_path: Path,
    year: int,
    lookahead: int = 12,
    require_author: bool = True,
) -> list[str]:
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            for line in txt.split("\n"):
                line = clean_line(line)
                if line:
                    lines.append(line)

    titles = []
    for i, line in enumerate(lines):
        if not looks_like_title_line_wfa(line, year):
            continue

        authorish = False
        for j in range(1, lookahead + 1):
            if i + j >= len(lines):
                break
            nxt = lines[i + j]
            if looks_like_author_line(nxt):
                authorish = True
                break
            if looks_like_title_line_wfa(nxt, year) and j >= 2:
                break

        if authorish or not require_author:
            titles.append(line)

    return dedupe_titles(titles)


def ocr_pdf_to_lines(pdf_path: Path, poppler_path: str | None = None, dpi: int = 250) -> list[str]:
    lines = []
    images = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=poppler_path)

    for img in tqdm(images, desc=f"OCR {pdf_path.name}", unit="page"):
        text = pytesseract.image_to_string(img)
        for line in text.splitlines():
            line = clean_line(line)
            if line:
                lines.append(line)
    return lines


def extract_titles_from_lines_ocr(lines: list[str], year: int) -> list[str]:
    titles = []
    for i, line in enumerate(lines):
        if not looks_like_title_line_wfa(line, year):
            continue

        authorish = False
        for j in range(1, 18):
            if i + j >= len(lines):
                break
            if looks_like_author_line(lines[i + j]):
                authorish = True
                break
        if authorish:
            titles.append(line)

    return dedupe_titles(titles)


def dedupe_titles(titles: list[str]) -> list[str]:
    seen = set()
    out = []
    for t in titles:
        nt = normalize_title(t)
        if nt and nt not in seen:
            seen.add(nt)
            out.append(t)
    return out


# ----------------------------
# BUILD INDICES
# ----------------------------

def load_or_build_titles(key: str, builder_fn, force_rebuild: bool = False) -> list[str]:
    titles_path = CACHE_DIR / f"{key}_titles.txt"
    if titles_path.exists() and not force_rebuild:
        titles = [
            clean_line(x)
            for x in titles_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        ]
        return [t for t in titles if t]

    titles = builder_fn()
    titles_path.write_text("\n".join(titles), encoding="utf-8", errors="ignore")
    return titles


def build_afa_index(force_rebuild: bool = False) -> dict[str, list[str]]:
    index = {}
    for key, url in AFA_URLS.items():
        year = int(key.replace("afa", ""))
        html_path = CACHE_DIR / f"{key}.html"
        html = fetch_url_text(url, html_path)

        def builder() -> list[str]:
            if year <= 2023:
                return extract_afa_titles_2018_2023(html)
            return extract_afa_titles_2024_plus(html)

        titles = load_or_build_titles(key, builder, force_rebuild=force_rebuild)
        index[key] = [normalize_title(t) for t in titles]
        print(f"[AFA] {key}: {len(index[key])} extracted titles")
    return index


def build_wfa_index(
    force_rebuild: bool = False,
    poppler_path: str | None = None,
    tesseract_cmd: str | None = None,
) -> dict[str, list[str]]:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    index = {}
    for key, url in WFA_PDFS.items():
        pdf_path = CACHE_DIR / f"{key}.pdf"
        download_pdf(url, pdf_path)
        yr = int(key.replace("wfa", ""))

        def builder() -> list[str]:
            lookahead = 12 if yr <= 2022 else 4
            titles = extract_wfa_titles_from_pdf(
                pdf_path,
                year=yr,
                lookahead=lookahead,
                require_author=True,
            )

            if yr <= 2022 and len(titles) < 90:
                print(f"[WFA] {key}: only {len(titles)} titles via text extraction -> running OCR fallback")
                tesseract_exe = tesseract_cmd or shutil.which("tesseract")
                if not tesseract_exe:
                    print(f"[WFA] {key}: OCR skipped because tesseract is not configured")
                else:
                    try:
                        ocr_lines = ocr_pdf_to_lines(pdf_path, poppler_path=poppler_path, dpi=250)
                        titles = extract_titles_from_lines_ocr(ocr_lines, year=yr)
                        print(f"[WFA] {key}: OCR extracted {len(titles)} titles")
                    except Exception as exc:
                        print(f"[WFA] {key}: OCR failed ({exc}); keeping text-extracted titles")

            return titles

        titles = load_or_build_titles(key, builder, force_rebuild=force_rebuild)
        index[key] = [normalize_title(t) for t in titles]
        print(f"[WFA] {key}: {len(index[key])} extracted titles")
    return index


# ----------------------------
# MATCHING
# ----------------------------

def match_flags(df: pd.DataFrame, index: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    out["_norm_title"] = out[TITLE_COL].map(normalize_title)

    for year_key, choices in index.items():
        if not choices:
            out[year_key] = 0
            continue

        if year_key.startswith("afa"):
            yr = int(year_key[3:])
            cutoff = AFA_2018_2023_CUTOFF if yr <= 2023 else AFA_2024_2026_CUTOFF
        else:
            cutoff = WFA_CUTOFF

        flags = []
        for q in tqdm(out["_norm_title"].tolist(), desc=f"Matching {year_key}", unit="paper"):
            if not q or q == "nan":
                flags.append(0)
                continue
            flags.append(1 if has_acceptable_match(q, choices, cutoff) else 0)

        out[year_key] = flags

    return out.drop(columns=["_norm_title"])


def parse_sheet_name(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def acceptable_title_match(query: str, candidate: str, score: float) -> bool:
    """Reject high-scoring substring matches where titles are clearly different lengths."""
    if score >= 98:
        return True

    q_words = query.split()
    c_words = candidate.split()
    if not q_words or not c_words:
        return False

    word_ratio = min(len(q_words), len(c_words)) / max(len(q_words), len(c_words))
    char_ratio = min(len(query), len(candidate)) / max(len(query), len(candidate))
    return word_ratio >= 0.65 and char_ratio >= 0.60


def has_acceptable_match(query: str, choices: list[str], cutoff: int) -> bool:
    matches = process.extract(
        query,
        choices,
        scorer=fuzz.WRatio,
        score_cutoff=cutoff,
        limit=10,
    )
    return any(acceptable_title_match(query, candidate, score) for candidate, score, _ in matches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match SSRN paper titles to AFA and WFA conference program titles."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_XLSX, help="Input .xlsx path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_XLSX, help="Output .xlsx path.")
    parser.add_argument("--sheet", default=str(DEFAULT_SHEET_NAME), help="Sheet name or zero-based index.")
    parser.add_argument("--title-col", default=TITLE_COL, help="Input title column name.")
    parser.add_argument("--abstract-id-col", default=ABSTRACT_ID_COL, help="Input abstract id column name.")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR), help="Directory for downloaded programs and title caches.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild extracted title caches.")
    parser.add_argument("--poppler-path", default=os.environ.get("POPPLER_PATH"), help="Poppler bin path for OCR fallback.")
    parser.add_argument("--tesseract-cmd", default=os.environ.get("TESSERACT_CMD"), help="Tesseract executable path.")
    return parser.parse_args()


def main() -> None:
    global CACHE_DIR, TITLE_COL, ABSTRACT_ID_COL

    args = parse_args()
    CACHE_DIR = Path(args.cache_dir)
    CACHE_DIR.mkdir(exist_ok=True)
    TITLE_COL = args.title_col
    ABSTRACT_ID_COL = args.abstract_id_col

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input Excel file not found: {input_path}")

    df = pd.read_excel(input_path, sheet_name=parse_sheet_name(args.sheet), engine="openpyxl")
    for col in [ABSTRACT_ID_COL, TITLE_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Found columns: {list(df.columns)}")

    base = df[[ABSTRACT_ID_COL, TITLE_COL]].copy()

    afa_index = build_afa_index(force_rebuild=args.force_rebuild)
    wfa_index = build_wfa_index(
        force_rebuild=args.force_rebuild,
        poppler_path=args.poppler_path,
        tesseract_cmd=args.tesseract_cmd,
    )

    tmp = match_flags(base, afa_index)
    tmp = match_flags(tmp, wfa_index)

    if ABSTRACT_ID_COL != "abstract_id":
        tmp = tmp.rename(columns={ABSTRACT_ID_COL: "abstract_id"})

    ordered_cols = (
        ["abstract_id"]
        + [f"afa{y}" for y in range(2018, 2027)]
        + [f"wfa{y}" for y in range(2018, 2026)]
    )
    for c in ordered_cols:
        if c not in tmp.columns:
            tmp[c] = 0
    tmp = tmp[ordered_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.to_excel(output_path, index=False, engine="openpyxl")
    print(f"\nWrote: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
