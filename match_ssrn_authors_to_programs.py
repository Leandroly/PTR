import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process


PROGRAM_KEYS = [f"afa{y}" for y in range(2018, 2027)] + [f"wfa{y}" for y in range(2018, 2026)]

AFA_OLD_YEARS = range(2018, 2024)
AFA_NEW_YEARS = range(2024, 2027)
WFA_YEARS = range(2018, 2026)

NAME_MATCH_CUTOFF = 92

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

WFA_URLS = {
    "wfa2018": "http://westernfinance.org/wp-content/uploads/2018.pdf",
    "wfa2019": "https://westernfinance.org/wp-content/uploads/2019.pdf",
    "wfa2020": "https://westernfinance.org/wp-content/uploads/2020.pdf",
    "wfa2021": "https://westernfinance.org/wp-content/uploads/2021.pdf.pdf",
    "wfa2022": "https://westernfinance.org/wp-content/uploads/2022.pdf",
    "wfa2023": "https://westernfinance.org/wp-content/uploads/2023.pdf",
    "wfa2024": "https://westernfinance.org/wp-content/uploads/2024.pdf",
    "wfa2025": "https://westernfinance.org/wp-content/uploads/2025_links.pdf",
}

AFA_MANAGEMENT_BASE = "https://afajof.org/management/"

AFFILIATION_MARKERS = [
    "university",
    "college",
    "school",
    "institute",
    "federal reserve",
    "bank",
    "nber",
    "centre",
    "center",
    "business",
    "management",
    "economics",
    "finance",
    "insead",
]

STOP_WORDS = {
    "session",
    "chair",
    "chairs",
    "discussant",
    "discussants",
    "panelist",
    "panelists",
    "speaker",
    "speakers",
    "location",
    "program",
    "details",
    "room",
    "ballroom",
    "meeting",
    "break",
    "lunch",
    "coffee",
    "reception",
    "award",
    "awards",
}


def clean_line(value: str) -> str:
    value = "" if value is None else str(value)
    value = value.replace("\u00a0", " ")
    value = value.replace("\u2019", "'").replace("\u2018", "'")
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    value = value.replace("\u2013", "-").replace("\u2014", "-")
    value = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def split_camel_name(value: str) -> str:
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", value)
    value = re.sub(r"(?<=[A-Za-z])(?=[A-Z]\.)", " ", value)
    return value


def normalize_name(value: str) -> str:
    value = clean_line(value)
    value = split_camel_name(value)
    value = strip_accents(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", " ", value)
    value = re.sub(r"[^a-z\s'-]", " ", value)
    value = value.replace("-", " ")
    value = value.replace("'", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def name_last_token(value: str) -> str:
    parts = normalize_name(value).split()
    return parts[-1] if parts else ""


def plausible_person_name(value: str) -> bool:
    name = normalize_name(value)
    parts = name.split()
    if len(parts) < 2 or len(parts) > 6:
        return False
    low = name.lower()
    if any(word in low.split() for word in STOP_WORDS):
        return False
    if any(ch.isdigit() for ch in value):
        return False
    return True


def extract_name_before_affiliation(line: str) -> str | None:
    line = clean_line(line)
    if not line:
        return None
    low = line.lower()
    if low.rstrip(":") in STOP_WORDS or low.startswith(("discussant", "chair", "session chair")):
        return None

    if ";" in line:
        candidate = line.split(";", 1)[0]
    elif "," in line:
        candidate = line.split(",", 1)[0]
    else:
        return None

    candidate = clean_line(candidate)
    candidate = split_camel_name(candidate)
    return candidate if plausible_person_name(candidate) else None


def title_norm(value: str) -> str:
    value = strip_accents(clean_line(value)).lower()
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value


def load_title_set(cache_dir: Path, key: str) -> set[str]:
    path = cache_dir / f"{key}_titles.txt"
    if not path.exists():
        return set()
    return {title_norm(x) for x in path.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()}


def source_link_for_program(program: str) -> str:
    if program.startswith("afa"):
        return AFA_URLS.get(program, "")
    return WFA_URLS.get(program, "")


def make_afa_detail_link(href: str) -> str:
    href = clean_line(href)
    if href.lower().startswith("http"):
        return href
    return AFA_MANAGEMENT_BASE + href.lstrip("/")


def add_person(
    rows: list[dict],
    program: str,
    name: str,
    paper_title: str = "",
    role: str = "author_or_presenter",
    source_link: str = "",
) -> None:
    name = clean_line(split_camel_name(name))
    if not plausible_person_name(name):
        return
    rows.append(
        {
            "program": program,
            "paper_title": clean_line(paper_title),
            "name": name,
            "normalized_name": normalize_name(name),
            "role": role,
            "source_link": source_link or source_link_for_program(program),
        }
    )


def extract_afa_old(cache_dir: Path, key: str) -> list[dict]:
    html_path = cache_dir / f"{key}.html"
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    lines = [clean_line(x) for x in soup.get_text("\n", strip=True).splitlines()]
    lines = [x for x in lines if x]

    rows = []
    skip_next_discussant = False
    next_presenter = False
    current_title = ""
    title_set = load_title_set(cache_dir, key)

    for line in lines:
        low = line.lower().rstrip(":")
        if title_norm(line) in title_set:
            current_title = line
            next_presenter = False
            skip_next_discussant = False
            continue
        if low == "presented by":
            next_presenter = True
            skip_next_discussant = False
            continue
        if low == "discussant":
            skip_next_discussant = True
            next_presenter = False
            continue

        name = extract_name_before_affiliation(line)
        if not name:
            continue
        if skip_next_discussant:
            skip_next_discussant = False
            continue
        if next_presenter:
            add_person(rows, key, name, current_title, "presenter", source_link_for_program(key))
            next_presenter = False
        elif ";" in line:
            add_person(rows, key, name, current_title, "author", source_link_for_program(key))

    return rows


def extract_afa_new(cache_dir: Path, key: str) -> list[dict]:
    html_path = cache_dir / f"{key}.html"
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    rows = []
    for block in soup.find_all("div", class_="col-lg-12"):
        link = block.find("a", href=lambda h: h and "viewp.php" in h.lower())
        if not link:
            continue
        paper_title = clean_line(link.get_text(" ", strip=True))
        source_link = make_afa_detail_link(link.get("href") or "")
        for strong in block.find_all("strong"):
            add_person(rows, key, strong.get_text(" ", strip=True), paper_title, "author_or_presenter", source_link)
    return rows


def extract_pdf_lines(pdf_path: Path) -> list[str]:
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                line = clean_line(line)
                if line:
                    lines.append(line)
    return lines


def looks_like_wfa_session_or_section(line: str) -> bool:
    low = line.lower()
    if low in {"program details", "best paper awards", "discussants:"}:
        return True
    if re.match(r"^[a-z]+day,\s+[a-z]+\s+\d{1,2},\s+\d{4}", low):
        return True
    if any(word in low for word in ["ballroom", "room", "salon", "pavilion", "seacliff", "coral", "primrose"]):
        return True
    return False


def extract_wfa(cache_dir: Path, key: str) -> list[dict]:
    pdf_path = cache_dir / f"{key}.pdf"
    lines = extract_pdf_lines(pdf_path)
    title_set = load_title_set(cache_dir, key)

    rows = []
    in_program_details = False
    in_discussants = False
    current_title = ""

    for line in lines:
        low = line.lower().strip()
        if low == "program details":
            in_program_details = True
            continue
        if not in_program_details:
            continue
        if low.startswith("discussants"):
            in_discussants = True
            continue

        name = extract_name_before_affiliation(line)
        is_title = title_norm(line) in title_set
        if is_title:
            current_title = line
            continue

        if name:
            if not in_discussants:
                add_person(rows, key, name, current_title, "author_or_presenter", source_link_for_program(key))
            continue

        # WFA discussants appear as a block at the end of each session. The block
        # ends when the next session/header line appears, which normally has no
        # affiliation comma and is therefore not a name line.
        if in_discussants and not low.startswith("discussants"):
            in_discussants = False
        if looks_like_wfa_session_or_section(line):
            current_title = ""

    return rows


def build_program_people(cache_dir: Path) -> pd.DataFrame:
    rows = []
    for year in AFA_OLD_YEARS:
        rows.extend(extract_afa_old(cache_dir, f"afa{year}"))
    for year in AFA_NEW_YEARS:
        rows.extend(extract_afa_new(cache_dir, f"afa{year}"))
    for year in WFA_YEARS:
        rows.extend(extract_wfa(cache_dir, f"wfa{year}"))

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["program", "paper_title", "name", "normalized_name", "role", "source_link"])
    return df.drop_duplicates(["program", "paper_title", "normalized_name", "role", "source_link"]).reset_index(drop=True)


def acceptable_name_match(query_norm: str, candidate_norm: str, score: float) -> bool:
    if score < NAME_MATCH_CUTOFF:
        return False
    q_parts = query_norm.split()
    c_parts = candidate_norm.split()
    if len(q_parts) < 2 or len(c_parts) < 2:
        return False
    if q_parts[-1] == c_parts[-1]:
        return True
    return score >= 97


def match_authors(input_df: pd.DataFrame, people_df: pd.DataFrame) -> pd.DataFrame:
    out = input_df.copy()
    out["normalized_name"] = out["name"].map(normalize_name)

    by_program = {
        program: people_df[people_df["program"].eq(program)].copy()
        for program in PROGRAM_KEYS
    }

    matched_programs_col = []
    matched_names_col = []
    matched_links_col = []

    for program in PROGRAM_KEYS:
        out[program] = 0

    for idx, query_norm in enumerate(out["normalized_name"].tolist()):
        matched_programs = []
        matched_names = []
        matched_links = []
        if not query_norm:
            matched_programs_col.append("")
            matched_names_col.append("")
            matched_links_col.append("")
            continue

        for program, pdf in by_program.items():
            if pdf.empty:
                continue
            choices = pdf["normalized_name"].tolist()
            hits = process.extract(
                query_norm,
                choices,
                scorer=fuzz.WRatio,
                score_cutoff=NAME_MATCH_CUTOFF,
                limit=5,
            )
            accepted = []
            accepted_links = []
            for candidate_norm, score, match_idx in hits:
                if acceptable_name_match(query_norm, candidate_norm, score):
                    row = pdf.iloc[match_idx]
                    accepted.append(f"{row['name']} [{score:.0f}]")
                    link = clean_line(row.get("source_link", ""))
                    if link:
                        accepted_links.append(link)

            if accepted:
                out.at[idx, program] = 1
                matched_programs.append(program)
                matched_names.append(f"{program}: " + "; ".join(dict.fromkeys(accepted)))
                if accepted_links:
                    matched_links_col_value = "; ".join(dict.fromkeys(accepted_links))
                    matched_links.append(f"{program}: {matched_links_col_value}")

        matched_programs_col.append("; ".join(matched_programs))
        matched_names_col.append(" | ".join(matched_names))
        matched_links_col.append(" | ".join(matched_links))

    out["matched_programs"] = matched_programs_col
    out["matched_names"] = matched_names_col
    out["matched_links"] = matched_links_col
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match SSRN author names to AFA/WFA program authors and presenters.")
    parser.add_argument("--input", default="data/ssrn_id_list_may18.xlsx")
    parser.add_argument("--cache-dir", default="data/fwptr")
    parser.add_argument("--output", default="data/ssrn_author_program_matches.xlsx")
    parser.add_argument("--people-output", default="data/extracted_program_people.xlsx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    cache_dir = Path(args.cache_dir)

    input_df = pd.read_excel(input_path, engine="openpyxl")
    for col in ["name", "ssrn_id"]:
        if col not in input_df.columns:
            raise ValueError(f"Missing required column '{col}'. Found columns: {list(input_df.columns)}")

    people_df = build_program_people(cache_dir)
    people_df.to_excel(args.people_output, index=False, engine="openpyxl")
    print(f"Extracted {len(people_df)} program people rows -> {args.people_output}")
    print(people_df.groupby("program").size().to_string())

    output_df = match_authors(input_df[["ssrn_id", "name"]].copy(), people_df)
    ordered_cols = ["ssrn_id", "name", "normalized_name"] + PROGRAM_KEYS + ["matched_programs", "matched_names", "matched_links"]
    output_df = output_df[ordered_cols]
    output_df.to_excel(args.output, index=False, engine="openpyxl")
    print(f"Wrote {len(output_df)} matched author rows -> {args.output}")


if __name__ == "__main__":
    main()
