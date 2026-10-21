
from pathlib import Path
import re

import pandas as pd
import pymupdf


# ============================================================
# 1. FILE PATHS
# ============================================================

DATA_DIR = Path("data/nbp")
OUTPUT_FILE = Path("data/nbp_communications.csv")

pdf_files = sorted(DATA_DIR.glob("*.pdf"))

print("Number of PDF files:", len(pdf_files))

if not pdf_files:
    raise FileNotFoundError("No PDF files found in data/nbp.")


# ============================================================
# 2. PDF TEXT EXTRACTION
# ============================================================

data = []

for pdf_file in pdf_files:
    with pymupdf.open(pdf_file) as document:
        full_text = "\n".join(
            page.get_text() for page in document
        )

    txt_file = pdf_file.with_suffix(".txt")
    txt_file.write_text(full_text, encoding="utf-8")

    data.append({
        "filename": txt_file.name,
        "text": full_text
    })

print("PDF-to-TXT conversion completed.")


# ============================================================
# 3. DATASET CREATION
# ============================================================

df = pd.DataFrame(data)

print("Dataset shape:", df.shape)


# ============================================================
# 4. INITIAL DATA QUALITY CHECKS
# ============================================================

duplicate_count = df["text"].duplicated().sum()
empty_count = df["text"].str.strip().eq("").sum()

print("\nInitial quality checks:")
print("Duplicate documents:", duplicate_count)
print("Empty documents:", empty_count)


# ============================================================
# 5. TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Remove technical separators and normalize whitespace.
    Preserve original PDF text in a separate column.
    """
    text = re.sub(r"={2,}", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["clean_text"] = df["text"].apply(clean_text)


# ============================================================
# 6. REMOVE TECHNICAL HEADERS
# ============================================================

df["clean_text"] = df["clean_text"].str.replace(
    r"^.*?(?=Warszawa,\s*\d{1,2}\s)",
    "",
    regex=True
)


# ============================================================
# 7. CHECK COMMUNICATION TITLE OCCURRENCES
# ============================================================

phrase = "INFORMACJA PO POSIEDZENIU"

title_counts = df["clean_text"].str.count(
    phrase,
    flags=re.IGNORECASE
)

print("\nTitle occurrence distribution:")
print(title_counts.value_counts().sort_index())


# ============================================================
# 8. REMOVE STANDARD COMMUNICATION TITLES
# ============================================================

df["clean_text"] = df["clean_text"].str.replace(
    r"INFORMACJA PO POSIEDZENIU RADY POLITYKI PIENIĘ[Żś]NEJ\s+w dni(?:u|ach)",
    "",
    flags=re.IGNORECASE,
    regex=True
)


# ============================================================
# 9. FINAL WHITESPACE NORMALIZATION
# ============================================================

df["clean_text"] = df["clean_text"].str.replace(
    r"\s+", " ", regex=True
).str.strip()


# ============================================================
# 10. TEXT QUALITY CHECKS
# ============================================================

remaining_titles = df["clean_text"].str.contains(
    phrase,
    case=False,
    na=False
)

remaining_count = remaining_titles.sum()
empty_clean_count = df["clean_text"].eq("").sum()

print("\nText quality checks:")
print("Remaining communication titles:", remaining_count)
print("Empty cleaned documents:", empty_clean_count)

if remaining_count > 0:
    print("\nDocuments requiring additional cleaning:")
    print(
        df.loc[remaining_titles, "filename"]
        .to_string(index=False)
    )


# ============================================================
# 11. EXTRACT COMMUNICATION DATES
# ============================================================

date_pattern = (
    r"Warszawa,\s*(\d{1,2}\s+"
    r"[a-ząćęłńóśźż]+\s+\d{4})"
)

df["date_raw"] = df["clean_text"].str.extract(
    date_pattern,
    flags=re.IGNORECASE
)

print("\nDocuments with missing raw dates:",
      df["date_raw"].isna().sum())


# ============================================================
# 12. CONVERT POLISH DATES
# ============================================================

MONTHS = {
    "stycznia": "01",
    "lutego": "02",
    "marca": "03",
    "kwietnia": "04",
    "maja": "05",
    "czerwca": "06",
    "lipca": "07",
    "sierpnia": "08",
    "września": "09",
    "października": "10",
    "listopada": "11",
    "grudnia": "12"
}

date_parts = df["date_raw"].str.split(expand=True)

date_strings = (
    date_parts[2] + "-" +
    date_parts[1].str.lower().map(MONTHS) + "-" +
    date_parts[0].str.zfill(2)
)

df["date"] = pd.to_datetime(
    date_strings,
    format="%Y-%m-%d",
    errors="coerce"
)

invalid_dates = df["date"].isna().sum()

print("Invalid dates:", invalid_dates)

if invalid_dates > 0:
    print("\nDocuments requiring date verification:")
    print(
        df.loc[df["date"].isna(),
               ["filename", "date_raw"]]
        .to_string(index=False)
    )


# ============================================================
# 13. EXTRACT YEAR AND MONTH FROM FILENAMES
# ============================================================

df["year"] = pd.to_numeric(
    df["filename"].str.extract(
        r"nbp_(\d{4})_", expand=False
    ),
    errors="coerce"
)

df["month"] = pd.to_numeric(
    df["filename"].str.extract(
        r"nbp_\d{4}_(\d{2})", expand=False
    ),
    errors="coerce"
)

print("\nMissing years:", df["year"].isna().sum())
print("Missing months:", df["month"].isna().sum())


# ============================================================
# 14. VERIFY DATE CONSISTENCY
# ============================================================

date_mismatch = (
    df["date"].notna()
    & (
        df["date"].dt.year.ne(df["year"])
        | df["date"].dt.month.ne(df["month"])
    )
)

print("\nDate mismatches:", date_mismatch.sum())

if date_mismatch.any():
    print(
        df.loc[date_mismatch,
               ["filename", "date_raw", "year", "month"]]
        .to_string(index=False)
    )


# ============================================================
# 15. SORT COMMUNICATIONS CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    by=["date", "filename"],
    na_position="last"
).reset_index(drop=True)

print("\nFirst five communications:")
print(df[["filename", "date"]].head())

print("\nLast five communications:")
print(df[["filename", "date"]].tail())


# ============================================================
# 16. FINAL DATA QUALITY REPORT
# ============================================================

print("\nFINAL DATA QUALITY REPORT")
print("-------------------------")
print("Total documents:", len(df))
print("Duplicate documents:", duplicate_count)
print("Empty original documents:", empty_count)
print("Empty cleaned documents:", empty_clean_count)
print("Remaining titles:", remaining_count)
print("Missing dates:", invalid_dates)
print("Date mismatches:", date_mismatch.sum())
print("Duplicate dates:", df["date"].dropna().duplicated().sum())

print("\nFirst cleaned communication:")
print(df["clean_text"].iloc[0][:300])


# ============================================================
# 17. SAVE DATASET
# ============================================================

# Stop if critical quality checks fail.
if (
    duplicate_count > 0
    or empty_count > 0
    or empty_clean_count > 0
    or remaining_count > 0
    or invalid_dates > 0
    or df["year"].isna().any()
    or df["month"].isna().any()
    or date_mismatch.any()
    or df["date"].dropna().duplicated().any()
):
    raise ValueError(
        "Data quality checks failed. Review the report "
        "before saving the dataset."
    )

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
    date_format="%Y-%m-%d"
)

print("\nDataset saved successfully:", OUTPUT_FILE)
print("Final dataset shape:", df.shape)
