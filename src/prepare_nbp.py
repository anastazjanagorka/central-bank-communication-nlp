
from pathlib import Path
import re

import pymupdf
import pandas as pd

# 1. FILE PATHS

DATA_DIR = Path("data/nbp")
OUTPUT_FILE = Path("data/nbp_communications.csv")

pdf_files = sorted(DATA_DIR.glob("*.pdf"))

print("Number of PDF files:", len(pdf_files))

if not pdf_files:
    raise FileNotFoundError("No PDF files found in data/nbp.")

# 2. PDF TEXT EXTRACTION

for pdf_file in pdf_files:
    with pymupdf.open(pdf_file) as document:
        full_text = ""

        for page in document:
            full_text += page.get_text()

    txt_file = pdf_file.with_suffix(".txt")
    txt_file.write_text(full_text, encoding="utf-8")

print("PDF-to-TXT conversion completed.")

# 3. DATASET CREATION

data = []

for pdf_file in pdf_files:
    txt_file = pdf_file.with_suffix(".txt")
    text = txt_file.read_text(encoding="utf-8")

    data.append({
        "filename": txt_file.name,
        "text": text
    })

df = pd.DataFrame(data)

print("Dataset shape:", df.shape)

# 4. INITIAL DATA QUALITY CHECKS

duplicate_count = df["text"].duplicated().sum()
empty_count = df["text"].str.strip().eq("").sum()

print("Duplicate documents:", duplicate_count)
print("Empty documents:", empty_count)

# 5. TEXT CLEANING

def clean_text(text):
    """
    Remove technical separators and normalize whitespace.
    Preserve the original text in a separate column.
    """
    text = re.sub(r"={2,}", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["clean_text"] = df["text"].apply(clean_text)

# 6. REMOVE TECHNICAL HEADERS

df["clean_text"] = df["clean_text"].str.replace(
    r"^.*?(?=Warszawa,\s*\d{1,2}\s)",
    "",
    regex=True
)

# 7. CHECK COMMUNICATION TITLE OCCURRENCES

phrase = "INFORMACJA PO POSIEDZENIU"

title_counts = df["clean_text"].str.count(
    phrase,
    flags=re.IGNORECASE
)

print("\nTitle occurrence distribution:")
print(title_counts.value_counts().sort_index())

# 8. REMOVE STANDARD COMMUNICATION TITLES

df["clean_text"] = df["clean_text"].str.replace(
    r"INFORMACJA PO POSIEDZENIU RADY POLITYKI PIENIĘ[Żś]NEJ\s+w dni(?:u|ach)",
    "",
    flags=re.IGNORECASE,
    regex=True
)

# 9. FINAL WHITESPACE NORMALIZATION

df["clean_text"] = df["clean_text"].str.replace(
    r"\s+", " ", regex=True
).str.strip()

# 10. FINAL DATA QUALITY CHECKS

remaining_titles = df["clean_text"].str.contains(
    "INFORMACJA PO POSIEDZENIU",
    case=False,
    na=False
)

remaining_count = remaining_titles.sum()
empty_clean_count = df["clean_text"].eq("").sum()

print("\nFinal quality checks:")
print("Remaining communication titles:", remaining_count)
print("Empty cleaned documents:", empty_clean_count)
print("Number of cleaned documents:", len(df))

if remaining_count > 0:
    print("\nDocuments requiring additional cleaning:")
    print(df.loc[remaining_titles, "filename"].to_string(index=False))

# 11. TEXT PREVIEW

print("\nFirst cleaned communication:")
print(df["clean_text"].iloc[0][:300])

# 12. SAVE DATASET

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nDataset saved successfully:", OUTPUT_FILE)
