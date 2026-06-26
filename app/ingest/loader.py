import csv
import io
import re
from pathlib import Path

from pypdf import PdfReader


def load_text_from_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    if suffix == ".csv":
        return _csv_to_text(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _csv_to_text(file_path: Path) -> str:
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(raw))
    rows = list(reader)
    if not rows:
        return ""
    header = rows[0]
    lines = [f"Columns: {', '.join(header)}"]
    for row in rows[1:]:
        pairs = []
        for col, val in zip(header, row):
            if val.strip():
                pairs.append(f"{col}: {val.strip()}")
        if pairs:
            lines.append(" | ".join(pairs))
    return "\n".join(lines)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + chunk_size - overlap, start + 1)
    return chunks
