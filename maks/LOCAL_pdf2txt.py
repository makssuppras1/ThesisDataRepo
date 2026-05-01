### IMPORTS ###
import os
import pymupdf
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

### PATHS ###
# pdf_path = "Data/RAW_test"
# txt_path = "Data/TXT_test"
pdf_path = Path("/Volumes/NoaNoa/Theses_PDF_raw")
txt_path = "maks/data/thesis_txts"

# Speed tuning (optional env):
#   PDF2TXT_WORKERS   — process count (default: all CPUs, max 32)
#   PDF2TXT_SKIP_UP_TO_DATE=0 — force re-convert even if .txt is newer than .pdf
#   PDF2TXT_VERBOSE=1 — print every written file (noisy with many workers)

_WORKERS = min(32, int(os.environ.get("PDF2TXT_WORKERS", os.cpu_count() or 4)))
_SKIP_UP_TO_DATE = os.environ.get("PDF2TXT_SKIP_UP_TO_DATE", "1") != "0"
_VERBOSE = os.environ.get("PDF2TXT_VERBOSE", "") == "1"

# Plain-text extraction only (faster than default block/dict layout).
_TEXT_FLAGS = pymupdf.TEXTFLAGS_TEXT

### CHOOSE PDF FILES TO CONVERT ###
# Skip macOS AppleDouble sidecars (._name.pdf) — not real PDFs; PyMuPDF raises FileDataError.
list_pdf_files = sorted(
    f
    for f in os.listdir(pdf_path)
    if f.endswith(".pdf") and not f.startswith("._")
)
print(f"Found {len(list_pdf_files)} PDF files to convert (workers={_WORKERS}).")


def _convert_one_pdf(job: tuple[str, str, str, bool]) -> tuple[str, str | None]:
    """Worker: returns (status, detail). status in ok, skipped, unreadable, error."""
    pdf_dir, pdf_file, out_dir, skip_if_newer = job
    pdf_full = os.path.join(pdf_dir, pdf_file)
    out_path = os.path.join(out_dir, os.path.splitext(pdf_file)[0] + ".txt")

    if skip_if_newer and os.path.isfile(out_path):
        try:
            if os.path.getmtime(out_path) >= os.path.getmtime(pdf_full):
                return ("skipped", None)
        except OSError:
            pass

    try:
        doc = pymupdf.open(pdf_full)
    except pymupdf.FileDataError as e:
        return ("unreadable", f"{pdf_file}: {e}")

    try:
        parts: list[str] = []
        append = parts.append
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            append(f"==PAGE:{page_num + 1}==\n")
            append(page.get_text(flags=_TEXT_FLAGS))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("".join(parts))
        if _VERBOSE:
            print(f"Wrote: {out_path}")
        return ("ok", out_path)
    except Exception as e:
        return ("error", f"{pdf_file}: {e}")
    finally:
        doc.close()


### MAIN ###
if __name__ == "__main__":
    os.makedirs(txt_path, exist_ok=True)
    pdf_dir_s = str(pdf_path)
    jobs = [(pdf_dir_s, name, txt_path, _SKIP_UP_TO_DATE) for name in list_pdf_files]

    counts: Counter[str] = Counter()
    if _WORKERS <= 1:
        for job in jobs:
            status, detail = _convert_one_pdf(job)
            counts[status] += 1
            if status == "unreadable":
                print(f"Skip (unreadable): {detail}")
            elif status == "error":
                print(f"Error: {detail}")
            elif status == "ok" and not _VERBOSE:
                pass
    else:
        with ProcessPoolExecutor(max_workers=_WORKERS) as pool:
            futures = {pool.submit(_convert_one_pdf, job): job for job in jobs}
            for fut in as_completed(futures):
                status, detail = fut.result()
                counts[status] += 1
                if status == "unreadable":
                    print(f"Skip (unreadable): {detail}")
                elif status == "error":
                    print(f"Error: {detail}")

    print(
        "Done: "
        f"wrote {counts['ok']}, skipped up-to-date {counts['skipped']}, "
        f"unreadable {counts['unreadable']}, errors {counts['error']}"
    )
