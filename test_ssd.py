import pymupdf
import os
from pathlib import Path

### PATHS ###
single_pdf = Path("/Volumes/NoaNoa/Theses_PDF_raw/2497547039.pdf")
txt_path = "TXT_handin_test"

### FUNCTIONS ###
def pdf_to_txt(pdf_file: Path, txt_dir: str) -> str:
    os.makedirs(txt_dir, exist_ok=True)
    pdf_document = pymupdf.open(pdf_file)
    extracted_text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        extracted_text += f"==PAGE:{page_num + 1}==\n"
        extracted_text += page.get_text()
    out_path = os.path.join(txt_dir, pdf_file.stem + ".txt")
    with open(out_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(extracted_text)
    print(f"Wrote: {out_path}")
    return out_path

### MAIN ###
if __name__ == "__main__":
    if not single_pdf.is_file():
        raise SystemExit(f"PDF not found: {single_pdf}")
    pdf_to_txt(single_pdf, txt_path)
