import os
import fitz  # PyMuPDF

PDF_DIR = "backend/data/pdf-files/"
TXT_DIR = "backend/data/txt-files/"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def save_text_to_file(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def process_all_pdfs(pdf_folder, txt_folder):
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            txt_filename = filename.replace(".pdf", ".txt")
            txt_path = os.path.join(txt_folder, txt_filename)

            print(f"Extracting: {filename}")
            text = extract_text_from_pdf(pdf_path)
            save_text_to_file(text, txt_path)
            print(f"Saved to: {txt_path}")

if __name__ == "__main__":
    process_all_pdfs(PDF_DIR, TXT_DIR)
