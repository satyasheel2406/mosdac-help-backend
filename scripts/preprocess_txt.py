import os
import re

TXT_DIR = "../data/txt-files"
CLEANED_DIR = "../data/cleaned-txt"

# Ensure output directory exists
os.makedirs(CLEANED_DIR, exist_ok=True)

def clean_text(text):
    # Remove non-ASCII characters
    text = text.encode("ascii", "ignore").decode()

    # Replace multiple spaces/newlines with a single space or newline
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Keep paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)      # Collapse multiple spaces/tabs
    text = re.sub(r'\n{3,}', '\n\n', text)   # Max two newlines
    text = text.strip()

    return text

def preprocess_txt_files():
    for filename in os.listdir(TXT_DIR):
        if filename.endswith(".txt"):
            input_path = os.path.join(TXT_DIR, filename)
            output_path = os.path.join(CLEANED_DIR, filename)

            with open(input_path, "r", encoding="utf-8") as infile:
                raw_text = infile.read()

            cleaned = clean_text(raw_text)

            with open(output_path, "w", encoding="utf-8") as outfile:
                outfile.write(cleaned)

            print(f"[✔] Cleaned: {filename}")

if __name__ == "__main__":
    preprocess_txt_files()
