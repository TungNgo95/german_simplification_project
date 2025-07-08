import pandas as pd
import re
import os

def clean_text(text, lowercase=True):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces between words
    text = re.sub(r'[^\w\s.,!?äöüÄÖÜß-]', '', text)  # keep only basic German characters + punctuation
    if lowercase:
        text = text.lower()
    return text

def preprocess_csv(input_path, output_path, lowercase=True):
    print(f"Processing: {input_path}")
    df = pd.read_csv(input_path)

    # Assume "source" and "target" columns in the CSV
    df['source'] = df['source'].astype(str).apply(lambda x: clean_text(x, lowercase))
    df['target'] = df['target'].astype(str).apply(lambda x: clean_text(x, lowercase))

    df.to_csv(output_path, index=False)
    print(f"Saved processed file to: {output_path}")

if __name__ == "__main__":
    base_dir = "data/gnats"
    files = ["train.csv", "val.csv", "test.csv"]

    for file in files:
        input_file = os.path.join(base_dir, file)
        output_file = os.path.join(base_dir, f"clean_{file}")
        preprocess_csv(input_file, output_file, lowercase=True)