# scripts/prepare_gnats_data.py
import os
import pandas as pd

def load_and_combine(source_path, target_path, out_path):
    with open(source_path, "r", encoding="utf-8") as f:
        source_lines = [line.strip() for line in f if line.strip()]
    with open(target_path, "r", encoding="utf-8") as f:
        target_lines = [line.strip() for line in f if line.strip()]

    assert len(source_lines) == len(target_lines), f"Length mismatch between {source_path} and {target_path}"

    df = pd.DataFrame({
        "source": source_lines,
        "target": target_lines
    })
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path} ({len(df)} rows)")

def main():
    base = "data/gnats"
    for split in ["train", "val", "test"]:
        source_file = os.path.join(base, f"{split}-source.txt")
        target_file = os.path.join(base, f"{split}-target.txt")
        output_csv = os.path.join(base, f"{split}.csv")

        if os.path.exists(source_file) and os.path.exists(target_file):
            load_and_combine(source_file, target_file, output_csv)
        else: 
            print(f"Skipping {split}, files not found.")

if __name__ == "__main__":
    main()