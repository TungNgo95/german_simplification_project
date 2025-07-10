import random
import pandas as pd
from tatoebatools import ParallelCorpus
import os

source_lang = "deu"
target_lang = "deu"
num_pairs = 2000  # Số lượng cặp câu muốn lấy
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

print("Lade Datensatz von Tatoeba...")
pairs = list(ParallelCorpus(source_lang, target_lang))
print(f"Gesamtanzahl verfügbarer Paare: {len(pairs)}")

random.shuffle(pairs)
selected_pairs = pairs[:num_pairs]

num_train = int(train_ratio * num_pairs)
num_val = int(val_ratio * num_pairs)

train = selected_pairs[:num_train]
val = selected_pairs[num_train:num_train + num_val]
test = selected_pairs[num_train + num_val:]

output_dir = "data/gnats/tatoeba"
os.makedirs(output_dir, exist_ok=True)

splits = {
    "clean_train.csv": train,
    "clean_val.csv": val,
    "clean_test.csv": test
}

for filename, data in splits.items():
    filepath = os.path.join(output_dir, filename)
    df = pd.DataFrame(
        [(s.text, t.text) for s, t in data],
        columns=["source", "target"]
    )
    df.to_csv(filepath, index=False)
    print(f"{filepath} gespeichert mit {len(data)} Satzpaaren.")

print("Dataset erfolgreich generiert und gespeichert!")