import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path
import evaluate
import json

checkpoint_path = "models/t5-gnats-clean/checkpoint-1050"
test_file = "data/gnats/tatoeba/clean_test.csv"
prediction_file = "results/test_predictions.csv"
sari_file = "results/sari_score.json"

tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)
model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)

test_df = pd.read_csv(test_file)

predictions = []
for idx, row in test_df.iterrows():
    source = row["source"]
    prompt = "translate German to simplified German: " + source
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=128)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(prediction)

Path("results").mkdir(exist_ok=True)
test_df["prediction"] = predictions
test_df.to_csv(prediction_file, index=False)
print(f"✅ Predictions saved to {prediction_file}")

sources = test_df["source"].fillna("").tolist()
references = [[ref] for ref in test_df["target"].fillna("").tolist()]
sari = evaluate.load("sari")
results = sari.compute(predictions=predictions, sources=sources, references=references)

Path("results").mkdir(exist_ok=True)
Path(sari_file).write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"SARI scores saved to {sari_file}")

print("\nEvaluation with SARI:")
for key, value in results.items():
    print(f"{key}: {value:.2f}")