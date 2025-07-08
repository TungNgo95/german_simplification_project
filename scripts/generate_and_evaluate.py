import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path
import evaluate
import json

# Config
checkpoint_path = "models/t5-gnats-clean/checkpoint-21"
test_file = "data/gnats/clean_test.csv"
prediction_file = "results/test_predictions.csv"
sari_file = "results/sari_score.json"

# Load model và tokenizer
tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)
model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)

# Load test set
test_df = pd.read_csv(test_file)

# Generate predictions
predictions = []
for idx, row in test_df.iterrows():
    source = row["source"]
    prompt = "translate German to simplified German: " + source
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=128)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(prediction)

# Save predictions
Path("results").mkdir(exist_ok=True)
test_df["prediction"] = predictions
test_df.to_csv(prediction_file, index=False)
print(f"✅ Predictions saved to {prediction_file}")

# Evaluate with SARI
sources = test_df["source"].fillna("").tolist()
references = [[ref] for ref in test_df["target"].fillna("").tolist()]
sari = evaluate.load("sari")
results = sari.compute(predictions=predictions, sources=sources, references=references)

# Save SARI scores
Path("results").mkdir(exist_ok=True)
Path(sari_file).write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"✅ SARI scores saved to {sari_file}")

# Print results
print("\n✅ Evaluation with SARI:")
for key, value in results.items():
    print(f"{key}: {value:.2f}")