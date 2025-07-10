import pandas as pd
import evaluate
import json
from pathlib import Path
import sys

input_path = sys.argv[1] if len(sys.argv) > 1 else "results/test_predictions.csv"
df = pd.read_csv(input_path)

sources = df["source"].fillna("").tolist()
predictions = df["prediction"].fillna("").tolist()
references = [[ref] for ref in df["target"].fillna("").tolist()]  # for SARI
flat_references = [ref[0] for ref in references]  # for BLEU, ROUGE, BERTScore

# Load metrics
metrics = {
    "sari": evaluate.load("sari"),
    "bleu": evaluate.load("bleu"),
    "rouge": evaluate.load("rouge"),
    "bertscore": evaluate.load("bertscore"),
}

results = {}

print("\nEvaluating SARI...")
sari_score = metrics["sari"].compute(predictions=predictions, sources=sources, references=references)
results["sari"] = sari_score
print(f"SARI: {sari_score['sari']:.2f}")

print("\nEvaluating BLEU...")
bleu_score = metrics["bleu"].compute(predictions=predictions, references=flat_references)
results["bleu"] = bleu_score
print(f"BLEU: {bleu_score['bleu']:.2f}")

print("\nEvaluating ROUGE...")
rouge_score = metrics["rouge"].compute(predictions=predictions, references=flat_references)
results["rouge"] = rouge_score
print(f"ROUGE-L: {rouge_score['rougeL']:.2f}")

print("\nEvaluating BERTScore...")
bertscore_result = metrics["bertscore"].compute(predictions=predictions, references=flat_references, lang="de")  # German language
bertscore_avg = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])
results["bertscore"] = {"average_f1": bertscore_avg}
print(f"BERTScore (avg F1): {bertscore_avg:.2f}")

output_path = Path(input_path).parent / "multiple_metrics_score.json"
Path(output_path).write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\nAll scores saved to {output_path}")