German Text Simplification Pipeline

This project focuses on simplifying German texts using Transformer-based models, particularly T5, trained on the GNATS dataset.

⸻

📦 Installation

Ensure you have Python 3.10 or 3.11. Then, install all dependencies:

pip install -r requirements.txt


⸻

🚀 Pipeline Overview

Step 0: (Optional) Prepare GNATS Dataset

If you don’t have the GNATS dataset (train.csv, val.csv, test.csv in data/gnats/), run:

python scripts/prepare_gnats_data.py


⸻

Step 1: Preprocess Dataset

Clean and preprocess the dataset:

python scripts/preprocess_text.py

Output:
	•	data/gnats/clean_train.csv
	•	data/gnats/clean_val.csv
	•	data/gnats/clean_test.csv

⸻

Step 2: Train T5 Model

Fine-tune the T5 model on the cleaned dataset:

python scripts/train_t5_small.py

Output:
	•	Model checkpoints saved in models/t5-gnats-clean/

⸻

Step 3: Generate Predictions & Evaluate with SARI

Generate predictions on the test set and evaluate using SARI metric:

python scripts/generate_and_evaluate.py

Output:
	•	results/test_predictions.csv (model predictions)
	•	results/sari_score.json (SARI evaluation score)

⸻

Step 4: Evaluate Multiple Metrics

Evaluate predictions with multiple metrics: SARI, BLEU, ROUGE, and BERTScore.

python scripts/evaluate_multiple_metrics.py results/test_predictions.csv

Output:
	•	results/multiple_metrics_score.json

⸻

📝 Notes
	•	Follow the pipeline in the given order for consistent results.
	•	You only need to run Step 0 if you don’t already have the GNATS dataset.
	•	You can adjust parameters in each script to customize training or evaluation.

⸻

👨‍💻 Author

This pipeline was developed as part of a thesis project on German text simplification using NLP and Transformer models.