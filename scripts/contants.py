import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, "data")
GNATS_DIR = os.path.join(DATA_DIR, "gnats")
GNATS_TATOEBA_DIR = os.path.join(GNATS_DIR, "tatoeba")

CLEAN_TRAIN_CSV = os.path.join(GNATS_TATOEBA_DIR, "clean_train.csv")
CLEAN_VAL_CSV = os.path.join(GNATS_TATOEBA_DIR, "clean_val.csv")
CLEAN_TEST_CSV = os.path.join(GNATS_TATOEBA_DIR, "clean_test.csv")

MODELS_DIR = os.path.join(ROOT_DIR, "models")
T5_GNATS_MODEL_DIR = os.path.join(MODELS_DIR, "t5-gnats-clean")

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TEST_PREDICTIONS_CSV = os.path.join(RESULTS_DIR, "test_predictions.csv")
SARI_SCORE_JSON = os.path.join(RESULTS_DIR, "sari_score.json")
MULTIPLE_METRICS_JSON = os.path.join(RESULTS_DIR, "multiple_metrics_score.json")

SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")