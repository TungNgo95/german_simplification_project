from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset
import pandas as pd

# Load CSVs
train_df = pd.read_csv("data/gnats/clean_train.csv")
val_df = pd.read_csv("data/gnats/clean_val.csv")

# Prepare Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Preprocessing function
def tokenize_function(example):
    model_input = tokenizer("translate German to simplified German: " + example["source"], truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(text_target=example["target"], truncation=True, padding="max_length", max_length=128)
    model_input["labels"] = labels["input_ids"]
    return model_input

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=False)
val_dataset = val_dataset.map(tokenize_function, batched=False)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/t5-gnats-clean",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    num_train_epochs=3,
    do_eval=True, 
)

# Trainer setup
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()