# scripts/run_inference_t5_small.py
import sys
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Path to the checkpoint (you can change the checkpoint number here)
model_path = "models/t5-gnats-clean/checkpoint-21"

# Load tokenizer and model from the checkpoint
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Check input from the command line
if len(sys.argv) < 2:
    print("❗️ You need to provide a sentence to simplify:")
    print("Usage: python scripts/run_inference_t5_small.py \"Your sentence\"")
    sys.exit(1)

input_text = sys.argv[1]

# Add prefix for T5
prompt = "translate German to Simple German: " + input_text

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")

# Generate output
outputs = model.generate(**inputs, max_length=128)

# Decode
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n🟢 Simplified result (de_SI):")
print(result)