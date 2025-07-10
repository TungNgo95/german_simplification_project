# scripts/run_inference_t5_small.py

import sys
from transformers import T5Tokenizer, T5ForConditionalGeneration

checkpoint_number = 1050  # You can change this number to use another checkpoint
model_path = f"models/t5-gnats-clean/checkpoint-{checkpoint_number}"

print(f"Using model checkpoint from: {model_path}")

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

if len(sys.argv) < 2:
    print("❗️ You need to provide a sentence to simplify:")
    print("Usage: python scripts/run_inference_t5_small.py \"Your sentence\"")
    sys.exit(1)

input_text = sys.argv[1]

prefix = "translate German to Simple German: "
if not input_text.strip().startswith(prefix):
    prompt = prefix + input_text
else:
    prompt = input_text

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=128)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nSimplified result (de_SI):")
print(result)