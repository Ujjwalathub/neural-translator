# setup_model.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# The specific ID for the English -> Hindi model
MODEL_ID = "Helsinki-NLP/opus-mt-en-hi"

print(f"⬇️ Downloading {MODEL_ID}...")

# 1. Download & Cache Tokenizer (The tool that turns text into numbers)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2. Download & Cache Model (The brain)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

# 3. Save to a local directory explicitly (Optional but recommended for stability)
model.save_pretrained("./local_marian_model")
tokenizer.save_pretrained("./local_marian_model")

print("✅ Installation Complete. Model saved to './local_marian_model'")
