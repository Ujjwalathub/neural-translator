# translator.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Load the installed model
model_path = "./local_marian_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("⏳ Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

def translate(text):
    # 2. Preprocess (Text -> Numbers)
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    
    # 3. Generate (The Model "Thinks")
    translated_tokens = model.generate(
        **inputs, 
        max_length=128,
        num_beams=4, 
        early_stopping=True
    )
    
    # 4. Decode (Numbers -> Hindi Text)
    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return result[0]

# Test
if __name__ == "__main__":
    en_text = "I am installing a neural network on my laptop."
    hi_text = translate(en_text)
    print(f"\nEnglish: {en_text}")
    print(f"Hindi:   {hi_text}")
