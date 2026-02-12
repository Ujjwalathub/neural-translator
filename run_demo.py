"""
English-to-Hindi Translation Demo
Demonstrates the trained model's translation capabilities
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configuration
MODEL_DIR = "./final_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("ENGLISH-TO-HINDI NEURAL MACHINE TRANSLATION DEMO")
print("=" * 70)

# Load model
print(f"\n⏳ Loading model from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()
print(f"✓ Model loaded successfully on {DEVICE}")

# Test sentences
test_sentences = [
    "Hello, how are you?",
    "I love machine learning.",
    "Where is the train station?",
    "What is your name?",
    "I am learning Hindi and English.",
    "The weather is beautiful today.",
    "Can you help me with this problem?"
]

print(f"\n{'English':<40} | {'Hindi Translation':<40}")
print("-" * 82)

# Translate each sentence
for sentence in test_sentences:
    inputs = tokenizer(sentence, return_tensors="pt", padding=True).to(DEVICE)
    
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(f"{sentence:<40} | {translation:<40}")

print("\n" + "=" * 70)
print("✓ Demo completed successfully!")
print("=" * 70)
