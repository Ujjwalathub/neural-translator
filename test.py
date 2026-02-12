"""
English-to-Hindi Neural Machine Translation (NMT)
Inference Script

This script loads the trained model and performs English-to-Hindi translations.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = "./final_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_model_and_tokenizer():
    """Load the trained model and tokenizer."""
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory '{MODEL_DIR}' not found!")
        print("Please train the model first using: python train_en_hi.py")
        return None, None
    
    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded successfully!")
    print(f"Device: {DEVICE}\n")
    
    return model, tokenizer


def translate(text, model, tokenizer):
    """
    Translate English text to Hindi.
    
    Args:
        text (str): English text to translate
        model: Loaded translation model
        tokenizer: Loaded tokenizer
    
    Returns:
        str: Translated Hindi text
    """
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(DEVICE)
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True,
        )
    
    # Decode the translation
    translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return translation


def interactive_mode(model, tokenizer):
    """Interactive translation mode."""
    print("="*70)
    print("INTERACTIVE TRANSLATION MODE")
    print("="*70)
    print("Enter English text to translate to Hindi (type 'exit' to quit)\n")
    
    while True:
        user_input = input("English: ").strip()
        
        if user_input.lower() == 'exit':
            print("Exiting translation mode. Thank you!")
            break
        
        if not user_input:
            print("Please enter some text.\n")
            continue
        
        translation = translate(user_input, model, tokenizer)
        print(f"Hindi: {translation}\n")


def batch_mode(model, tokenizer):
    """Batch translation with predefined examples."""
    print("="*70)
    print("BATCH TRANSLATION MODE")
    print("="*70 + "\n")
    
    # Example sentences for testing
    test_sentences = [
        "Hello, how are you?",
        "Artificial Intelligence is changing the world.",
        "Machine learning is a subset of artificial intelligence.",
        "What is your name?",
        "I love machine translation.",
        "This is a beautiful day.",
        "Python is a powerful programming language.",
        "Deep learning is revolutionizing natural language processing.",
    ]
    
    print(f"Translating {len(test_sentences)} example sentences:\n")
    
    for i, english_text in enumerate(test_sentences, 1):
        hindi_text = translate(english_text, model, tokenizer)
        print(f"{i}. English: {english_text}")
        print(f"   Hindi:   {hindi_text}\n")


def main():
    """Main inference pipeline."""
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        return
    
    print("="*70)
    print("ENGLISH-TO-HINDI TRANSLATION")
    print("="*70)
    print("\nChoose a mode:")
    print("1. Batch Translation (Predefined examples)")
    print("2. Interactive Mode (Enter your own text)")
    print()
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        batch_mode(model, tokenizer)
    elif choice == "2":
        interactive_mode(model, tokenizer)
    else:
        print("Invalid choice. Running batch mode by default.\n")
        batch_mode(model, tokenizer)


if __name__ == "__main__":
    main()
