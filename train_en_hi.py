"""
English-to-Hindi Neural Machine Translation (NMT)
Main Training Script

This script fine-tunes a pre-trained MarianMT model on the IIT Bombay English-Hindi Corpus
with optimizations for consumer GPUs (NVIDIA RTX 3050 with 6GB VRAM).

Optimizations:
- Mixed Precision Training (fp16)
- Gradient Accumulation
- Limited dataset size to prevent memory overflow
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset
from torch.cuda.amp import autocast
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# MODEL CONFIGURATION CLASS
# ============================================================================

class ModelConfig:
    """Configuration for MarianMT model optimized for RTX 3050 (6GB VRAM)."""
    
    # Model paths: Use local path if setup_model.py was run, otherwise use HuggingFace ID
    model_name = "./local_marian_model"  # Change to "Helsinki-NLP/opus-mt-en-hi" if local version unavailable
    output_dir = "./final_model"
    checkpoint_dir = "./en-hi-model"
    
    # 6GB VRAM OPTIMIZATIONS
    batch_size = 8                  # Keep low to prevent crashing
    max_length = 128                # Max sentence length (English words)
    fp16 = True                     # Enable Mixed Precision (Crucial for RTX 3050)
    
    # Text Generation Settings
    num_beams = 4                   # Look for 4 best translations (Beam Search)
    
    # Training Parameters (Optimized for RTX 3050 6GB)
    grad_accumulation = 4           # Gradient accumulation steps (8 * 4 = 32 effective batch)
    num_epochs = 3                  # Number of training epochs
    learning_rate = 2e-5            # Learning rate
    warmup_steps = 500              # Warmup steps
    
    # Data Parameters
    dataset_size = 10000            # Limit dataset to prevent RAM overflow
    max_input_length = 128          # Max input sequence length
    max_target_length = 128         # Max target sequence length

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load configuration
config = ModelConfig()

# Model and Tokenizer
MODEL_NAME = config.model_name
OUTPUT_DIR = config.output_dir
CHECKPOINT_DIR = config.checkpoint_dir

# Training Parameters (Optimized for RTX 3050 6GB)
BATCH_SIZE = config.batch_size
GRAD_ACCUMULATION = config.grad_accumulation
NUM_EPOCHS = config.num_epochs
LEARNING_RATE = config.learning_rate
WARMUP_STEPS = config.warmup_steps

# Data Parameters
DATASET_SIZE = config.dataset_size
MAX_INPUT_LENGTH = config.max_input_length
MAX_TARGET_LENGTH = config.max_target_length

# Device and precision
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = config.fp16 if torch.cuda.is_available() else False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_device_info():
    """Print device and CUDA information."""
    print("\n" + "="*70)
    print("DEVICE INFORMATION")
    print("="*70)
    print(f"Device: {DEVICE}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Mixed Precision (FP16): {USE_FP16}")
    else:
        print("WARNING: No CUDA device found. Training will be VERY slow on CPU!")
    
    print("="*70 + "\n")


def preprocess_function(examples):
    """Preprocess the dataset."""
    # Handle nested translation structure
    inputs = [ex['en'] for ex in examples['translation']]
    targets = [ex['hi'] for ex in examples['translation']]
    
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_and_prepare_dataset():
    """Load and preprocess the IIT Bombay English-Hindi Corpus."""
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    try:
        # Load the dataset from Hugging Face Hub
        print("Downloading IIT Bombay English-Hindi Corpus...")
        dataset = load_dataset("cfilt/iitb-english-hindi", split="train")
        
        # Limit dataset size to prevent RAM overflow
        dataset = dataset.select(range(min(DATASET_SIZE, len(dataset))))
        print(f"Dataset size: {len(dataset)} samples")
        
        # Split into train and validation
        train_val = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_val["train"]
        val_dataset = train_val["test"]
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Tokenize the datasets
        print("Tokenizing datasets...")
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train dataset"
        )
        
        val_dataset = val_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation dataset"
        )
        
        print("="*70 + "\n")
        return train_dataset, val_dataset
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you are connected to the internet and have an active HuggingFace account.")
        raise


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training pipeline."""
    
    # Print device information
    print_device_info()
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    print(f"Model loaded: {MODEL_NAME}")
    print(f"Model parameters: {model.num_parameters():,}")
    print()
    
    # Load and prepare dataset
    train_dataset, val_dataset = load_and_prepare_dataset()
    
    # Training arguments with memory optimizations
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_steps=50,
        fp16=USE_FP16,                           # Mixed Precision Training
        fp16_backend="auto",
        gradient_checkpointing=True,             # Additional memory saving
        save_total_limit=2,                      # Keep only 2 latest checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        dataloader_pin_memory=True if torch.cuda.is_available() else False,
        dataloader_num_workers=0,                # Prevent multiprocessing issues on Windows
        seed=42,
    )
    
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Gradient Accumulation Steps: {GRAD_ACCUMULATION}")
    print(f"Effective Batch Size: {BATCH_SIZE * GRAD_ACCUMULATION}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Num Epochs: {NUM_EPOCHS}")
    print(f"Mixed Precision (FP16): {USE_FP16}")
    print(f"Gradient Checkpointing: True")
    print("="*70 + "\n")
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("Starting training...\n")
    try:
        trainer.train()
        
        # Save the final model
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print("Saving final model...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Final model saved to: {OUTPUT_DIR}")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()
