# 📘 English-to-Hindi Neural Machine Translation (NMT)

## 1. Project Overview

This project implements a Neural Machine Translation system capable of translating English text into Hindi.

Instead of training a massive model from scratch (which requires industrial-grade GPUs), this project utilizes **Transfer Learning**. We fine-tune a pre-trained MarianMT model (specifically `Helsinki-NLP/opus-mt-en-hi`) on the IIT Bombay English-Hindi Corpus.

### Key Features:
- **Lightweight**: Optimized to run on consumer hardware (NVIDIA RTX 3050 6GB).
- **Architecture**: Transformer (Encoder-Decoder).
- **Optimization**: Uses Mixed Precision (FP16) and Gradient Accumulation to bypass memory bottlenecks.

---

## 2. System Architecture

The underlying architecture is the **Transformer**, the industry standard for NLP tasks.

### Why MarianMT?
- **Size**: ~300MB parameters (fits easily in 6GB VRAM).
- **Pre-training**: It has already "seen" millions of sentence pairs, meaning it understands basic grammar and vocabulary before we even start training.
- **Tokenizer**: Comes with a built-in SentencePiece tokenizer optimized for multilingual data.

---

## 3. Installation & Setup Guide

### Hardware Requirements
- **CPU**: AMD Ryzen 5 7000HS (or equivalent).
- **GPU**: NVIDIA RTX 3050 (6GB VRAM). Crucial for training.
- **RAM**: 8GB minimum (16GB recommended).

### Step 1: Verify Python Version
Ensure you have Python 3.10 or newer installed. Open Terminal (Command Prompt or PowerShell) and run:

```bash
python --version
```

### Step 2: Create Virtual Environment
The project includes a `venv` folder for an isolated Python environment:

```bash
python -m venv venv
```

Activate it:
- **Windows (PowerShell)**:
  ```bash
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```bash
  venv\Scripts\activate.bat
  ```
- **Mac/Linux**:
  ```bash
  source venv/bin/activate
  ```

### Step 3: Install PyTorch with CUDA (GPU Support)

⚠️ **IMPORTANT**: Do not just run `pip install torch`. That usually installs the slow CPU version.

Run this specific command to install the GPU-optimized version (CUDA 11.8 is compatible with RTX 3050):

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Verify GPU Connection

We've provided `check_gpu.py` to confirm your RTX 3050 is active. Run:

```bash
python check_gpu.py
```

**Expected Output**:
```
PyTorch Version: 2.x.x
CUDA Available: True
GPU Name: NVIDIA GeForce RTX 3050
✅ Success: Your RTX 3050 is ready for training.
```

If you see "❌ Error", PyTorch is running on CPU. Re-install with the CUDA command above.

### Step 5: Install Dependencies

Install the Hugging Face ecosystem libraries and all dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

This includes:
- **transformers**: The library that downloads and runs the MarianMT model
- **datasets**: Tools to load and process training data
- **sentencepiece**: Required for Hindi/English sub-word tokenization
- **accelerate**: Optimizes training on your GPU
- **sacremoses**: Text processing utilities

### Step 6: Download & Cache the MarianMT Model

Instead of downloading the model every time, we cache it locally for stability and offline access.

Run the setup script once:

```bash
python setup_model.py
```

This will:
1. Download the MarianMT tokenizer (~50MB)
2. Download the MarianMT model weights (~300MB)
3. Save everything to `./local_marian_model/`

**Expected Output**:
```
⬇️ Downloading Helsinki-NLP/opus-mt-en-hi...
✅ Installation Complete. Model saved to './local_marian_model'
```

---

## 4. Project Directory Structure

After setup, your project folder should look like this:

```
e:\How\
│
├── venv/                      # Virtual environment (created by Step 2)
│
├── setup_model.py             # Script to download/install MarianMT
├── check_gpu.py               # Script to verify RTX 3050 usage
├── translator.py              # Script to run inference (translation)
├── train_en_hi.py             # Main training script
├── test.py                    # Testing/inference script
├── requirements.txt           # Python dependencies
├── README.md                  # This documentation
│
├── local_marian_model/        # [FOLDER] Created by setup_model.py
│   ├── config.json
│   ├── pytorch_model.bin      # The heavy weights (approx 300MB)
│   ├── source.spm
│   ├── target.spm
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
│
├── en-hi-model/               # [FOLDER] Auto-generated during training
│   └── (checkpoints)
│
└── final_model/               # [FOLDER] Auto-generated after training
    ├── config.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.json
```

---

## 5. Configuration Overview (train_en_hi.py)

The training script uses a `ModelConfig` class optimized for RTX 3050 (6GB VRAM):

```python
class ModelConfig:
    model_name = "./local_marian_model"  # Use local cached model
    batch_size = 8                       # Keep low to prevent crashing
    max_length = 128                     # Max sentence length
    fp16 = True                          # Enable Mixed Precision (saves 50% RAM)
    num_beams = 4                        # Beam search for better translations
    grad_accumulation = 4                # Effective batch size = 8 * 4 = 32
    num_epochs = 3                       # Training iterations
    learning_rate = 2e-5                 # How aggressively the model learns
    dataset_size = 10000                 # Limiting dataset to prevent RAM overflow
    max_input_length = 128               # Max input sequence length
    max_target_length = 128              # Max target sequence length
```

### Memory Optimization Techniques

**Mixed Precision Training (fp16=True)**:
- Standard training uses 32-bit floating-point numbers (FP32).
- We use 16-bit (FP16), which halves memory usage and speeds up computation on RTX cards with minimal accuracy loss.

**Gradient Accumulation**:
- Your GPU can only handle ~8 sentences (batch) at once. This is too small for stable learning.
- Solution: Process 4 batches of 8 sequentially, accumulating the gradients, then make one update.
- **Math**: Batch Size (8) × Accumulation Steps (4) = Effective Batch Size (32).

---

## 6. Using the Translation System

### Option A: Quick Translation (Inference Only)

If you just want to translate sentences without training:

```bash
python translator.py
```

This loads the pre-trained MarianMT model and translates: "I am installing a neural network on my laptop." to Hindi.

**To translate your own text**:
Edit `translator.py` and modify the test sentence:
```python
en_text = "Your English sentence here"
```

### Option B: Training (Fine-tuning)

To train the model on your own dataset:

```bash
python train_en_hi.py
```

**What to expect**:
- A progress bar showing training epochs
- **Loss Value**: Should decrease over time (e.g., 4.5 → 2.0 or lower)
- **Time**: 30 minutes to a few hours depending on dataset size and epochs
- Final model saved to `./final_model/`

### Option C: Custom Testing

Use `test.py` to evaluate translations on your own English sentences:

```bash
python test.py
```

---

## 7. Monitoring Training

### GPU Utilization

Open **Task Manager** → **Performance** → **GPU 0**:
- Watch "Dedicated GPU memory" (should be 4GB+)
- This confirms your RTX 3050 is being used
- If "Shared GPU memory" spikes, you're running low on VRAM

### Training Loss

The training script logs metrics using TensorBoard (optional). View real-time metrics:

```bash
tensorboard --logdir=./runs
```

Then open `http://localhost:6006` in your browser.

---

## 8. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **CUDA Out of Memory (OOM)** | Model exceeds 6GB VRAM | Decrease `batch_size` to 4 in `ModelConfig`. Increase `grad_accumulation` to 8. |
| **Training is very slow** | Running on CPU instead of GPU | Check `check_gpu.py` output. Reinstall PyTorch with CUDA command in Step 3. |
| **Translations are nonsense** | Model hasn't trained long enough | Increase `num_epochs` to 5-10. Increase `dataset_size` to 50,000. |
| **"ModuleNotFoundError: transformers"** | Missing dependencies | Run `pip install -r requirements.txt` |
| **GPU not detected** | Wrong PyTorch version | Run `check_gpu.py` and follow the error message. Reinstall PyTorch. |

---

## 9. Technical Details: Architecture

The MarianMT model is an **Encoder-Decoder Transformer**:

1. **Encoder**: Reads English text and creates a numerical representation (embeddings)
2. **Decoder**: Uses the encoder's output to generate Hindi text word-by-word
3. **Attention**: Allows the decoder to focus on relevant parts of the English input

**Why this works**: The model learned from millions of English-Hindi pairs, so it generalizes well to unseen sentences.

---

## 10. Future Improvements (Scaling Up)

If you get access to cloud GPUs (Google Colab Pro, AWS A100s, etc.), you can:

- **Remove data limit**: Train on full IIT Bombay corpus (1.6M sentences)
- **Increase sequence length**: Support longer paragraphs (max_length: 512)
- **Use larger models**: Switch to `mbart-large-50` (requires ~16GB VRAM)
- **Multi-GPU training**: Parallelize across multiple GPUs
- **Fine-tune on domain-specific data**: Legal documents, medical texts, etc.

---

## 11. Useful Links

- **MarianMT Model**: https://huggingface.co/Helsinki-NLP/opus-mt-en-hi
- **IIT Bombay Corpus**: https://huggingface.co/datasets/cfilt/iitb-english-hindi
- **Hugging Face Docs**: https://huggingface.co/docs/transformers
- **PyTorch Docs**: https://pytorch.org/tutorials/

---

**Happy translating! 🚀**
