# 🇮🇳 English to Hindi Neural Machine Translator

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A local, privacy-focused **Neural Machine Translation (NMT)** system capable of translating English text into Hindi. 

This project fine-tunes the **MarianMT** architecture (specifically `Helsinki-NLP/opus-mt-en-hi`) to run efficiently on consumer hardware. It includes a custom training pipeline optimized for 6GB VRAM GPUs (like the RTX 3050) and a user-friendly Web UI built with Streamlit.

---

## 🚀 Key Features

* **Local Inference:** Runs entirely offline; no API keys or cloud data transfer required.
* **Hardware Optimized:** Uses Mixed Precision (FP16) and Gradient Accumulation to train/run on 6GB VRAM.
* **Interactive UI:** Web interface for real-time translation testing.
* **Customizable:** Scripts included for both fine-tuning on new data and running inference.

---

## 🛠️ Tech Stack

* **Model:** MarianMT (Transformer Encoder-Decoder)
* **Framework:** PyTorch & Hugging Face Transformers
* **Interface:** Streamlit
* **Dataset:** Compatible with IIT Bombay English-Hindi Corpus

---

## 📂 Project Structure

```text
├── app.py                 # Streamlit Web Application
├── setup_model.py         # Downloads & caches model weights locally
├── train_en_hi.py         # Training script (Fine-tuning)
├── test.py                # CLI Inference script
├── requirements.txt       # Project dependencies
└── README.md              # Documentation
