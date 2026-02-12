import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Translator (En-Hi)",
    page_icon="🇮🇳",
    layout="centered"
)

# --- 1. MODEL LOADING (Cached) ---
# We use @st.cache_resource so the model loads only ONCE. 
# Without this, it would reload (taking 5s) every time you type a letter.
@st.cache_resource
def load_model():
    print("Loading model...")
    # Point this to your local folder
    model_path = "./local_marian_model" 
    
    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Move to GPU if available (RTX 3050)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return tokenizer, model, device

# Load the heavy AI stuff
tokenizer, model, device = load_model()

# --- 2. TRANSLATION FUNCTION ---
def translate_text(text, num_beams=4):
    # Prepare input
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    
    # Generate output
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs, 
            max_length=128,
            num_beams=num_beams, # Higher = Better quality, Slower
            early_stopping=True
        )
    
    # Decode to text
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# --- 3. UI LAYOUT ---
st.title("🇮🇳 English to Hindi AI Translator")
st.markdown("Run locally on **RTX 3050** | Powered by *MarianMT*")

# Two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### English (Input)")
    # Text Area for input
    source_text = st.text_area(
        "Type your sentence here:", 
        height=150,
        placeholder="e.g., Artificial Intelligence is the future."
    )

with col2:
    st.markdown("### Hindi (Output)")
    # Placeholder for the output
    output_placeholder = st.empty()
    
    # Initial empty box styling
    output_placeholder.text_area("Translation:", height=150, disabled=True)

# --- 4. ACTION BUTTON ---
if st.button("Translate 🚀", type="primary"):
    if source_text:
        with st.spinner("Translating..."):
            try:
                # Run translation
                translation = translate_text(source_text)
                
                # Update the output box
                output_placeholder.text_area("Translation:", value=translation, height=150)
                
                st.success("Done!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text first.")

# --- 5. SIDEBAR INFO ---
with st.sidebar:
    st.header("System Info")
    st.write(f"**Device:** `{device.upper()}`")
    if device == "cuda":
        st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.write(f"**VRAM:** {vram:.2f} GB")
    
    st.markdown("---")
    st.info("Tip: Shorter sentences usually yield more accurate results.")
