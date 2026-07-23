#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Converts the PEFT LoRA adapter saved by train_qlora.py
# (safetensors format) into a GGUF adapter Ollama can load via
# the ADAPTER directive in the Modelfile.
#
# Run this on the Jetson (or wherever the adapter was trained) --
# it needs the same Python env used for training, plus llama.cpp's
# conversion script and its (fairly light) requirements.
# ============================================================

# ---- CONFIG: match these to your training script -----------------
ADAPTER_DIR="./qwen-cli-lora"

# Must exactly match MODEL_NAME in train_qlora.py -- this is used
# both to download the base model and to confirm it matches what
# the adapter was actually trained against.
BASE_MODEL_ID="Qwen/Qwen2.5-Coder-1.5B-Instruct"

# convert_lora_to_gguf.py's --base flag requires a LOCAL directory,
# not a Hugging Face Hub id -- it does not download anything itself.
# We fetch the base model here once and point --base at this folder.
BASE_MODEL_LOCAL_DIR="./base_model_local"

OUTFILE="./qwen2.5-cli-1.5b.gguf"
LLAMA_CPP_DIR="./llama.cpp"
# --------------------------------------------------------------

if [ ! -d "$ADAPTER_DIR" ]; then
    echo "Adapter directory not found: $ADAPTER_DIR"
    echo "Run train_qlora.py first, or fix ADAPTER_DIR above."
    exit 1
fi

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "Cloning llama.cpp for its GGUF conversion scripts..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
fi

echo "Installing llama.cpp's Python conversion requirements..."
pip install --break-system-packages -r "$LLAMA_CPP_DIR/requirements.txt"

if [ ! -d "$BASE_MODEL_LOCAL_DIR" ] || [ -z "$(ls -A "$BASE_MODEL_LOCAL_DIR" 2>/dev/null)" ]; then
    echo "Downloading base model ($BASE_MODEL_ID) to a local directory..."
    echo "(convert_lora_to_gguf.py needs local files, not a Hub id)"
    huggingface-cli download "$BASE_MODEL_ID" --local-dir "$BASE_MODEL_LOCAL_DIR"
else
    echo "Base model already present at $BASE_MODEL_LOCAL_DIR, skipping download."
fi

echo "Converting adapter to GGUF..."
python3 "$LLAMA_CPP_DIR/convert_lora_to_gguf.py" \
    "$ADAPTER_DIR" \
    --base "$BASE_MODEL_LOCAL_DIR" \
    --outfile "$OUTFILE" \
    --outtype q8_0

echo "============================================================"
echo "Done. Adapter written to: $OUTFILE"
echo ""
echo "Next steps:"
echo "  1. Place $OUTFILE next to your Modelfile"
echo "     (Modelfile expects it at ./qwen2.5-cli-1.5b.gguf)"
echo "  2. ollama create cli-assistant -f Modelfile"
echo "  3. ollama run cli-assistant"
echo "============================================================"
