#!/bin/bash
# merge_and_convert.sh
# One-shot: merge LoRA adapter into base model (torch-only, NO PEFT),
# convert to GGUF fp16, then quantize to Q4_K_M (~1 GB)
#
# Usage:
#   chmod +x merge_and_convert.sh
#   ./merge_and_convert.sh

set -e

# ── Configuration ───────────────────────────────────────────────────────────
BASE_MODEL_ID="Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_DIR="./qwen-cli-lora"
MERGED_DIR="./merged_model"
F16_GGUF="qwen2.5-cli-fp16.gguf"
Q4_GGUF="qwen2.5-cli-q4_k_m.gguf"
LLAMA_CPP="./llama.cpp"

# ── Step 0: Check prerequisites ─────────────────────────────────────────────
echo "[0/5] Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

if ! python3 -c "import safetensors" 2>/dev/null; then
    echo "Installing safetensors..."
    pip install safetensors -q
fi

if [ ! -d "$LLAMA_CPP" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP"
fi

if [ ! -d "$ADAPTER_DIR" ]; then
    echo "Error: Adapter not found at $ADAPTER_DIR"
    echo "Run training first: python3 train_qlora.py"
    exit 1
fi

# ── Step 1: Merge adapter into base (torch-only, no model imports) ──────────
echo ""
echo "[1/5] Merging LoRA adapter into base model (torch-only, no transformers bloat)..."
python3 << 'PYEOF'
import torch
import json
import os
import shutil
from safetensors.torch import load_file, save_file

BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_DIR   = "./qwen-cli-lora"
OUTPUT_DIR    = "./merged_model"
CACHE_DIR     = os.path.expanduser("~/.cache/huggingface/hub")

print("  Locating base model in cache...")
cache_name = "models--" + BASE_MODEL_ID.replace("/", "--")
cache_path = os.path.join(CACHE_DIR, cache_name)

if not os.path.exists(cache_path):
    print(f"  Base model not cached. Downloading...")
    os.system(f"huggingface-cli download {BASE_MODEL_ID}")

snapshots_dir = os.path.join(cache_path, "snapshots")
if not os.path.exists(snapshots_dir):
    print(f"  ERROR: No snapshots in {cache_path}")
    exit(1)

snapshot_ids = os.listdir(snapshots_dir)
if not snapshot_ids:
    print("  ERROR: No snapshot versions found")
    exit(1)

base_path = os.path.join(snapshots_dir, snapshot_ids[0])
print(f"  Found: {base_path}")

print("  Loading base model weights...")
model_index = os.path.join(base_path, "model.safetensors.index.json")
if os.path.exists(model_index):
    with open(model_index) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    base_weights = {}
    loaded_shards = set()
    for tensor_name, shard_file in weight_map.items():
        if shard_file not in loaded_shards:
            shard_path = os.path.join(base_path, shard_file)
            shard_weights = load_file(shard_path)
            loaded_shards.add(shard_file)
        base_weights[tensor_name] = shard_weights[tensor_name]
else:
    st_file = os.path.join(base_path, "model.safetensors")
    if os.path.exists(st_file):
        base_weights = load_file(st_file)
    else:
        print("  ERROR: No safetensors found")
        exit(1)

print(f"  Loaded {len(base_weights)} tensors")

print("  Copying tokenizer/config files...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "merges.txt", "vocab.json"]:
    src = os.path.join(base_path, fname)
    if os.path.exists(src):
        shutil.copy2(src, OUTPUT_DIR)

print("  Loading adapter and merging...")
with open(os.path.join(ADAPTER_DIR, "adapter_config.json")) as f:
    adapter_config = json.load(f)

r = adapter_config["r"]
lora_alpha = adapter_config["lora_alpha"]
target_modules = adapter_config["target_modules"]
print(f"    r={r}, alpha={lora_alpha}")

lora_weights = load_file(os.path.join(ADAPTER_DIR, "adapter_model.safetensors"))
print(f"    Loaded {len(lora_weights)} adapter tensors")

scale = lora_alpha / r
merged_count = 0

for tensor_name in list(base_weights.keys()):
    is_target = any(t in tensor_name for t in target_modules)
    if not is_target:
        continue
    lora_a_key = f"base_model.model.{tensor_name}.lora_A.weight"
    lora_b_key = f"base_model.model.{tensor_name}.lora_B.weight"
    if lora_a_key not in lora_weights or lora_b_key not in lora_weights:
        continue
    lora_a = lora_weights[lora_a_key]
    lora_b = lora_weights[lora_b_key]
    delta_w = (lora_b @ lora_a) * scale
    base_weights[tensor_name] = (base_weights[tensor_name].to(torch.float32) + delta_w.to(torch.float32)).to(torch.float16)
    merged_count += 1

print(f"    Merged {merged_count} layers")

print("  Saving merged model...")
total_size = sum(v.numel() * v.element_size() for v in base_weights.values())
print(f"    Total size: {total_size / 1e9:.2f} GB")
save_file(base_weights, os.path.join(OUTPUT_DIR, "model.safetensors"))
print("  Done.")
PYEOF

# ── Step 2: Install llama.cpp deps ──────────────────────────────────────────
echo ""
echo "[2/5] Installing llama.cpp conversion dependencies..."
cd "$LLAMA_CPP"
pip install -r requirements.txt -q 2>/dev/null || pip install sentencepiece protobuf -q
cd - > /dev/null

# ── Step 3: Convert merged HF model to fp16 GGUF ────────────────────────────
echo ""
echo "[3/5] Converting merged model to fp16 GGUF..."
cd "$LLAMA_CPP"
python3 convert_hf_to_gguf.py \
    "../$MERGED_DIR" \
    --outfile "../$F16_GGUF" \
    --outtype f16
cd - > /dev/null

# ── Step 4: Build llama-quantize with CMake ─────────────────────────────────
echo ""
echo "[4/5] Building llama-quantize with CMake..."
cd "$LLAMA_CPP"
if [ ! -f "./build/bin/llama-quantize" ]; then
    echo "  Configuring with CMake..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    echo "  Compiling llama-quantize..."
    cmake --build build --config Release --target llama-quantize -j$(nproc)
else
    echo "  llama-quantize already built."
fi
cd - > /dev/null

# ── Step 5: Quantize fp16 GGUF to Q4_K_M ────────────────────────────────────
echo ""
echo "[5/5] Quantizing fp16 GGUF to Q4_K_M (~1 GB)..."
cd "$LLAMA_CPP"
./build/bin/llama-quantize \
    "../$F16_GGUF" \
    "../$Q4_GGUF" \
    Q4_K_M
cd - > /dev/null

# ── Verify and report ───────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "SINGLE-FILE GGUF READY"
echo "========================================"
echo ""
if [ -f "$Q4_GGUF" ]; then
    echo "File: $(realpath $Q4_GGUF)"
    echo "Size: $(du -h $Q4_GGUF | cut -f1)"
    echo ""
    echo "Intermediate fp16: $(du -h $F16_GGUF | cut -f1) (can delete to save space)"
    echo ""
    echo "Create Ollama model:"
    echo "  ollama create cli-merged -f Modelfile.merged"
    echo ""
    echo "Run:"
    echo "  ollama run cli-merged"
else
    echo "ERROR: $Q4_GGUF not found"
    exit 1
fi
echo "========================================"