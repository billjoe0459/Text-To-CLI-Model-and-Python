import os
import json
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


# ============================================================
# CONFIGURATION
# ============================================================

# Coding-specialized, mature ecosystem, fits an 8 GB Jetson budget.
# See notes at the bottom of the chat response for why this beats
# Qwen3.5-4B for this particular project.
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct" # or "Qwen/Qwen2.5-Coder-3B-Instruct"

DATASET_FILE = "Final_Project/AI_Training/CLI_training_set/dataset.json"

OUTPUT_DIR = "Final_Project/AI_Training/outputs"
FINAL_ADAPTER_DIR = "Final_Project/AI_Training/qwen-cli-lora"

MAX_SEQ_LENGTH = 512

# Jetson Orin Nano 8 GB
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

EPOCHS = 2

LEARNING_RATE = 2e-4

SEED = 3407

# A short system prompt anchors the model's existing instruct-tuned
# behavior instead of fighting it with a made-up format.
SYSTEM_PROMPT = (
    "You are a helpful assistant that converts natural language "
    "requests into a single shell command. Respond with only the "
    "command, no explanation."
)


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 60)
print("SYSTEM CHECK")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU:", torch.cuda.get_device_name(0))

gpu_properties = torch.cuda.get_device_properties(0)

print(
    f"GPU memory: "
    f"{gpu_properties.total_memory / 1024**3:.2f} GB"
)

print("=" * 60)


# ============================================================
# TOKENIZER (loaded early so formatting can use the chat template
# and the real eos_token)
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print("Examples loaded:", len(raw_data))


# Actual dataset format (confirmed from the uploaded file):
#
# [
#   {
#     "prompt": "Find all Python files recursively",
#     "response": "find . -type f -name '*.py'"
#   },
#   {
#     "prompt": "Show the current directory",
#     "response": "pwd"
#   }
# ]
#
# NOTE: this dataset has 780 rows but only 651 unique prompts (129
# exact duplicates), and 8 prompts map to two different responses
# depending on which copy is read first (e.g. "Turn on Bluetooth" ->
# both "sudo systemctl start bluetooth" and "rfkill unblock
# bluetooth"). Left as-is per instructions -- just flagging that
# training will see some prompts up to 4x more often than others,
# and a couple of prompts with inconsistent target commands, which
# may show up as repetition or hedging in generations later.


def normalize_output(output):
    """
    Converts the response field into a plain string regardless of
    whether the dataset stored it as a str, list, or dict.
    """

    if isinstance(output, list):
        return "\n".join(str(x) for x in output).strip()

    if isinstance(output, dict):
        return json.dumps(output, ensure_ascii=False).strip()

    return str(output).strip()


def format_example(example):
    """
    Converts each dataset example into Qwen's chat template, so the
    model trains on the same turn structure (system/user/assistant
    with real special tokens) it was instruction-tuned on, rather
    than an ad-hoc "### Input / ### Output" format it has never seen.

    apply_chat_template appends the model's real end-of-turn token
    for us, so the model learns where to stop -- this was silently
    missing in the original "### Input/### Output" string format.
    """

    user_input = str(example["prompt"]).strip()
    assistant_output = normalize_output(example["response"])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_output},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


formatted_data = [
    format_example(example)
    for example in raw_data
]


dataset = Dataset.from_list(formatted_data)


print("Formatted dataset:", len(dataset))
print("\nExample:")
print("-" * 60)
print(dataset[0]["text"])
print("-" * 60)


# ============================================================
# 4-BIT QUANTIZATION
# ============================================================

print("Configuring 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True,
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print(f"Loading {MODEL_NAME}...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,

    quantization_config=bnb_config,

    torch_dtype=torch.float16,

    device_map="auto",

    trust_remote_code=True,
)


model.config.use_cache = False


# ============================================================
# PREPARE FOR QLORA
# ============================================================

print("Preparing model for QLoRA...")

model = prepare_model_for_kbit_training(
    model
)


# ============================================================
# LoRA CONFIGURATION
# ============================================================

print("Adding LoRA adapters...")

lora_config = LoraConfig(

    # Start conservatively on 8 GB
    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",

        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)


model = get_peft_model(
    model,
    lora_config,
)


model.print_trainable_parameters()


# ============================================================
# COMPLETION-ONLY LOSS MASKING
# ============================================================
# Without this, cross-entropy is computed over the whole
# "<system><user><assistant>" sequence, including the prompt tokens,
# which wastes capacity teaching the model to predict inputs it will
# never need to generate. This collator zeroes out the loss for
# everything before the assistant's turn.
#
# response_template must match a token sequence that actually
# appears verbatim in the rendered chat template. Qwen's ChatML-style
# template opens each turn with "<|im_start|>{role}\n", so the
# assistant turn begins with this literal string.

response_template = "<|im_start|>assistant\n"

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    # 8 GB Jetson-friendly
    per_device_train_batch_size=MICRO_BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    num_train_epochs=EPOCHS,

    learning_rate=LEARNING_RATE,

    warmup_steps=50,

    lr_scheduler_type="cosine",

    weight_decay=0.01,

    # CUDA FP16
    fp16=True,

    bf16=False,

    # bitsandbytes optimizer
    optim="paged_adamw_8bit",

    logging_steps=10,

    save_strategy="epoch",

    save_total_limit=2,

    seed=SEED,

    report_to="none",

    dataloader_pin_memory=False,

    gradient_checkpointing=True,

    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },

    remove_unused_columns=True,

)


# ============================================================
# TRAINER
# ============================================================
# NOTE ON TRL VERSIONS:
# This uses the classic SFTTrainer(tokenizer=..., dataset_text_field=...,
# max_seq_length=...) signature. TRL >= 0.12-0.13 moved those args into
# SFTConfig and renamed `tokenizer` to `processing_class`. Check
# `import trl; print(trl.__version__)` before running -- if you're on
# a newer trl this call will raise a TypeError for unexpected kwargs,
# and you'll need to pass max_seq_length/dataset_text_field via
# SFTConfig instead of TrainingArguments/SFTTrainer directly.

print("Creating trainer...")

trainer = SFTTrainer(

    model=model,

    tokenizer=tokenizer,

    train_dataset=dataset,

    dataset_text_field="text",

    max_seq_length=MAX_SEQ_LENGTH,

    dataset_num_proc=1,

    data_collator=collator,

    args=training_args,

)


# ============================================================
# TRAIN
# ============================================================

print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

print(
    "Effective batch size:",
    MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
)

print("Epochs:", EPOCHS)
print("Max sequence length:", MAX_SEQ_LENGTH)

trainer_stats = trainer.train()


# ============================================================
# SAVE LoRA ADAPTER
# ============================================================

print("Saving LoRA adapter...")

model.save_pretrained(
    FINAL_ADAPTER_DIR
)

tokenizer.save_pretrained(
    FINAL_ADAPTER_DIR
)


print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "Adapter saved to:",
    FINAL_ADAPTER_DIR
)


# ============================================================
# MEMORY REPORT
# ============================================================

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    print(
        f"GPU memory allocated: {allocated:.2f} GB"
    )

    print(
        f"GPU memory reserved: {reserved:.2f} GB"
    )