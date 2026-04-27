# BANKING77 Intent Classification (Unsloth + Llama-3.2)

## 1. Overview

This project aims to fine-tune `Llama3.2-3B-unsloth-bnb-4bit` for BANKING77 intent classification.

Main pipeline:

1. Download the original BANKING77 dataset (using the `.csv` in git)
2. Clean and split the dataset into train/validation/test
3. Fine-tune with Unsloth + LoRA
4. Run single-message inference and return the corresponding intent label

---

## 2. Project Structure

```text
banking-intent-unsloth/
├── configs/
│   ├── train.yaml          # Hyperparameters and LoRA settings for training
│   └── inference.yaml      # Model path and generation settings for inference
├── model/
│   ├── adapter_config.json         # LoRA adapter configuration
│   ├── adapter_model.safetensors   # Trained LoRA weights
│   ├── tokenizer.json              # Tokenizer vocabulary
│   └── tokenizer_config.json       # Tokenizer settings
├── sample_data/
│   ├── train.csv           # Training split of BANKING77
│   ├── test.csv            # Test split of BANKING77
│   ├── categories.json     # List of 77 intent category names
│   └── map.json            # Category name → numeric label mapping
├── scripts/
│   ├── preprocess.py       # Dataset loading, label mapping, and train/val splitting
│   ├── label_map.py        # LabelConverter: loads category/map JSONs
│   ├── train.py            # IntentTrainer: end-to-end fine-tuning pipeline
│   └── inference.py        # IntentClassification: loads adapter and runs prediction
├── train.sh                # Shell entry-point for training
├── inference.sh            # Shell entry-point for inference
├── requirements.txt        # Pinned Python dependencies
└── README.md
```

---

## 3. System Requirements

| Component | Requirement |
| ----------- | ------------- |
| **OS** | Linux / Windows (WSL recommended for training) |
| **Python** | 3.10 or 3.11 |
| **GPU** | NVIDIA GPU with ≥ 14 GB VRAM (e.g. RTX 3090 / A100) |
| **CUDA** | 12.1+ |
| **Disk** | ~5 GB free (model weights + dataset) |

> **Note:** Unsloth requires a CUDA-capable GPU. CPU-only environments are **not** supported for training or inference.

---

## 4. Environment Setup

### 4.1 Clone the repository

```bash
git clone https://github.com/Akuseru0509/banking-intent-unsloth.git
cd banking-intent-unsloth
```

### 4.2 Create and activate a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 4.3 Install dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Data Preparation

The repository already includes preprocessed data in `sample_data/`. No additional download is required.

### 5.1 Dataset overview

The [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) dataset contains **13,083** customer service queries across **77** banking intent categories.

| Split | File | Samples |
| ------- | ------ | --------- |
| Train (90%) | `sample_data/train.csv` | ~9,000 |
| Validation (10% of train) | held-out at runtime | ~1,000 |
| Test | `sample_data/test.csv` | ~3,080 |

### 5.2 Expected CSV format

```csv
text,category
"What is the exchange rate?",exchange_rate
"I lost my card.",lost_or_stolen_card
...
```

### 5.3 Label files

- `categories.json` — ordered list of the 77 intent names
- `map.json` — `{ "intent_name": numeric_id, ... }` mapping used during training and inference

### 5.4 How preprocessing works

`scripts/preprocess.py` (`DataProcessor`):

1. Loads a CSV with `datasets.load_dataset`
2. Maps each `category` string to a numeric label via `map.json`
3. Performs a stratified 90/10 train/validation split (seed `42`)

---

## 6. Training

### 6.1 Configuration (`configs/train.yaml`)

```yaml
model_name: "unsloth/llama-3.2-3b-unsloth-bnb-4bit"
max_seq_length: 512
load_in_4bit: True
learning_rate: 1e-4
optimizer: "adamw_8bit"
batch_size: 2
gradient_accumulation: 8
epochs: 1
scheduler: "linear"
warmup_steps: 10
eval_steps: 100
save_steps: 100
output_dir: "../model"

# LoRA
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj]
```

Adjust these values before running if needed.

### 6.2 Prompt template

The model is trained with an alpaca-style instruction prompt:

```text
### Instruction:
Classify the intent of the following banking request.

### Input:
<user message>

### Response:
Answer: <|label|> <numeric_label_id>
```

Only the tokens after `<|label|>` contribute to the loss (prefix tokens are masked with `-100`).

### 6.3 Run training

```bash
bash train.sh
```

This executes `cd scripts && python train.py`.

The best checkpoint (lowest `eval_loss`) is automatically saved to `model/`.

---

## 7. Evaluation Results

> 📊 *Run `bash train.sh` (which also calls `_evaluate`) or execute `scripts/train.py` to populate these numbers.*

| Metric | Score |
| ---------- | ------- |
| Accuracy | 91.56% |
| Macro F1 | 91.56% |

**Test set:** `sample_data/test.csv` (~3,080 samples, 77 classes)

---

## 8. Inference

### 8.1 Configuration (`configs/inference.yaml`)

```yaml
model_path: "../model"
max_seq_length: 512
load_in_4bit: True
max_new_tokens: 2
```

### 8.2 Finetuned model configurations

Model configurations can be found in the evaluate/inference Kaggle notebook. Link shown below.

### 8.3 Run inference

```bash
bash inference.sh
```

This executes `cd scripts && python inference.py --message "Am I able to get a card in EU?"`.

The default message in `inference.py` is:

```python
message = "Am I able to get a card in EU?"
```

You can change this to any banking-related query. The script will print the predicted intent label, e.g.:

```text
country_support
```

### 8.3 Using `IntentClassification` programmatically

```python
from scripts.inference import IntentClassification
from pathlib import Path

classifier = IntentClassification(
    model_path=Path("path/to/model"),
    yaml_path=Path("path/to/config")
)

intent = classifier("Am I able to get a card in EU?")
print(intent)  # → "country_support"
```

---

## 9. Demonstration Video

> **Please note that my laptop doesn't have a GPU so my demo will be on Kaggle**
> **Link to Kaggle inference and evaluation notebook: https://www.kaggle.com/code/akuseru59/banking-intent-evalutation-inference**
> **Link to Kaggle train notebook: https://www.kaggle.com/code/akuseru59/banking-intent-train**
> *Link to demo: https://drive.google.com/drive/folders/1FIKelfPbxUhSbxNdDDfXKXgZSLCjrpGe?usp=sharing*

---

## 10. References

- [BANKING77 Dataset — PolyAI](https://huggingface.co/datasets/PolyAI/banking77)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Llama 3.2 — Meta AI](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- [TRL — Hugging Face](https://github.com/huggingface/trl)
- [PEFT — Hugging Face](https://github.com/huggingface/peft)
