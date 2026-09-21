# Semantic Code Clone Detection

### IITG.ai Tune Quest Hackathon

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-Not%20specified-lightgrey.svg)](#license)

> A binary classifier for detecting whether Python and Java code snippets implement the same behavior, even when their variable names, formatting, or coding style differ.

## Overview

This project was developed during the **IITG.ai Tune Quest**, a 48-hour solo hackathon. The task is semantic code-clone detection: given two functions, `func1` and `func2`, predict whether they are logically equivalent.

The key distinction is between **surface similarity** and **semantic similarity**. Two equivalent functions may use different identifiers, whitespace, control-flow structure, or language syntax. Conversely, two snippets with many shared tokens may implement different behavior.

The solution progresses from interpretable non-neural baselines to parameter-efficient fine-tuning:

1. A token-based Jaccard heuristic provides a zero-training reference point.
2. A balanced Logistic Regression model uses engineered structural features.
3. `google/gemma-3-1b-it` is adapted as a sequence classifier with LoRA, using a single chat-templated prompt containing both snippets.

The notebooks are designed for GPU/TPU-enabled Kaggle environments and produce validation metrics as well as competition-ready predictions.

## Dataset and preprocessing

The competition data is distributed as JSONL records with the following logical schema:

| Field | Description |
| --- | --- |
| `func1` | First code snippet |
| `func2` | Second code snippet |
| `label` | Boolean target: `True` when the snippets are semantically equivalent |

The complete training corpus contains approximately **8.44 million pairs**. For faster iteration, this project uses a representative **500,000-row sample** (`train_small.jsonl`) while developing and evaluating the pipeline. The competition test set has the same two-snippet structure without labels.

### Prompt construction and truncation

Gemma 3 is a decoder-only instruction model, so the snippets are not passed as an encoder-style sentence pair. Instead, they are placed in an instruction prompt and rendered with the tokenizer's official chat template:

```text
Compare the following two code snippets and determine whether they are
semantically equivalent (same logic / behavior), ignoring differences in
variable names, formatting, coding style, or structure.

### Code A:
<func1>

### Code B:
<func2>
```

Each snippet is clipped independently to a **1,600-character budget** before prompt construction. The final chat-formatted sequence is tokenized with truncation and a fixed maximum sequence length. Independent clipping prevents a long first function from consuming the entire context and effectively removing the second function from the model input.

## Model architecture and baselines

### Baseline 1: token Jaccard overlap

Each snippet is tokenized into identifiers, numeric literals, and punctuation. The baseline computes:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

The pair is predicted as equivalent when the similarity is at least `0.5`. This baseline is fast and interpretable, but it is sensitive to vocabulary overlap and does not model execution semantics.

### Baseline 2: engineered features with Logistic Regression

The second baseline combines Jaccard similarity with inexpensive structural features:

- Character lengths of both snippets
- Absolute character-length difference
- Character-length ratio
- Line counts of both snippets
- Absolute line-count difference
- Token Jaccard similarity

A class-balanced Logistic Regression model learns how to combine these signals. It provides a stronger classical-ML reference while remaining easy to inspect and reproduce.

### Main model: Gemma 3 with LoRA

The main model uses [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) with a newly initialized binary sequence-classification head and last-token pooling. The base model is adapted with **Low-Rank Adaptation (LoRA)** rather than fully fine-tuned:

| Configuration | Value |
| --- | --- |
| Base model | `google/gemma-3-1b-it` |
| Task | Binary sequence classification |
| Pooling | Last-token pooling |
| LoRA rank (`r`) | `16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.05` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Bias | `none` |
| Maximum sequence length | `512` tokens in the current GPU notebook; increase after memory testing |

LoRA substantially reduces the number of trainable parameters and makes a 1B-parameter model practical on limited-compute hardware. The training pipeline also enables gradient checkpointing and automatically selects `bf16` or `fp16` when supported.

## Results

The following results are from the held-out validation split in the executed GPU notebook. The reported operating point uses a default classification threshold of `0.5`; threshold tuning is also evaluated in the notebook.

| Model | F1 | ROC-AUC |
| --- | ---: | ---: |
| Token Jaccard (`@ 0.5`) | 0.5801 | 0.8308 |
| Logistic Regression (engineered features) | 0.7600 | 0.8534 |
| Gemma 3 1B + LoRA | **0.8640** | **0.9457** |

The fine-tuned model improves substantially over both non-neural baselines, indicating that contextual modeling is useful for distinguishing semantic equivalence from simple token overlap. Because the competition metric is binary F1, the notebook additionally searches for a validation threshold; the Gemma model reached approximately **0.8642 F1** at its tuned threshold.

## Repository structure

```text
cross-language-code-equivalence/
├── README.md
├── tunequest_gpu.ipynb   # GPU training, validation, and inference pipeline
└── tunequest_tpu.ipynb   # TPU v5e-8 variant using PyTorch/XLA
```

The notebooks include data loading, exploratory checks, both baselines, prompt construction, tokenization, LoRA configuration, training, evaluation, and submission generation. No separate Python package is required for the current repository layout.

## Setup and installation

### 1. Clone the repository

```bash
git clone https://github.com/prakashgupta2035/cross-language-code-equivalence.git
cd cross-language-code-equivalence
```

### 2. Create an environment

Python 3.10 or newer is recommended. Install the libraries used by the notebooks:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install torch transformers datasets peft accelerate scikit-learn pandas numpy jupyter
```

For a Kaggle run, the standard Kaggle image already provides most scientific Python dependencies. Confirm that the installed `transformers` version supports the v5 Trainer API used in the notebook, or update it before training:

```bash
pip install --upgrade "transformers>=5.0" peft datasets accelerate
```

### 3. Authenticate with Hugging Face

Gemma 3 is a gated model. Before downloading it:

1. Create or sign in to a [Hugging Face account](https://huggingface.co/join).
2. Open the [`google/gemma-3-1b-it` model page](https://huggingface.co/google/gemma-3-1b-it).
3. Accept Google's usage license and request/enable access if prompted.
4. Create a read-access user token at [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens).
5. Authenticate without placing the token in source code:

```bash
huggingface-cli login
```

Alternatively, set `HF_TOKEN` in the environment or add it as a Kaggle secret. Never commit the token to the repository or paste it into a notebook cell that will be shared.

## Usage

### Training in Kaggle

1. Create a Kaggle notebook for the [IITG.ai Code Semantics Similarity Challenge](https://www.kaggle.com/competitions/iitg-ai-code-semantics-similarity-challenge).
2. Add the competition dataset and enable Internet access.
3. Select a GPU accelerator and add `HF_TOKEN` under Kaggle Secrets.
4. Upload or open `tunequest_gpu.ipynb`.
5. Run the notebook from top to bottom.

The GPU notebook loads `train_small.jsonl`, evaluates the two baselines, creates chat-templated Gemma inputs, applies LoRA, trains for one epoch by default, evaluates F1/ROC-AUC, and writes the best adapter/tokenizer checkpoint under the notebook's working directory. Adjust `MAX_TRAIN_SAMPLES`, batch size, and sequence length only after confirming that the selected accelerator has enough memory.

For TPU execution, use `tunequest_tpu.ipynb` on a Kaggle TPU v5e-8 runtime. It uses PyTorch/XLA and launches training across TPU cores.

### Local or notebook inference

After training, inference uses the same preprocessing path as training: independently clip both snippets, build the prompt, apply Gemma's chat template, tokenize, and take the positive-class probability from the sequence-classification logits.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE_MODEL = "google/gemma-3-1b-it"
ADAPTER_DIR = "gemma_clone_model/best"  # produced by the training notebook

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

def predict_equivalence(func1: str, func2: str) -> tuple[bool, float]:
    prompt = (
        "Compare the following two code snippets and determine whether they are "
        "semantically equivalent (same logic / behavior), ignoring differences "
        "in variable names, formatting, coding style, or structure.\n\n"
        f"### Code A:\n{func1[:1600]}\n\n"
        f"### Code B:\n{func2[:1600]}"
    )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
    probability = float(probabilities[1])
    return probability >= 0.5, probability
```

For competition submission, run the final inference section in the notebook so predictions are aligned with `sample_submission.csv` and saved in the expected format.

## Future work

- Train on the full **8.44M-pair** corpus with distributed or staged training.
- Compare Gemma 3 with another permitted compact model, such as **Llama 3.2 1B**.
- Add symmetry augmentation by training on both `(func1, func2)` and `(func2, func1)`.
- Evaluate language-aware normalization and AST/control-flow features alongside learned representations.
- Tune the decision threshold using cross-validation rather than a single validation split.
- Calibrate probabilities and report confidence intervals across multiple random seeds.
- Explore hard-negative mining: prioritize syntactically similar pairs with different behavior.
- Package preprocessing and inference into a small command-line or API interface.

## Reproducibility notes

- Set a fixed random seed for Python, NumPy, and PyTorch.
- Keep the preprocessing function identical between validation and test inference.
- Record the sampled training rows, model revision, tokenizer revision, accelerator type, and decision threshold for each experiment.
- Treat the validation metrics above as an experiment result, not a guarantee of leaderboard performance; the competition uses a hidden test set and a private leaderboard split.

## License

No license file is currently included in this repository. Add a license before distributing or reusing the project beyond personal evaluation.
