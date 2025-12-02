# Sentinel Integrate

### Embedding Alignment Into Model Weights

> **Layer 2: Training-Level Safety**

[![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Status: In Development

This project is part of the Sentinel safety framework. It represents **Layer 2** (Training) of our three-layer approach.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SENTINEL — THREE LAYERS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 1: INFERENCE       "Guardrails on existing models"                   │
│  └── sentinel              Available now                                    │
│                                                                             │
│  Layer 2: TRAINING        "Safety embedded during fine-tuning"              │
│  └── sentinel-integrate    In Development ← You are here                    │
│                                                                             │
│  Layer 3: FOUNDATION      "Values from pre-training"                        │
│  └── sentinel-essence      Future                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Motivation

AI alignment research often focuses on either theoretical frameworks or closed-source implementations. We believe alignment should be:

1. **Practical** — Working code, not just papers
2. **Open** — Reproducible by anyone
3. **Validated** — Tested before deployed
4. **Accessible** — Available to all developers

The problem with prompt-based alignment (system prompts, constitutional AI prompts) is that it:
- Adds latency (more tokens per request)
- Can be bypassed through jailbreaks
- Must be applied every request
- Consumes context window

**Fine-tuning solves these problems** by embedding alignment directly into model weights — making it permanent, efficient, and more robust.

## Strategy: Validation-First

Unlike approaches that start with pre-training or theory, we validate at each layer before going deeper:

### Phase 1: VALIDATE (Prompt Layer) ✅
Tested alignment principles at the prompt layer using the [Sentinel Seed](https://github.com/sentinel-seed/sentinel):
- **97.6% average safety** across 6 models, 4 benchmarks
- Validated THSP Protocol (Truth, Harm, Scope, Purpose)
- Discovered PURPOSE gate as critical for agent/robot safety
- Established baseline metrics for fine-tuning

### Phase 2: INTEGRATE (Fine-tuning) ← Current
Take validated principles and embed them into model weights:
- Use proven alignment patterns from Phase 1
- Fine-tune open-source models (Llama, Mistral, Qwen)
- Maintain utility while improving safety
- Publish open weights on Hugging Face

### Phase 3: EMBED (Pre-training) → Future
Eventually influence model behavior from the start:
- Alignment-aware pre-training datasets
- Training methodology for aligned foundation models
- Collaboration with model providers

## The THSP Protocol

Our alignment framework is built on **four gates** that every response must pass:

| Gate | Question | Purpose |
|------|----------|---------|
| **TRUTH** | "Is this factually correct?" | Prevent misinformation |
| **HARM** | "Does this cause harm?" | Prevent dangerous outputs |
| **SCOPE** | "Is this within appropriate limits?" | Prevent overreach |
| **PURPOSE** | "Does this serve a legitimate benefit?" | Require teleological justification |

All four gates must pass for an action to proceed. The key insight of THSP is that **the absence of harm is not sufficient** — there must be genuine purpose that benefits those being served.

This creates a simple, auditable framework that can be embedded into model behavior. Validated at inference layer with **97.6% average safety** across 6 models and 4 benchmarks.

## Project Structure

```
sentinel-integrate/
├── datasets/                       # Training data
│   ├── thsp-alignment/            # THSP Protocol examples
│   │   ├── truth_gate.jsonl       # Factual accuracy
│   │   ├── harm_gate.jsonl        # Safety boundaries
│   │   ├── scope_gate.jsonl       # Appropriate limits
│   │   └── purpose_gate.jsonl     # Teleological justification
│   ├── refusals/                  # Correct refusal patterns
│   ├── utility/                   # Helpful responses
│   └── prepared/                  # Processed training data
│
├── training/
│   ├── configs/                   # Model-specific configurations
│   │   └── llama-3.2-1b.yaml     # Llama 3.2 1B config
│   └── scripts/
│       ├── train.py              # Main training script
│       ├── merge.py              # Merge LoRA adapters
│       └── prepare_dataset.py    # Dataset preparation
│
├── evaluation/
│   ├── benchmarks/               # Test datasets
│   │   ├── safety_test.jsonl    # Safety evaluation (20 cases)
│   │   └── utility_test.jsonl   # Utility evaluation (20 cases)
│   ├── evaluate.py              # Evaluation script
│   ├── compare_models.py        # Base vs fine-tuned comparison
│   └── results/                 # Evaluation results
│
├── models/
│   ├── checkpoints/             # Training checkpoints
│   └── merged/                  # Final merged models
│
└── docs/
    └── METHODOLOGY.md           # Detailed methodology
```

## Development Plan

### Current Status: Infrastructure Complete

- [x] Project structure and repository setup
- [x] Training infrastructure (LoRA/QLoRA)
- [x] Dataset schema and initial examples (~120)
- [x] Evaluation pipeline
- [x] Methodology documentation

### Next Steps

| Phase | Task | Status |
|-------|------|--------|
| **Data** | Expand to 1000+ examples per category | Planned |
| **Data** | Synthetic data generation pipeline | Planned |
| **Training** | First fine-tune: Llama 3.2 1B | Planned |
| **Eval** | Benchmark on SafeAgentBench, HarmBench | Planned |
| **Release** | Publish first model on Hugging Face | Planned |
| **Training** | Fine-tune Mistral 7B | Planned |
| **Training** | Fine-tune Qwen 2.5 7B | Planned |

### Target Models

| Model | Parameters | Why |
|-------|------------|-----|
| **Llama 3.2** | 1B, 3B | Small, fast iteration, good baseline |
| **Mistral** | 7B | Strong performance, popular |
| **Qwen 2.5** | 7B, 14B | Multilingual, growing ecosystem |
| **Phi-3** | 3.8B | Efficient, Microsoft backing |

## Technical Approach

### Why LoRA/QLoRA?

We use parameter-efficient fine-tuning (PEFT) because:

1. **Accessibility** — Train on consumer GPUs (24GB VRAM)
2. **Preservation** — Base model capabilities stay intact
3. **Distribution** — Small adapter files (~100MB vs 14GB)
4. **Composition** — Multiple adapters can be combined

### Training Configuration

```yaml
# Key hyperparameters (Llama 3.2 1B example)
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_quant_type: nf4

training:
  epochs: 3
  learning_rate: 2e-4
  batch_size: 4
  gradient_accumulation_steps: 4
```

### Evaluation Metrics

Every model is evaluated on:

| Metric | Target | Description |
|--------|--------|-------------|
| Safety Refusal Rate | >90% | Harmful requests correctly refused |
| Utility Rate | >95% | Benign requests helpfully answered |
| False Refusal Rate | <5% | Benign requests incorrectly refused |

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended: 24GB VRAM for 7B models)
- 50GB disk space for models

### Installation

```bash
# Clone the repository
git clone https://github.com/sentinel-seed/Sentinel-Integrate.git
cd Sentinel-Integrate

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Prepare Dataset

```bash
# Combine all datasets into train/val splits
python training/scripts/prepare_dataset.py \
  --dataset-dir datasets \
  --output datasets/prepared/sentinel.jsonl
```

### Train a Model

```bash
# Fine-tune Llama 3.2 1B
python training/scripts/train.py \
  --config training/configs/llama-3.2-1b.yaml
```

### Evaluate

```bash
# Evaluate a fine-tuned model
python evaluation/evaluate.py \
  --model models/merged/llama-3.2-1b-sentinel \
  --safety-tests evaluation/benchmarks/safety_test.jsonl \
  --utility-tests evaluation/benchmarks/utility_test.jsonl

# Compare base vs fine-tuned
python evaluation/compare_models.py \
  --base-model meta-llama/Llama-3.2-1B-Instruct \
  --finetuned-model models/merged/llama-3.2-1b-sentinel
```

## Principles

### What We Optimize For

1. **Safety without sacrificing utility** — A model that refuses everything is useless
2. **Measurable improvement** — If we can't measure it, we don't claim it
3. **Reproducibility** — Anyone can verify our results
4. **Transparency** — All data, code, and models are open

### What We Avoid

- Cherry-picking results
- Benchmark gaming
- Claims without evidence
- Closed-source "trust us" approaches

## Related Projects

| Project | Layer | Status |
|---------|-------|--------|
| [Sentinel](https://github.com/sentinel-seed/sentinel) | Inference | ✅ Available |
| [Sentinel-Integrate](https://github.com/sentinel-seed/Sentinel-Integrate) | Training | In Development |
| [Sentinel-Essence](https://github.com/sentinel-seed/sentinel-essence) | Foundation | Future |
| [Sentinel Platform](https://sentinelseed.dev) | — | ✅ Live |

## Credits

This work was inspired by the work developed by Foundation Labs (Gabriel). Check it out at <https://github.com/davfd>.

## Contributing

We welcome contributions! Areas where help is needed:

- **Dataset expansion** — More training examples
- **Evaluation** — Additional benchmark implementations
- **Testing** — Run training on different hardware
- **Documentation** — Tutorials and guides

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — Use freely, improve openly.

---

<p align="center">
  <strong>Sentinel</strong> — Practical AI Alignment for Developers
  <br>
  <a href="https://sentinelseed.dev">Website</a> •
  <a href="https://github.com/sentinel-seed/sentinel">Phase 1</a> •
  <a href="https://twitter.com/sentinel_Seed">Twitter</a>
</p>

