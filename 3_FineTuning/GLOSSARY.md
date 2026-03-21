# Fine-Tuning Glossary (Simple + Internal)

This glossary explains key fine-tuning concepts in two layers:
- **Simple explanation**: what you notice in practice.
- **What it does internally**: the technical mechanism.

## `bitsandbytes`

**Simple explanation**

`bitsandbytes` helps you run large models with much less GPU memory.

In this project, it enables 4-bit quantization so a 7B model can fit on consumer hardware.

**What it does internally**

It stores weights in low precision (for example 4-bit) and uses optimized CUDA kernels for quantized math.
That reduces VRAM use while keeping most of the model quality.

## `QLoRA`

**Simple explanation**

QLoRA is a way to fine-tune a big model cheaply.

Instead of retraining the whole model, you keep the base model compressed and only train small adapter layers.

**What it does internally**

It combines low-bit quantization for the frozen base model with LoRA adapters for the trainable part.
This keeps memory use low while still allowing the model to learn new behavior.

## `NF4`

**Simple explanation**

NF4 is a smart 4-bit format for storing model weights.

You can think of it as a compact box that wastes less space than normal storage.

**What it does internally**

NF4 stands for NormalFloat4.
It is designed for weight values that roughly follow a normal distribution, so it usually preserves model quality better than simpler 4-bit formats.

## `Double Quantization`

**Simple explanation**

Double quantization is "compression for the compression settings".

It saves a bit more VRAM on top of normal 4-bit loading.

**What it does internally**

When weights are quantized, extra scaling values are needed to decode them.
Double quantization compresses those scaling values too, which reduces memory overhead.

## `LoRA rank` (`r`)

**Simple explanation**

Rank controls the size of the LoRA adapter's learning capacity.

- Lower rank: less memory, less adaptation power
- Higher rank: more memory, more adaptation power

**What it does internally**

LoRA replaces a full weight update with two small matrices, typically $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times d}$.
The rank $r$ is the bottleneck dimension that limits how much change the adapter can represent.

## `Alpha` (`lora_alpha`)

**Simple explanation**

Alpha controls how strongly LoRA updates affect the base model.

Short version:
- Rank = capacity
- Alpha = strength

**What it does internally**

LoRA applies a scaled update of roughly:

$$
\Delta W \propto \frac{\alpha}{r}BA
$$

So increasing alpha increases the effective magnitude of adapter updates.

## `Projections` (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.)

**Simple explanation**

Projections are core linear layers in the transformer.
Targeting them means LoRA edits the most important places where information is transformed.

**What it does internally**

In attention:
- `q_proj`: creates queries
- `k_proj`: creates keys
- `v_proj`: creates values
- `o_proj`: maps attention output back to model space

In feed-forward blocks:
- `gate_proj`, `up_proj`, `down_proj` control expansion, gating, and compression of hidden states.

Applying LoRA on these layers gives high leverage per trainable parameter.

## `BF16` vs `FP16`

**Simple explanation**

Both are 16-bit formats that save memory and speed up training.

- `FP16`: can be less stable
- `BF16`: usually more stable on modern GPUs

**What it does internally**

BF16 keeps the wider exponent range of FP32, which greatly reduces overflow/underflow issues.
FP16 has a smaller exponent range, so gradients and activations can be numerically fragile.

## `Flash Attention`

**Simple explanation**

Flash Attention is a faster and more memory-efficient attention implementation.

**What it does internally**

It computes attention in tiled blocks and avoids storing large intermediate matrices in full precision.
This cuts memory traffic and improves throughput, especially with long context windows.

## `SDPA`

**Simple explanation**

SDPA is PyTorch's built-in attention engine.

If Flash Attention is not available, SDPA is the safe backup.

**What it does internally**

SDPA means Scaled Dot-Product Attention.
It uses PyTorch's optimized attention kernels instead of a custom external package.

## `device_map="auto"`

**Simple explanation**

This tells the library: "you decide where model parts should go".

That is useful when the model is too large to place manually without mistakes.

**What it does internally**

Transformers inspects available hardware and automatically places layers on the right device, usually one or more GPUs and sometimes CPU if needed.

## `AdamW`

**Simple explanation**

AdamW is the optimizer, the rule that updates weights after each batch.

**What it does internally**

It combines adaptive moments (Adam) with decoupled weight decay.
Decoupling weight decay from gradient updates usually gives better regularization behavior than classic Adam + L2.

## `Cosine Scheduler`

**Simple explanation**

A cosine scheduler lowers the learning rate smoothly over time.

Typical effect:
- bigger steps early
- smaller, safer steps later

**What it does internally**

It applies a cosine-shaped decay curve to the learning rate across training steps.
This often helps convergence by reducing update noise near the end of training.

## `Data Collator`

**Simple explanation**

The data collator is the part that packs training examples into a batch.

It makes sure all examples in that batch have the same shape so the GPU can process them together.

**What it does internally**

It pads sequences to a common length, builds tensors, and prepares the exact keys the model expects, such as `input_ids`, `attention_mask`, and `labels`.

## `label_pad_token_id = -100`

**Simple explanation**

This tells the loss function to ignore the fake padding positions.

In other words: "these empty filler tokens do not count when scoring the model".

**What it does internally**

PyTorch loss functions commonly ignore targets with value `-100`.
That means padded label positions are skipped during loss computation and do not affect gradients.

## `Adapter`

**Simple explanation**

An adapter is a small extra set of trained weights that teaches the base model a new behavior.

It is much smaller than saving the whole model again.

**What it does internally**

With LoRA, the adapter contains the low-rank update matrices that are added to selected base-model layers at inference time.
Loading the base model plus the adapter recreates the fine-tuned behavior.
