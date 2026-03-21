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
