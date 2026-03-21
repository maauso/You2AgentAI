# Fine-Tuning Module

This is the most performance-critical part of the project. We transform a **Base** model into a specialized chat agent by leveraging **QLoRA** and modern hardware optimizations.

## 🛠️ QLoRA (4-bit Quantization)

To fit a 7B parameter model like Mistral-7B-v0.3 into consumer GPUs (like the **RTX 4090**) while maintaining a large context window, we use 4-bit quantization via `bitsandbytes`.
- **VRAM target**: ~5GB in idle.
- **Precision**: Uses Normalized Float 4 (NF4) and Double Quantization to minimize accuracy loss.

## 🧠 High-Capacity LoRA

Since we are training from a **Base** model (which lacks any inherent chat structure), we increase the LoRA rank (`r`) to provide more "learning capacity."
- **Rank (`r`)**: 16
- **Alpha**: 32
- **Target Modules**: Covers all major projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) to ensure the model learns the ChatML format effectively.

## ⚡ Hardware Optimizations (RTX 4090)

We leverage the latest engineering techniques for the Ada Lovelace architecture:
- **Bfloat16 (BF16)**: Native support to prevent gradient collapse and improve numerical stability compared to FP16.
- **Flash Attention 2**: Speeds up the attention mechanism by up to 200%, significantly reducing training time.
- **Paged AdamW 8-bit**: An optimized optimizer that further reduces VRAM usage by paging memory to the CPU when needed.

## 📈 Monitoring & Convergence

Integrated with **Weights & Biases (WandB)** for real-time tracking of:
- **Loss Curve**: Should show clear descent as the model adopts ChatML.
- **Learning Rate**: Managed with a **Cosine Scheduler** for a smooth convergence.

## 🏃 Running the Fine-Tuning

Ensure your tokenized dataset is ready, then run:

```bash
python 3_FineTuning/fineTuning.py
```

The final output will be a **Peft Adapter** (`mistral-7b-chatml-adapter`) that can be loaded for inference.
