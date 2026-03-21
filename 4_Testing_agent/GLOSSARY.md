# Inference Glossary (Simple + Internal)

This glossary explains generation parameters at two levels:
- **Simple explanation**: what effect you will notice in the response.
- **What it does internally**: the technical logic behind it.

## `max_new_tokens`

**Simple explanation**

This is the maximum response length.
If you set it to `256`, the response cannot grow beyond 256 new tokens.

**What it does internally**

The model generates one token at a time in a loop. This parameter stops that loop when the max number is reached, even if the end token has not appeared.

## `do_sample`

**Simple explanation**

This enables "creative" mode.
- `False`: it almost always picks the most likely option (more rigid).
- `True`: it allows variation (more natural, less repetitive).

**What it does internally**

With `do_sample=False`, decoding is deterministic (argmax/greedy).
With `do_sample=True`, the model samples a token from a probability distribution.

## `temperature`

**Simple explanation**

This is the creativity "thermostat".
- Low (`0.2 - 0.6`): more conservative responses.
- Medium (`0.7 - 0.9`): balance between coherence and variation.
- High (`1.0+`): riskier responses.

**What it does internally**

Before converting logits to probabilities, logits are divided by `temperature`:

$$
	ext{adjusted logits} = \frac{\text{logits}}{T}
$$

If $T<1$, the distribution becomes sharper (fewer realistic options).
If $T>1$, the distribution becomes flatter (more low-probability options are allowed).

## `top_p` (Nucleus Sampling)

**Simple explanation**

This means: "sample only from the most likely options whose total probability adds up to `p`".
With `top_p=0.9`, the model discards the unlikely tail of options.

**What it does internally**

1. Sorts tokens by probability (highest to lowest).
2. Accumulates probabilities until reaching `p`.
3. Removes the rest of the tokens.
4. Samples inside that reduced subset.

This reduces nonsense outputs without making text fully rigid.

## `stopping_criteria`

**Simple explanation**

This is the stop condition.
In this project, generation stops when `<|im_end|>` appears.

**What it does internally**

After each generated token, Transformers calls the criterion.
If it returns `True`, generation stops immediately.

## `pad_token_id`

**Simple explanation**

Defines which token is used as padding when sequence lengths need alignment.
Here we use `<|im_end|>` to stay consistent with ChatML.

**What it does internally**

It is used in batching/alignment operations and helps avoid warnings or ambiguous behavior when an explicit padding token is missing.

## How to read `temperature` + `top_p` together

Think of it this way:
- `temperature` decides **how much freedom** there is inside the options.
- `top_p` decides **which options are allowed** in the first place.

Current setup (`temperature=0.8`, `top_p=0.9`):
- Keeps enough variation to sound natural.
- Removes very unlikely options to preserve coherence.

It is a common setup for technical chat: creative, but controlled.
