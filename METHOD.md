Decode‑time (evaluation‑only, no training)

Given LM with hidden state h_t and logits_t for step t:

1. logits_ref_t = W_ref · h_t     # tiny projection head (|W_ref| ≪ LM)
2. p_ref_t = softmax(logits_ref_t)
3. logits′_t = (1 − α) · logits_t + α · log(p_ref_t + ε)   # bias toward identity prior
4. p_t = softmax(logits′_t); sample/argmax next token

Training‑time (optional regularizer)

Loss_total = Loss_task + α · KL( softmax(logits_t) ∥ softmax(W_ref · h_t) )

Notes
- α controls anticipatory channel strength.  
- W_ref can be shared across layers or tied to a pooled representation; keep <~16 MB total.
- For extreme constraints, W_ref can be low‑rank or quantized.
- Hedging detection at eval: simple lexicons (EN/FR) for “hedge/apology” tokens per output.
