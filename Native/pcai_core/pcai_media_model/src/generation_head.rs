// Generation head for Janus-Pro.
//
// This module will project the LLM's hidden states (e.g. 4096-dim for 7B)
// onto the image vocabulary (16384 tokens) via a linear layer followed by
// log-softmax, enabling autoregressive image token generation.
//
// Planned components:
// - JanusGenHead: wraps a candle_nn::Linear projection
// - forward(): takes hidden states [B, seq, hidden] → logits [B, seq, vocab]
//
// Status: stub — implementation pending.
