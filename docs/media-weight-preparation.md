# Preparing Janus-Pro weights for pcai-media

`pcai-media` loads only complete SafeTensors weights. It never reads PyTorch
`.bin` checkpoints and it refuses to run with random weights (see
`GenerationPipeline::load` in
[`generate.rs`](../Native/pcai_core/pcai_media/src/generate.rs) and the typed
`IncompleteWeights` error in
[`hub.rs`](../Native/pcai_core/pcai_media/src/hub.rs)). Both published
repositories, `deepseek-ai/Janus-Pro-1B` and `deepseek-ai/Janus-Pro-7B`,
contain only `.bin` weights (checked against the Hugging Face file listings on
2026-11-22), so a model id on its own can never load. Prepare the weights once.

## Prepare

```powershell
# 1. Download config.json, the tokenizer and pytorch_model*.bin into Models/Janus-Pro-1B.
pwsh Tools/Get-JanusProModel.ps1 -ModelId deepseek-ai/Janus-Pro-1B

# 2. Convert to model.safetensors in the same directory (needs torch, safetensors, numpy;
#    AI-Media/.venv has them, or run it with `uv run --with torch --with safetensors --with numpy`).
uv run python Tools/convert-sharded-weights.py Models/Janus-Pro-1B
```

[`Get-JanusProModel.ps1`](../Tools/Get-JanusProModel.ps1) accepts `-OutputDir` and
points `Config/pcai-media.json` at the result.
[`convert-sharded-weights.py`](../Tools/convert-sharded-weights.py) handles a
single `pytorch_model.bin` (1B) and sharded `pytorch_model-*-of-*.bin` (7B) and
exits without work if `model.safetensors` already exists.

## Point the loader at it

- Pass the directory: `pcai-media --model Models/Janus-Pro-1B`, or set `model`
  in `Config/pcai-media.json` (it already uses `Models/...` paths).
- Or keep the default model id. `deepseek-ai/Janus-Pro-1B` resolves to
  `Models/Janus-Pro-1B/` (relative to the working directory) or
  `$PCAI_MODELS_DIR/Janus-Pro-1B/` when that directory has `config.json`,
  `tokenizer.json` and SafeTensors weights; otherwise only config and tokenizer
  are downloaded and the load fails with the commands above.

## Migration

Nothing is needed for a directory that already contains `model.safetensors`.
Anyone who relied on the old behaviour, loading a checkpoint whose weights were
absent or partial and silently running with random tensors, now gets an error
that names the missing tensors. Fix the checkpoint rather than the check.

## Not verified here

No real Janus checkpoint was downloaded or converted for this change. The
converter was exercised on tiny synthetic `.bin` fixtures only, and a complete
converted checkpoint has not been run through the loader's header preflight.
