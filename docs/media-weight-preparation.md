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

Both commands run as written from the repository root in a clean environment:
the download needs no Bitwarden login (`-SkipAuth`, the DeepSeek repos are
public) and the conversion pulls its own Python dependencies through `uv`, so
only `pwsh` and `uv` need to be installed. The same strings are emitted by the
loader's refusal message; `test_preparation_commands_are_runnable_and_match_the_documentation`
in [`hub.rs`](../Native/pcai_core/pcai_media/src/hub.rs) fails if this page and
the code drift apart.

```powershell
# 1. Download config.json, the tokenizer and pytorch_model*.bin into Models/Janus-Pro-1B.
pwsh Tools/Get-JanusProModel.ps1 -ModelId deepseek-ai/Janus-Pro-1B -SkipAuth

# 2. Convert to model.safetensors in the same directory.
uv run --no-project --with torch --with safetensors --with numpy python Tools/convert-sharded-weights.py Models/Janus-Pro-1B
```

For the 7B model use `deepseek-ai/Janus-Pro-7B` and `Models/Janus-Pro-7B`.
[`Get-JanusProModel.ps1`](../Tools/Get-JanusProModel.ps1) also accepts `-OutputDir` and
points `Config/pcai-media.json` at the result.
[`convert-sharded-weights.py`](../Tools/convert-sharded-weights.py) takes the model
directory as its only required argument (`--output <name>` is optional),
handles a single `pytorch_model.bin` (1B) and sharded `pytorch_model-*-of-*.bin`
(7B), and exits without work if `model.safetensors` already exists.

## Point the loader at it

- Pass the directory: `pcai-media --model Models/Janus-Pro-1B`, or set `model`
  in `Config/pcai-media.json` (it already uses `Models/...` paths).
- Or keep the default model id. The two ids `deepseek-ai/Janus-Pro-1B` and
  `deepseek-ai/Janus-Pro-7B` resolve to `Models/<name>/` (relative to the
  working directory) or `$PCAI_MODELS_DIR/<name>/` when that directory has
  `config.json`, `tokenizer.json` and SafeTensors weights; otherwise only
  config and tokenizer are downloaded and the load fails with the commands
  above.

## Resolution rules

- An existing directory is used as it is.
- A model written as a filesystem path (absolute, drive-qualified, starting
  with `.` or `~`, containing a backslash, or with more than one `/`) that does
  not exist is an error naming that path. It never falls back to a prepared
  or downloaded model.
- Only the two exact ids above resolve to a prepared copy. The match is
  case-sensitive and exact: `thirdparty/Janus-Pro-1B`,
  `DeepSeek-AI/Janus-Pro-1B` and `deepseek-ai/janus-pro-1b` are ordinary Hub
  ids that go to the download path and then to the SafeTensors refusal, so a
  similarly named local directory can never stand in for a different model.

## Migration

Nothing is needed for a directory that already contains `model.safetensors`.
Anyone who relied on the old behaviour, loading a checkpoint whose weights were
absent or partial and silently running with random tensors, now gets an error
that names the missing tensors. Fix the checkpoint rather than the check.

## Not verified here

No real Janus checkpoint was downloaded or converted for this change. The
converter was exercised on tiny synthetic `.bin` fixtures only, and a complete
converted checkpoint has not been run through the loader's header preflight.
