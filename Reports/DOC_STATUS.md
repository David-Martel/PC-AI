# DOC_STATUS

Generated: 2026-02-22 09:42:16

## Counts
- @status: 1
- DEPRECATED: 2
- FIXME: 2
- INCOMPLETE: 2
- TODO: 30

## Matches
- AGENTS.md:201 ## Known gaps / TODOs
- Deploy\rust-functiongemma-train\TODO.md:1 # TODO - Rust FunctionGemma (PC_AI)
- Deploy\rust-functiongemma-train\TODO.md:3 This TODO captures the minimum work required to reach feature parity with
- docs\plans\2026-01-30-pcai-inference-dual-backend.md:788 prompt_tokens: 0, // TODO: Implement token counting
- docs\plans\2026-01-30-pcai-inference-dual-backend.md:937 // TODO: Implement actual model loading with llama-cpp-2
- docs\plans\2026-01-30-pcai-inference-dual-backend.md:948 architecture: "unknown".into(), // TODO: Read from GGUF metadata
- docs\plans\2026-01-30-pcai-inference-dual-backend.md:949 quantization: None, // TODO: Read from GGUF metadata
- docs\plans\2026-01-30-pcai-inference-dual-backend.md:980 // TODO: Implement actual generation with llama-cpp-2
- docs\plans\2026-01-30-pcai-inference-dual-backend.md:1007 // TODO: Implement streaming generation
- docs\plans\2026-01-30-pcai-inference-dual-backend.md:1142 // TODO: Implement actual mistralrs model loading
- docs\plans\2026-01-30-pcai-inference-dual-backend.md:1191 // TODO: Implement actual mistralrs generation
- docs\plans\2026-01-30-pcai-inference-dual-backend.md:1224 // TODO: Implement streaming with mistralrs
- rules\doc-status.yml:7 - pattern: "TODO"
- rules\doc-status.yml:8 - pattern: "FIXME"
- rules\doc-status.yml:9 - pattern: "INCOMPLETE"
- rules\doc-status.yml:10 - pattern: "DEPRECATED"
- Modules\PC-AI.Acceleration\Public\Search-ContentFast.ps1:40 Search-ContentFast -Path "." -LiteralPattern "TODO:" -Context 2
- Modules\PC-AI.Acceleration\Public\Search-ContentFast.ps1:41 Finds TODO comments with context
- TODO.md:1 # TODO
- Deploy\rust-functiongemma-train\src\trainer.rs:712 optimizer_state: vec![], // TODO: Save optimizer state if needed
- Deploy\rust-functiongemma-train\src\trainer.rs:713 rng_state: None, // TODO: Save RNG state for reproducibility
- Tools\Invoke-DocPipeline.ps1:142 # Step 1: Generate DOC_STATUS report (TODO/FIXME/DEPRECATED markers)
- Tools\update-doc-status.ps1:8 Scans the repo for TODO/FIXME/INCOMPLETE/@status/DEPRECATED markers and writes:
- Tools\update-doc-status.ps1:67 $markers = 'TODO|FIXME|INCOMPLETE|@status|DEPRECATED'
- Tools\update-doc-status.ps1:158 if ($_.Match -match 'TODO') { 'TODO' }
- Tools\update-doc-status.ps1:159 elseif ($_.Match -match 'FIXME') { 'FIXME' }
- Tools\update-doc-status.ps1:160 elseif ($_.Match -match 'INCOMPLETE') { 'INCOMPLETE' }
- Tools\update-doc-status.ps1:161 elseif ($_.Match -match '@status') { '@status' }
- Tools\update-doc-status.ps1:162 elseif ($_.Match -match 'DEPRECATED') { 'DEPRECATED' }
- Deploy\rust-functiongemma-train\README.md:20 - FunctionGemma model inference with tool-call parsing (TODO)
- Deploy\rust-functiongemma-train\README.md:26 - LoRA/QLoRA fine-tuning (TODO)
- Deploy\rust-functiongemma-train\README.md:27 - Eval harness + regression checks (TODO)
- Deploy\rust-functiongemma-train\README.md:149 with the Python pipeline and has several TODOs (see TODO.md).
- Deploy\rust-functiongemma-train\examples\checkpoint_usage.md:75 // TODO: Restore optimizer state and RNG
- Deploy\rust-functiongemma-train\examples\checkpoint_usage.md:105 optimizer_state: vec![0.1, 0.2, 0.3], // TODO: Serialize actual optimizer
- Deploy\rust-functiongemma-train\examples\checkpoint_usage.md:106 rng_state: Some(rand::random()), // TODO: Get actual RNG state
- Native\pcai_core\pcai_inference\src\backends\mistralrs.rs:234 // TODO: Implement a custom RequestLike to support full sampling control.

