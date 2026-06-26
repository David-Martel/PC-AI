# Workstation cleanup lane - 2026-06-26

## Scope

Inventory and safe cleanup for `C:\`, `T:\`, `F:\`, and `W:\` with emphasis on
preserving active work-in-progress caches, Docker runner state, cloud-sync roots,
and VHD rollback paths.

## Actions completed

- Fixed `PC-AI.Performance` import failure by disabling stale native type-data
  registration and adding a clear `PcaiNative.PerformanceModule` availability
  guard in `Get-PcaiDiskUsage`.
- Validated repeated import and native disk usage in both source and release
  module layouts.
- Created Google Drive archive folders with `rclone`:
  `gdrive-personal:Archives/Workstation Cleanup/2026-06-26-cleanup-lane/`.
- Ran conservative Docker BuildKit cleanup:
  `docker builder prune --all --filter until=168h --force`.
- Removed old public/base Ollama models by name:
  `nemotron-3-nano:latest`, `qwen3:30b`, `qwen3:14b`,
  `gpt-oss:20b`, `gpt-oss-122k:latest`, `gemma3:12b`,
  `gemma3:12b-it-qat`, `gemma4:e4b`, `orieg/gemma3-tools:12b`,
  `deepseek-r1:7b`, `deepseek-r1:8b`, `mistral:latest`, and
  `llama3.1:latest`.
- Started `uv cache clean` and stale user temp cleanup; both were CPU-active
  after several minutes and were stopped to avoid leaving background cleanup
  running unattended.
- Ran `npm cache verify` and `npm cache clean --force`; verification garbage
  collected 754 entries, about 1.03 GB.

## Measured result

Before cleanup, `C:` had about 234 GB free. After the completed cleanup work,
`C:` had `382,521,475,072` bytes free, about 356 GB.

Final measured selected surfaces:

| Surface | Final size / state |
| --- | ---: |
| `C:\Users\david\.ollama` | 20.2 GB |
| `C:\Users\david\AppData\Local\uv\cache` | 12.6 GB |
| Docker images | 75.1 GB total, 25.03 GB reclaimable |
| Docker containers | 5.013 GB total, near-zero reclaimable |
| Docker volumes | 21.27 GB total, 9.105 GB reclaimable |
| Docker BuildKit cache | 29.79 GB total, 18.34 GB reclaimable |
| `T:\vm\docker\wsl\disk` | 167.7 GB |
| `T:\vm\docker-backup` | 167.9 GB |
| `T:\vm\docker\disk` | 86.3 GB |
| `T:\vm\wsl-vhdx` | 677.3 GB |
| `T:\vm\shared-dev.vhdx` | 549.8 GB |
| `T:\vm\cloud-cache-disk.vhdx` | 1.14 TB |

## Deferred cleanup targets

- Docker BuildKit cache newer than 7 days: still 18.34 GB reclaimable, but likely
  tied to active runner and recent build work.
- Docker images with no containers, especially
  `ghcr.io/rust-cross/cargo-zigbuild:latest`, `vigil-friction:ubuntu`,
  `vigilclarius-rt-build:{amd64,arm64}`, `rust:latest`, and old dangling
  images. Remove only after current runner/build consumers are checked.
- Docker detached volumes: broad `docker volume prune` is high-risk because
  detached named volumes include model and database caches.
- `T:\vm\docker-backup\DockerDesktopWSL-old\disk\docker_data.vhdx`: strong
  archive/delete candidate after backup and rollback check.
- VSS shadow storage on `C:`: used 374 GB with 409 GB maximum. Retarget to a new
  restore point and reduce/delete old restore points only after confirming the
  desired rollback policy.
- `T:\UniversalMac_26.2_25C5037j_Restore.ipsw`: 18.7 GB archive/delete
  candidate.

## Google Drive archival gate

Use the dated Drive folder as the deletion gate:

```text
Archives/
  Workstation Cleanup/
    2026-06-26-cleanup-lane/
      00-manifests/
      01-docker-vhds/
      02-apple-ipsw/
      03-repo-bundles/
      04-reports/
      90-upload-logs/
```

For large opaque files, upload raw files with hashes instead of recompressing:

- `.vhdx` / `.vhd`: upload raw, preferably with `rclone --checksum`.
- `.ipsw`: upload raw; it is already compressed and signed.
- Git repositories: use `git bundle create <repo>.bundle --all`, then separately
  archive uncommitted or untracked work if needed.

Deletion gate:

1. Record source path, size, SHA-256, MD5, modified time, and restore notes.
2. Upload to Drive.
3. Verify Drive metadata or `rclone check`.
4. Only then delete or move the local source.

## WSL, VHD, and model runtime consolidation proposal

Use a two-tier storage model:

1. Windows-visible immutable model store:
   - `T:\Models\huggingface`
   - `T:\Models\ollama`
   - `T:\Models\vllm`
   - `T:\Models\stable-diffusion`
2. Linux-native runtime scratch/cache in ext4-backed WSL storage:
   - `T:\vm\wsl-vhdx\model-runtime\ext4.vhdx` or an equivalent WSL-mounted
     ext4 VHD.

For Docker/vLLM, prefer explicit bind mounts over opaque named volumes:

```yaml
volumes:
  - T:/Models/huggingface:/root/.cache/huggingface:ro
  - T:/Models/vllm:/root/.cache/vllm
```

If Linux filesystem performance matters more than Windows direct access, keep
runtime caches inside an ext4 VHD and expose only immutable model weights from
`T:\Models`. If Windows tools also need direct model access, keep weights on
`T:\Models` and put only scratch/cache on ext4.

Next implementation steps:

1. Inventory actual model consumers: Ollama, vLLM Compose, WSL distros, Docker
   named volumes, and Python/Hugging Face caches.
2. Add explicit model-cache environment variables to Compose/WSL launchers:
   `HF_HOME`, `TRANSFORMERS_CACHE`, `VLLM_CACHE_ROOT`, and `OLLAMA_MODELS`.
3. Move one runtime at a time, validate startup and inference, then compact the
   old VHD after the old cache is removed.
4. Keep rollback metadata in the Drive cleanup manifest.
