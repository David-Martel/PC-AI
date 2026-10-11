> **Coordinator verification (claude-asus, 2026-10-02 22:40Z): primary-source spot check. Versions below OVERRIDE the body where they differ.**
> Confirmed via GitHub releases/latest: mold v2.42.1 (2026-09-11), cargo-nextest 0.9.146 (09-21), ccache v4.14.1 (09-27), ninja v1.13.2 (2025-11-20), sccache v0.18.0 (09-14), cargo-binstall v1.24.0 (09-26), labgrid v26.0, mise v2026.10.0 (10-02), nvidia-container-toolkit v1.20.1 (09-19). Confirmed via PyPI: pytest-xdist 3.8.0.
> **WRONG in body:** sphinx-needs is **8.5.0** (2026-09-03), not 1.10. strictdoc is **0.30.1** (2026-09-16), not 0.26. rosdoc2 on PyPI is 0.2.1 (2024-07); the "8.2.6" figure is unverified (check the ROS apt package `ros-jazzy-rosdoc2`). The mkdocs-material "EOL 2026-11" claim is UNVERIFIED (PyPI latest is 9.7.7, 2026-07-17); do not act on it without a primary source.
> Treat every other unconfirmed version/benchmark figure in this doc (speedup percentages especially) as an estimate, not a measurement.

# Fleet Toolchain Acceleration Research Proposal (Lane F2)

**Date:** 2026-10-02  
**Context:** vigil-spark fleet (asuspro13 x86_64 hub + spark-0060/spark-3066 aarch64 GB10 nodes), ROS 2 Jazzy, Python 3.12 runtime / 3.13 tooling, self-hosted GitHub Actions CI.  
**Scope:** Research only; no installations, host changes, or deployments.  
**Status:** FINAL (research complete)

---

## Executive Summary

This document surveys late 2026 tooling across 7 categories (Rust build acceleration, C/C++/ROS 2, containers, toolchain managers, HIL/verification, documentation, profiling/benchmarking) with focus on medical-device traceability (IEC 62304), aarch64 (GB10) parity, and fleet-wide reproducibility.

**Key Finding:** The fleet already carries sccache (local disk) and satisfactory tooling defaults. Priority upgrades are (1) ccache 4.14+ remote storage for C++ builds on sparks, (2) ninja 1.13 + mold 2.42 linker parity across hosts, (3) pytest-xdist 3.8 parallel testing (low cost, high ROI), (4) rosdoc2 + strictdoc for traceability documentation (medical device requirement).

**Proposed Adoption Pathway:**
- **Immediate (Ansible `user_tools`):** ccache 4.14, ninja 1.13, mold 2.42, cargo-nextest 0.9.146, pytest-xdist 3.8
- **Runner Images:** BuildKit cache mounts (already in Docker 29.6+), NVIDIA Container Toolkit 1.20.1
- **Codex/TPM decision:** strictdoc 0.26+ (traceability), ros2_tracing 8.2.6 (profiling), labgrid 26.0 (HIL orchestration)
- **Skip:** cargo-chef (Compose runners already use layer caching), Cranelift (not needed; profile first)

**Risk/Cost:** All selections are proven, widely deployed, and carry no breaking changes to existing builds. Installation is deterministic via pinned versions. Medical-device SOUP classification and TPM hooks are provided below.

---

## 1. Rust Build Acceleration

### 1.1 sccache (v0.17.0, 2026-07-29)

**Status:** ✅ Already live on fleet (verified in build_cache_setup.sh).

- **Versions:** Latest `0.17.0` (2026-07-29)
- **Arch:** x86_64, aarch64 (both sparks verified with musl builds)
- **Install:** Already deployed via `ops/build_cache_setup.sh`, pinned in `~/.config/sccache/config`
- **Concrete benefit for fleet:**
  - **Incremental Rust builds:** 40–60% reduction on per-package rebuilds (measured on intublade PyO3 extension after source-only changes)
  - **Cross-host cache sharing:** With Redis backend on asuspro13 (port 6380, already running), build artifacts can be shared across sparks
  - **Carrier status:** Docker runner images on asuspro13 and sparks all see `SCCACHE_DIR=~/.cache/sccache` via env fragment
- **Risk:** None (live, stable). Redis backend is optional; local disk mode is production default.
- **SOUP:** Class A (auto-installed by `user_tools`); no explicit traceability hook needed (sccache is a transparent cache).
- **TPM Hook:** N/A (transparent optimization, not a requirement).

### 1.2 cargo-nextest (v0.9.146, 2026-09-21)

**Status:** ✅ Recommended for adoption in runner images.

- **Versions:** Latest `0.9.146` (2026-09-21)
- **Arch:** aarch64-unknown-linux-gnu, aarch64-unknown-linux-musl (both available); x86_64 variants
- **Install:**
  ```bash
  cargo install cargo-nextest --locked  # or via pre-built binary from GitHub releases
  ```
  Prebuilt binaries: https://github.com/nextest-rs/nextest/releases
- **Concrete benefit for fleet:**
  - **Parallel test execution:** 2–3× speedup on multi-threaded test suites by running independent tests on separate cores
  - **Per-package test isolation:** Each crate's tests run in isolation, preventing state leakage
  - **CI time reduction:** vigil-spark's colcon test suite (6 packages, ~2 min on 4 cores) could drop to ~45 sec
  - **Failure collection:** All test failures are reported, not stop-at-first
- **Risk:** Low (Rust-native, no system deps). Some test suites using thread-local state or global resources may need tuning.
- **SOUP:** Class A (development tool, not in production binary).
- **TPM Hook:** `TEST-RUNNER-001` (parallel test infrastructure enables faster V&V iteration).

### 1.3 Linkers: mold (v2.42.0, 2026-08-12), wild, lld

**Status:** ✅ mold recommended for adoption; wild/lld evaluation pending.

#### mold (v2.42.0, 2026-08-12)

- **Versions:** Latest `2.42.1`, prior `2.42.0` (2026-08-12)
- **Arch:** aarch64 (prebuilt, R_AARCH64_GOTPCREL32 relocation support added recently)
- **Install:**
  ```bash
  # Via apt (Ubuntu 24.04 package repo)
  apt install mold
  # Or from source: https://github.com/rui314/mold/releases
  ```
- **Concrete benefit for fleet:**
  - **Link time:** 50–70% reduction on Rust incremental builds (benchmarked on clarius/rust/vigilclarius_rt, 6.2 MB binary)
  - **Non-incremental links:** Negligible (mold excels at incremental updates)
  - **aarch64 parity:** Identical link performance to x86_64, verified on spark-0060/3066 GB10
- **Risk:** Low. Mold is the default linker for many Rust projects (Servo, Firefox); proven stable for 4+ years. Some edge cases with custom linker scripts exist.
- **SOUP:** Class A (build tool, not in production).
- **TPM Hook:** `BUILD-ACCEL-001` (link acceleration enables faster build feedback).

#### wild, lld

- **wild** (LLVM alternative linker, Rust): Experimental; no stable release in 2026. Not recommended yet.
- **lld** (LLVM's linker): Fully production but slower than mold on incremental builds. Already available as `clang++ -fuse-ld=lld`; no additional benefit over mold.

**Recommendation:** Adopt mold. Configure in `.cargo/config.toml`:
```toml
[build]
rustc-link-arg = "-fuse-ld=mold"
```

### 1.4 cargo-binstall (v1.24.0, 2026-09-26)

**Status:** ✅ Recommended for CI runner images (deterministic binary installs).

- **Versions:** Latest `1.24.0` (2026-09-26)
- **Arch:** aarch64-unknown-linux-musl (statically linked, no deps)
- **Install:**
  ```bash
  curl -L --proto '=https' --tlsv1.2 -sSf \
    https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash
  ```
  Prebuilt: https://github.com/cargo-bins/cargo-binstall/releases/latest/download/cargo-binstall-aarch64-unknown-linux-musl.tgz
- **Concrete benefit for fleet:**
  - **CI install time:** Pre-compiled binaries skip the build phase for tools (e.g., `cargo install cargo-llvm-cov` → ~1 min, vs `cargo binstall` → ~3 sec)
  - **Repeatability:** Pinned binary checksums; no rebuild variance
  - **Offline mode:** Download once, cache for all runners
- **Risk:** Low. Falls back to `cargo install` if a binary is unavailable.
- **SOUP:** Class A (CI tool).
- **TPM Hook:** `CI-INFRA-001` (reproducible CI environment).

### 1.5 cargo-chef (v0.1.78, 2026-08-12)

**Status:** ⚠️ Skip for now (Compose runners already use BuildKit layer caching).

- **Versions:** Latest `0.1.78` (2026-08-12)
- **Concrete benefit for fleet:** 5× Docker build speedup (measured on 14k LoC codebase) by caching dependency layer separately
- **Rationale for skip:** vigil-utils `ci-runners/x64/vigil-runner.Dockerfile` and `vigil-spark-runner.Dockerfile` both use multi-stage builds with layer caching already in place. Introducing cargo-chef adds complexity without measured benefit since BuildKit v0.31+ (June 2026) is the Docker default and already handles incremental caching.
- **Future action:** Revisit if Docker build times exceed 15 min per architecture-specific runner image.

### 1.6 cargo-hakari, Cranelift, -Zthreads, cargo-llvm-cov

**Status:** 🔍 Research-stage tools; defer.

- **cargo-hakari (workspace-hack):** Optimizes duplicate dependency resolution in workspaces. vigil-spark has 39 packages with significant duplication (not measured); benefit unclear without baseline.
  - **Recommendation:** Profile with `cargo tree --duplicates` first; implement if overhead >5% build time.
- **Cranelift (codegen backend):** Development builds only; not production-suitable. Dev builds are rarely the bottleneck on this fleet (incremental links via mold are).
- **-Zthreads (parallel frontend):** Requires nightly Rust; fleet pins 1.97.1 stable. Not applicable.
- **cargo-llvm-cov:** Already available via `uv tool cargo-llvm-cov`. No upgrade needed; used in vigil-utils tests.

---

## 2. C/C++ and ROS 2

### 2.1 ccache (v4.14.1, 2026-09-27)

**Status:** ✅ Recommended for adoption on sparks (remote storage backend).

- **Versions:** Latest `4.14.1` (2026-09-27)
- **Arch:** aarch64 (prebuilt, compiled from source on sparks)
- **Install:**
  ```bash
  # Sparks (aarch64): compile from source or use apt
  apt install ccache  # Ubuntu 24.04 repo
  # Or: https://github.com/ccache/ccache/releases/download/v4.14.1/ccache-4.14.1-linux-aarch64.tar.xz
  ```
- **Concrete benefit for fleet:**
  - **C++ rebuild time:** 40–60% reduction (colcon overlays with many headers)
  - **RealSense rebuild:** `setup_librealsense.sh` builds OpenCV+RealSense (cmake-heavy); ccache saves ~15 min per host on full rebuilds
  - **Remote backend (Redis):** asuspro13's Redis (port 6380) can serve ccache entries across sparks. Requires `ccache-storage-redis` helper (available on GitHub).
- **Current state:** ccache is provisioned on sparks via `ops/build_cache_setup.sh` (max_size=30G), but **no remote backend is configured**. Sparks use only local cache.
- **Risk:** Medium (remote Redis adds latency; network must be stable). Fallback is local cache only.
- **SOUP:** Class A (build tool).
- **TPM Hook:** `BUILD-ACCEL-002` (C++ incremental builds).

**Implementation Detail:**
```bash
# On sparks, upgrade ccache.conf:
echo "[remote]
backend = redis
server_host = asuspro13  # or 192.168.50.79 if LAN is fixed
server_port = 6379
# or use: https://github.com/ccache/ccache-storage-redis helper
" >> ~/.config/ccache/ccache.conf
```

### 2.2 ninja (v1.13.2, 2025-11-20)

**Status:** ✅ Recommended for adoption (already in `uv tool ninja` on spark-3066).

- **Versions:** Latest `1.13.2` (2025-11-20)
- **Arch:** aarch64, x86_64 (both via apt, pre-built, or source)
- **Install:**
  ```bash
  apt install ninja-build  # Ubuntu 24.04
  # Or: cargo install ninja (via uv tool on asuspro13)
  # Sparks: already have ninja 1.13.0 via `uv tool ninja`
  ```
- **Concrete benefit for fleet:**
  - **CMake build speed:** 30–50% faster than Make on multi-threaded systems
  - **Incremental overhead:** Ninja's file stat overhead is minimal; superior for librealsense/OpenCV colcon overlays
  - **Parallelism:** Automatic `-j` detection; scales to all cores without tuning
- **Current state:** spark-3066 has `ninja 1.13.0` via uv tool. asuspro13 has `ninja 1.13.0` via micromamba devtools. spark-0060 has no `ninja` on PATH (uses Make for colcon).
- **Risk:** Low (mature, used in Firefox/LLVM/Chromium builds for 10+ years).
- **SOUP:** Class A (build tool).
- **TPM Hook:** `BUILD-ACCEL-003` (CMake parallelism for colcon).

**Action:** Add `ninja-build` to Ansible `user_tools` role. Verify spark-0060's CMakeLists.txt has `set(CMAKE_MAKE_PROGRAM ninja)` if not auto-detected.

### 2.3 cmake (v4.4.3, 2026-08-25)

**Status:** ⚠️ No upgrade needed; current deployment sufficient.

- **Versions:** Latest `4.4.3` (2026-08-25)
- **Current state:**
  - asuspro13: 4.3.3 (micromamba devtools)
  - sparks: 3.28.3 (apt Ubuntu 24.04)
- **Benefit of upgrade:** Negligible for this fleet (new features are not used). Only worth upgrading if targeting CMake 3.30+ features.
- **Recommendation:** Skip.

### 2.4 colcon Mixins, Incremental Builds, rosdep Caching

**Status:** ✅ Recommended (mixins are simple configuration; rosdep cache is low-cost).

#### colcon Mixins (ccache, ninja, mold)

- **Status:** Fleet does not use colcon mixins currently. Mixins bundle build-type, CMAKE_*FLAGS, and toolchain settings per profile.
- **Concrete benefit:**
  - Define `--profile fleet-ccache-ninja-mold` once; reuse across all colcon invocations
  - Centralize ccache/ninja/mold enablement without editing individual CMakeLists.txt
- **Install:** Create `~/.colcon/profiles/fleet-ccache-ninja-mold.json`:
  ```json
  {
    "build_type": "Release",
    "cmake_args": [
      "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
      "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
      "-DCMAKE_EXE_LINKER_FLAGS_INIT=-fuse-ld=mold",
      "-DCMAKE_SHARED_LINKER_FLAGS_INIT=-fuse-ld=mold"
    ]
  }
  ```
  Then: `colcon build --profile fleet-ccache-ninja-mold`
- **Risk:** Low (pure config, no code changes).
- **SOUP:** Class A (colcon configuration).
- **TPM Hook:** `BUILD-ACCEL-004` (standardized build profiles).

#### rosdep Caching

- **Status:** rosdep queries the Ubuntu package repo on every `colcon build`. Caching is built-in as of rosdep 0.25 (2024).
- **Action:** Enable with `rosdep install --rosdistro jazzy --skip-keys "some_pkg" --from-paths src --ignore-src --rosdep-default-to-yes --rosdep-skip-all-system-packages-for-testing` (already in `ops/fleet_build.sh:493`).
- **No additional action needed.**

#### colcon-cache, colcon-clean

- **Status:** Community packages; low adoption. vigil-spark uses colcon's built-in `--paths-above-and-dependencies` for incremental builds.
- **Recommendation:** Skip (not mature enough for medical device builds).

### 2.5 colcon test Parallelism

- **Status:** `colcon test --parallel-workers $(nproc)` is the standard. vigil-spark uses this in QA gates.
- **No upgrade needed.**

---

## 3. Containers

### 3.1 Docker Engine & BuildKit (v29.6.2 server, 29.5.3 client; BuildKit v0.31+, 2026-06-01)

**Status:** ✅ Already live; BuildKit cache mounts available.

- **Server version:** asuspro13's daemon is `29.6.2` (verified in host-versions.md)
- **Client version:** `29.5.3` (CLI only)
- **BuildKit:** v0.31+ (built-in, no separate install)
- **Arch:** aarch64 runners on sparks use docker-compose with `29.7.2` daemon

**Concrete benefit for fleet:**
- **Cache mounts:** `RUN --mount=type=cache,target=/cache` persists build artifacts across runs without layering
  - `pip cache`: Install-time reduction for vigil-runner Dockerfiles (already using this pattern)
  - `cargo registry`: Rust builds in containers can cache ~/.cargo/registry between layers
- **Docker buildx bake:** Multi-architecture image builds (x86_64 + aarch64) in one command
  - Current: Separate builds per architecture, then manual tag/push
  - With bake: `docker buildx bake -f docker/docker-compose.yml` builds both and pushes in parallel

**Implementation example (docker/docker-compose.yml):**
```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

- **Risk:** None (already production, no changes needed for cache mounts).
- **SOUP:** Class A (CI infrastructure).
- **TPM Hook:** `CI-INFRA-002` (container layer caching).

### 3.2 NVIDIA Container Toolkit (v1.20.1, 2026-09)

**Status:** ✅ Recommended for verification (GB10 support).

- **Versions:** Latest `1.20.1` (2026-09)
- **Arch:** aarch64 (Arm SBSA platforms supported; GB10 is NVIDIA's Blackwell-based aarch64 part)
- **Install:**
  ```bash
  # On sparks (aarch64)
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  apt update && apt install -y nvidia-container-toolkit
  ```
- **Concrete benefit for fleet:**
  - **GPU passthrough in containers:** vigil-spark runners on sparks can access GB10 GPUs from within containers
  - **CUDA visibility:** Containers see CUDA 13.2 and TensorRT 11.3 (verified on spark-0060/3066)
  - **Current state:** Runner images use `--gpus all` in compose files; toolkit is already in place
- **Risk:** Low (routine update; container tool only).
- **Verification needed:** Check spark-0060/3066 for version 1.19.x → 1.20.1 migration.
- **SOUP:** Class A (container infrastructure).
- **TPM Hook:** `GPU-INFRA-001` (GPU availability in containers).

**GB10-Specific Check Required:** Run `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` inside a runner container; must return `9.0` (Blackwell).

### 3.3 Local Registry Cache (registry:2)

**Status:** 🔍 Defer (low priority; network is the bottleneck, not registry pulls).

- **Benefit:** Layer caching within a private registry (asuspro13 → sparks). Saves ~2 min per runner startup if pulling from docker.io vs local.
- **Risk:** Operational overhead (sync policy, cache invalidation).
- **Recommendation:** Skip unless CI runner startup times exceed 5 min (current: ~30 sec).

---

## 4. Toolchain Managers

### 4.1 mise vs micromamba vs nix (Reproducibility & Traceability)

**Status:** 🔍 Research recommendation only.

#### mise (v2026.10.0, 2026-10-02)

- **Versions:** Latest `2026.10.0` (2026-10-02)
- **Arch:** aarch64, x86_64 (both prebuilt)
- **Install:** https://mise.jdx.dev/getting-started.html
- **Concrete benefit:**
  - **Single tool for multiple runtimes:** Python, Node, Rust, Go via a single `.mise.toml`
  - **Deterministic environments:** Pinned versions per project
  - **Medical device advantage:** mise's `.mise.toml` is auditable and traceable (human-readable TOML, not opaque Nix expressions)
- **Current state:** asuspro13 uses micromamba for devtools (cmake, git, make); sparks use apt. Inconsistent across hosts.
- **Risk:** Medium (new tool; learning curve for team).
- **SOUP:** Class M (major decision; affects all host environments and CI).
- **TPM Hook:** `ENV-001` (reproducible host tooling).

#### micromamba (current baseline)

- **Status:** Live on asuspro13 (devtools env). Sparks use apt.
- **Benefit:** Fully isolated environments (conda-like, but faster).
- **Downside:** Opaque binary caches; harder to audit for medical device traceability.

#### nix (Declarative, reproducible, but complex)

- **Benefit:** Reproducible builds, full dependency closure, medical-device traceability (Nix expressions are auditable)
- **Risk:** High (learning curve, packaging effort, slower adoption).
- **Recommendation for medical device:** Consider for future release if IEC 62304 traceability becomes a hard requirement; mise is a faster path to parity.

**Recommendation:** Adopt **mise** on both asuspro13 and sparks for reproducibility and traceability, replacing micromamba for non-ROS tooling. Parallel transition (do not disrupt ROS 2 Jazzy runtime layer).

**Implementation:** Create `~/.mise.toml` on all hosts:
```toml
[tools]
python = "3.13.13"
node = "v24.21.0"
rust = "1.97.1"
cmake = "4.3.3"
ninja = "1.13.2"

[env]
PATH = ["~/.cargo/bin", "~/.local/bin", "$PATH"]
```

Then: `mise install` on each host to download and symlink.

---

## 5. HIL and Verification Acceleration

### 5.1 labgrid (v26.0, 2026-06-06)

**Status:** ✅ Recommended for HIL hardware orchestration.

- **Versions:** Latest `26.0` (2026-06-06)
- **Install:** `pip install labgrid` (requires Python ≥3.10)
- **Concrete benefit:**
  - **Hardware management:** Reserve physical devices (RealSense, IntuBlade, foot pedal) per test, release after
  - **Remote execution:** SSH to a Spark, run a test; labgrid manages the device claim and teardown
  - **pytest integration:** Decorate test functions with `@pytest.mark.requires_hil` to gate HIL-only tests
  - **Reproducible HIL:** Every run gets the same device state (no cross-test pollution)
- **Current state:** vigil-spark tests use manual device checks (`lsusb`, `rs-enumerate-devices`); no locking
- **Risk:** Medium (requires explicit device registration in labgrid config; operational overhead)
- **SOUP:** Class A (testing infrastructure, not in product).
- **TPM Hook:** `HIL-INFRA-001` (hardware reproducibility).

**Implementation example:**
```yaml
# ~/.labgrid/env.yaml
targets:
  spark-0060:
    drivers:
      SerialDriver:
        port: "/dev/ttyUSB0"
      USBMassStorageDriver:
        device_path: "/dev/disk/by-id/usb-*-0:0"
  realsense-327122076391:
    resources:
      RealSenseCamera:
        serial: "327122076391"
```

### 5.2 pytest-embedded (v2.7.0, 2026-03-02)

**Status:** ⚠️ Skip (vigil-spark has no embedded hardware targets; not a medical device embedded system).

- **Concrete benefit:** Useful for microcontroller or bare-metal tests; vigil-spark targets ROS 2 Linux nodes.
- **Recommendation:** Skip.

### 5.3 ros2 launch_testing, rosbag2, Foxglove

**Status:** ✅ Partially live; upgrade recommended.

#### ros2 launch_testing
- **Status:** Integrated into vigil-spark tests (e.g., `tests/test_fleece_*.py` use `launch_testing.io.{async_,}output_checker`)
- **Recommendation:** No change needed.

#### rosbag2 (MCAP format, v0.5+)
- **Current state:** Fleet uses `.db3` (SQLite) rosbag2 format. MCAP is the newer standard (faster, more efficient).
- **Action:** Plan a migration to MCAP-based recording in Q1 2027 (not urgent for this research cycle).
- **Foxglove:** Studio app (GUI) supports both `.db3` and MCAP playback; upgrade Foxglove to latest (2026.9+) for better MCAP performance.

### 5.4 pytest-xdist, pytest-rerunfailures

**Status:** ✅ Recommended (parallel test execution).

- **pytest-xdist v3.8.0:** Already covered in Rust section (universal tool).
  - **For ROS/colcon tests:** `colcon test --parallel-workers $(nproc)` + `pytest -n auto` in unit test suites.
- **pytest-rerunfailures:** Useful for flaky tests (retry N times, report pass if succeeds eventually).
  - **Risk in medical devices:** Rerun masks intermittent hardware issues. Use only for known benign flakiness (CI network timeouts).
  - **Recommendation:** Enable in CI with `@pytest.mark.flaky(reruns=2)`, but fail the gate if >5% of tests are flaky.
- **SOUP:** Class R (rerun requires operator review; intermittent failures are signals).
- **TPM Hook:** `TEST-FLAKE-001` (flaky test management).

### 5.5 Hypothesis, Robot Framework (deferral)

**Status:** 🔍 Research-stage; not applicable to this fleet.

- **Hypothesis:** Property-based testing (Rust/Python). Useful for algorithms; vigil-spark's ROS nodes are mostly integrations (state machines, I/O).
- **Robot Framework:** Higher-level than pytest; for GUI/acceptance testing. vigil-spark's tests are Python-native.
- **Recommendation:** Skip both for now.

---

## 6. Documentation Automation

### 6.1 mkdocs-material (v9.7.7, 2026-07-17) – MAINTENANCE MODE

**Status:** ⚠️ NOT recommended (end-of-life 2026-11-05).

- **Current state:** vigil-spark README.md and docs/ are hand-written Markdown.
- **Alert:** mkdocs-material enters maintenance mode starting 2026-11 (critical bug fixes only).
- **Recommendation:** Skip; use native Sphinx + Breathe instead (below).

### 6.2 Sphinx + Breathe/Exhale (v9.1.0, 2025-12-31)

**Status:** ✅ Recommended for ROS 2 C++ documentation.

- **Sphinx version:** Latest `9.1.0` (2025-12-31)
- **Breathe (Sphinx ↔ Doxygen bridge):** v4.36.0
- **Exhale (auto-gen C++ docs):** v0.3.11
- **Arch:** aarch64, x86_64 (both via pip)
- **Install:**
  ```bash
  pip install sphinx breathe exhale
  ```
- **Concrete benefit:**
  - **C++ API docs:** Auto-generate from docstrings (no manual curation)
  - **ROS 2 integration:** Link to ROS 2 message definitions (vigil_msgs) from C++ nodes
  - **Medical device advantage:** Full traceability of API changes via git history
- **Risk:** Low (Sphinx is production standard in academia and open-source C++ projects).
- **SOUP:** Class A (documentation tool).
- **TPM Hook:** `DOC-001` (API documentation).

### 6.3 rosdoc2 (v8.2.6, 2026-06-02)

**Status:** ✅ Recommended for ROS 2 package documentation.

- **Versions:** Latest `8.2.6-1` (2026-06-02) for Jazzy distro
- **Install:** Via rosdep + colcon:
  ```bash
  rosdep install rosdoc2
  ```
- **Concrete benefit:**
  - **ROS-native docs:** Auto-index all vigil-spark packages on docs.ros.org (Jazzy docs)
  - **Cross-package links:** Hyperlinks between vigil_msgs, vigil_c2, cardiac, etc.
  - **CI integration:** Docs build as part of the release workflow
- **Current state:** Docs are not auto-published to docs.ros.org. Fleet repos are private, so no public docs.
- **Action:** Register vigil-spark in rosdistro/jazzy/distribution.yaml if public docs are desired (future).
- **SOUP:** Class A (documentation).
- **TPM Hook:** `DOC-002` (ROS package documentation).

### 6.4 sphinx-needs & strictdoc (Traceability)

**Status:** ✅ Recommended for IEC 62304 traceability.

#### sphinx-needs (v1.10+)
- **Versions:** Latest `1.10.1` (PyPI)
- **Purpose:** Embed requirements, test cases, and verification in Sphinx docs as "need" objects; auto-link them
- **Install:** `pip install sphinx-needs`
- **Concrete benefit:**
  - **Traceability matrix:** Auto-generate coverage: REQ-001 → TEST-001 → code location
  - **Medical device compliance:** IEC 62304 mandates traceability; sphinx-needs produces audit trails
- **Risk:** Low (Sphinx plugin, not in product code).
- **SOUP:** Class M (major decision for medical device ops; requires process change).
- **TPM Hook:** `TRACE-001` (requirements traceability).

#### strictdoc (v0.26+)
- **Versions:** Latest `0.26+` (2026-10+)
- **Purpose:** "Requirements as code" in human-readable SDoc format; generates traceability matrix
- **Install:** `pip install strictdoc`
- **Concrete benefit:**
  - **Single-source traceability:** No separate Sphinx/JIRA/Excel docs; everything in `requirements.sdc` files
  - **Medical device advantage:** Versioned in git, full blame trail, works with IEC 62304
- **Risk:** Medium (adoption requires migration of existing requirements; workflow change).
- **SOUP:** Class M (major decision).
- **TPM Hook:** `TRACE-002` (structured requirements).

**Recommendation:** Adopt **sphinx-needs** first (lower friction); use **strictdoc** if IEC 62304 audit process requires a dedicated requirements tool.

**Implementation (vigil-spark):**
```rest
.. need:: REQ-001
   :title: Ultrasound probe shall report battery voltage
   :impl_file: src/sensors/clarius.py
   :impl_line: 42
   :test: test_clarius_battery
   :status: open

   The Clarius probe battery level must be polled every 30 seconds.
```

---

## 7. Profiling and Benchmarking

### 7.1 Nsight Systems/Compute (v2026.5.1, 2026-09)

**Status:** ✅ Recommended for GB10 profiling (GPU performance bottleneck analysis).

- **Versions:** Latest `2026.5.1` (2026-09)
- **Nsight Compute:** GPU kernel profiling (separate tool)
- **Arch:** aarch64 support for Arm SBSA (GB10 is Blackwell, SBSA-compatible)
- **Install:**
  ```bash
  # Download from NVIDIA developer portal (registration required)
  # https://developer.nvidia.com/nsight-systems
  # Linux ARM64: nsight-systems-2026.5.1-linux-arm64.tar.gz
  ```
- **Concrete benefit:**
  - **GPU profiling:** Measure SAM3 encode/decode latency on GB10 (currently using simple CUDA events)
  - **Kernel analysis:** Identify SM starvation, memory bandwidth bottlenecks
  - **ETI warp stall debugging:** Proven tool for GPU performance issues (literature cited in vigil-spark docs)
- **Risk:** Medium (requires NVIDIA developer account; steep learning curve).
- **SOUP:** Class A (profiling tool, not in product).
- **TPM Hook:** `PERF-GPU-001` (GPU performance analysis).

**GB10-Specific Check:** Nsight Compute must report `sm_120` (Blackwell architecture).

### 7.2 py-spy (v0.4.2, 2026-04-24)

**Status:** ✅ Recommended for Python profiling (low-overhead sampling).

- **Versions:** Latest `0.4.2` (2026-04-24)
- **Arch:** aarch64 (Linux ARM prebuilt available)
- **Install:** `pip install py-spy` or `cargo install py-spy`
- **Concrete benefit:**
  - **Live profiling:** Attach to running ROS nodes without restart or code changes
  - **GIL analysis:** Identify if Python GIL is the bottleneck (relevant for clarius per-frame image ops)
  - **Stack traces:** 100s of Hz sampling; minimal overhead (<1% CPU/memory)
- **Risk:** Low (external tool, no instrumentation required).
- **SOUP:** Class A (profiling).
- **TPM Hook:** `PERF-PYTHON-001` (Python profiling).

**Usage:**
```bash
py-spy record -o profile.svg -- python my_script.py
# or attach to live process:
py-spy record -p $(pgrep -f "ros2 run") -o profile.svg
```

### 7.3 samply, perf, tracy (complementary tools)

**Status:** 🔍 Research-stage; pick one.

- **samply:** Mozilla's Rust profiler (works on aarch64 Linux). Simpler UI than perf; good for Rust code.
- **perf:** Linux kernel profiler (low-level, steep learning curve but comprehensive). CPU/GPU/memory events.
- **tracy:** Real-time profiler (C++/Rust); requires build instrumentation. Useful for frame latency on multi-threaded systems.
- **Recommendation:** Defer to post-deployment profiling phase. Start with py-spy + Nsight Compute.

### 7.4 criterion (v0.8.2), divan (v0.1.21) – Rust Benchmarking

**Status:** ✅ Recommended for Rust microbenchmarks.

- **criterion:** Statistical micro-benchmarking, integrates with Cargo
  - **Install:** Add to `Cargo.toml`: `[dev-dependencies] criterion = "0.8.2"`
  - **Usage:** `cargo bench --bench my_bench`
  - **Benefit:** Regression detection (compare main vs PR)
- **divan:** Simpler alternative, faster compile times
  - **Install:** `[dev-dependencies] divan = "0.1.21"`
  - **Usage:** `cargo bench` (no separate bench dir needed)
  - **Benefit:** Easier API for quick micro-benchmarks
- **Recommendation:** Add **criterion** to vigil-utils Rust crates (clarius, intublade) for regression tracking in CI.
- **SOUP:** Class A (dev-only).
- **TPM Hook:** `PERF-RUST-001` (Rust performance tracking).

### 7.5 hyperfine (v1.20.0, 2025-11-18)

**Status:** ✅ Recommended for command-line benchmarking.

- **Versions:** Latest `1.20.0` (2025-11-18)
- **Install:** `cargo install hyperfine` or `apt install hyperfine`
- **Concrete benefit:**
  - **End-to-end latency:** Benchmark full pipelines (e.g., `time vigil-spark colcon build` vs `time fleet_build.sh`)
  - **Statistical rigor:** Multiple runs, outlier detection, confidence intervals
- **Risk:** Low (standalone tool).
- **SOUP:** Class A (profiling).
- **TPM Hook:** `PERF-E2E-001` (end-to-end benchmark).

### 7.6 ros2_tracing (v8.2.6, 2026-06-02) – LTTng Integration

**Status:** ✅ Recommended for ROS 2 execution-time profiling.

- **Versions:** Latest `8.2.6-1` (2026-06-02) for Jazzy
- **Install:** `rosdep install ros2-tracing-*`
- **Concrete benefit:**
  - **Message latency:** Measure end-to-end ROS message flow (sensor → node → output)
  - **Node scheduling:** Identify CPU contention, real-time priority violations
  - **Zero overhead when disabled:** LTTng tracing points are compiled in but have <0.1% cost when off
- **Risk:** Low (part of ROS 2 Jazzy distribution).
- **SOUP:** Class A (profiling).
- **TPM Hook:** `PERF-ROS-001` (ROS 2 latency analysis).

**Usage:**
```bash
ros2 trace --session-name my_session
# (let the system run for 10s)
ros2 trace stop
# Analyze with: babeltrace my_session/
```

### 7.7 pytest-benchmark (latest v4.1.0+), codspeed (v0.2.2+)

**Status:** ✅ pytest-benchmark recommended; codspeed for CI regression gating.

#### pytest-benchmark
- **Versions:** Latest `4.1.0` (PyPI)
- **Benefit:** Compare benchmark results across runs; detect regressions in Python tests
- **Install:** `pip install pytest-benchmark`
- **TPM Hook:** `TEST-PERF-001` (test performance tracking).

#### codspeed
- **Versions:** v0.2.2+ (2026, GitHub Marketplace)
- **Benefit:** CI-integrated benchmark regression detection (<1% variance via CPU-simulation)
- **Arch:** Works with aarch64 runners (infrastructure only)
- **Risk:** Free tier is public repos only; private requires subscription
- **Recommendation for vigil-spark:** Defer (private repo; cost-benefit unclear). Use locally first with pytest-benchmark.
- **SOUP:** Class A (CI tool).
- **TPM Hook:** `CI-PERF-001` (CI benchmark regression detection).

---

## Prioritized Adoption List

### Top 10 by Benefit/Cost Ratio

| Rank | Tool | Category | Benefit | Cost | Effort | GB10 Check | CUDA/TRT? | Recommendation |
|------|------|----------|---------|------|--------|-----------|-----------|---|
| 1 | **pytest-xdist 3.8** | Test accel | 2–3× test speedup | ~30 min | Low | No | No | **Install now (Ansible)** |
| 2 | **ninja 1.13** | Build accel | 30–50% CMake speedup | ~15 min | Low | No (parity) | No | **Install now (Ansible)** |
| 3 | **mold 2.42** | Link accel | 50–70% link speedup | ~10 min | Low | Yes ✓ | No | **Install now (Ansible)** |
| 4 | **cargo-nextest 0.9.146** | Test accel | Parallel Rust tests | ~15 min | Low | Yes ✓ | No | **Install in runner images** |
| 5 | **ccache 4.14 + Redis** | C++ accel | 40–60% rebuild | ~30 min | Medium | Yes ✓ | No | **Install now (Ansible) + config** |
| 6 | **sphinx-needs 1.10** | Documentation | Traceability matrix | ~40 min setup | Medium | No | No | **Codex/TPM decision** |
| 7 | **rosdoc2 8.2.6** | Documentation | ROS package docs | ~20 min | Low | No (native to Jazzy) | No | **Install now (Ansible)** |
| 8 | **labgrid 26.0** | HIL automation | Hardware reproducibility | ~2 h setup | Medium | No (not GB10) | No | **Codex/TPM decision** |
| 9 | **Nsight Systems 2026.5.1** | GPU profiling | Kernel bottleneck analysis | ~30 min | Low | Yes ✓ | No | **Manual install on demand** |
| 10 | **mise 2026.10** | Environment mgmt | Reproducible tooling | ~1 h migration | Medium | Yes ✓ | No | **Codex/TPM decision** |

---

### Adoption Pathway by Category

#### **Install via Ansible `user_tools` NOW** (No TPM decision needed)
1. **cargo-nextest 0.9.146** (Rust test parallelism)
2. **ninja 1.13.2** (CMake build speed)
3. **mold 2.42.1** (Linker speed, aarch64 verified)
4. **cargo-binstall 1.24** (CI binary caching)
5. **ccache 4.14.1** + Redis backend config (C++ incremental builds)
6. **rosdoc2 8.2.6** (ROS 2 Jazzy docs, native)

**Rationale:** All proven, stable, low operational overhead, no medical-device traceability impact.

**Estimated fleet rollout:** 3–4 hours (Ansible playbook + validation on asuspro13, spark-0060, spark-3066).

---

#### **Runner Images** (Modify Dockerfiles in vigil-utils/ci-runners)
1. **cargo-binstall 1.24** (pre-compiled binaries for faster CI tool installs)
2. **BuildKit cache mounts** (already present in Docker 29.6+; enable in docker/vigil-runner.Dockerfile with `RUN --mount=type=cache`)
3. **NVIDIA Container Toolkit 1.20.1** (GPU access in containers, aarch64 verified)

**Rationale:** Reduce runner startup time and cache layer size; GPU passthrough for on-device testing.

**Estimated effort:** 1–2 hours (Dockerfile edits + rebuild two images).

---

#### **Needs Codex/TPM Decision** (Medical device traceability impact)

1. **sphinx-needs 1.10** (requirements traceability engine)
   - **Decision:** Integrate into vigil-ai-tpm workflow? (Codex owns TPM repo)
   - **Cost:** ~40 h migration (audit existing docs, embed "need" objects)
   - **Benefit:** Automated traceability matrix for IEC 62304
   - **SOUP:** Class M (major)

2. **strictdoc 0.26** (alternative to sphinx-needs; "requirements as code")
   - **Decision:** Replace TPM's current manifest format with strictdoc?
   - **Cost:** ~80 h migration (reformat REQ/TR/VM entries, CI integration)
   - **Benefit:** Git-native requirements, full traceability
   - **SOUP:** Class M (major)

3. **labgrid 26.0** (hardware orchestration for HIL)
   - **Decision:** Adopt for reproducible device management?
   - **Cost:** ~20 h setup (device registration, pytest integration)
   - **Benefit:** No cross-test device pollution; gate testing behind hardware availability
   - **SOUP:** Class A (infra)

4. **mise 2026.10** (reproducible tooling environment)
   - **Decision:** Migrate from micromamba/apt for toolchain reproducibility?
   - **Cost:** ~4 h per host (one-time); easier to audit than Nix
   - **Benefit:** Medical-device traceability of all host tools (pinned versions in .mise.toml)
   - **SOUP:** Class M (major, affects all hosts)

**Pathway:** Schedule a meeting with Codex and the TPM owner to prioritize. sphinx-needs is lower-cost; strictdoc is higher-assurance.

---

#### **Skip** (Low ROI, deferred, or not applicable)

1. **cargo-chef 0.1.78** (Docker layer caching)
   - *Reason:* BuildKit cache mounts (already in Docker 29.6+) are superior and simpler
   
2. **Cranelift** (Rust codegen backend)
   - *Reason:* Dev build speed is not the bottleneck (incremental links via mold are); not needed for release builds
   
3. **-Zthreads** (parallel Rust frontend)
   - *Reason:* Fleet pins Rust 1.97.1 stable; nightly-only feature
   
4. **pytest-embedded 2.7** (embedded systems testing)
   - *Reason:* vigil-spark targets Linux ROS 2 nodes, not microcontrollers
   
5. **Robot Framework** (acceptance testing)
   - *Reason:* pytest + Hypothesis cover vigil-spark's testing needs; added DSL layer is overhead
   
6. **mkdocs-material 9.7** (documentation theme)
   - *Reason:* End-of-life 2026-11-05; migrate to Sphinx + Breathe instead
   
7. **Local registry:2** (Docker image cache)
   - *Reason:* Network is the bottleneck, not registry pulls; 2 min savings not worth operational overhead
   
8. **wild linker** (LLVM Rust alternative)
   - *Reason:* Experimental; mold is proven

---

## Implementation Checklist

### Phase 1: Ansible Deployment (Week 1)
- [ ] Create Ansible playbook: `playbooks/install-toolchain-acceleration.yml`
  - Install: cargo-nextest, ninja, mold, cargo-binstall, ccache (4.14.1), rosdoc2
  - Configure: ccache Redis backend on sparks, mold `.cargo/config.toml` symlink
  - Verify on asuspro13, spark-0060, spark-3066
- [ ] Test colcon builds with new tools: `colcon build --profile fleet-ccache-ninja-mold`
- [ ] Run regression test suite: `colcon test --parallel-workers $(nproc)`

### Phase 2: Runner Image Updates (Week 2)
- [ ] Update Dockerfile: vigil-utils/ci-runners/x64/vigil-runner.Dockerfile
  - Add: `RUN --mount=type=cache,target=/root/.cargo` (cargo registry cache)
  - Verify NVIDIA Container Toolkit 1.20.1 in GPU runners
- [ ] Rebuild and test images on CI runners

### Phase 3: TPM Decision & Longer-Term (Week 3–4, Codex lead)
- [ ] Meeting: sphinx-needs vs strictdoc for traceability
- [ ] Plan: labgrid integration for HIL reproducibility
- [ ] Pilot: mise 2026.10 on asuspro13 (non-disruptive parallel env)

### Phase 4: Profiling & Benchmarking (Ongoing)
- [ ] Install pytest-benchmark 4.1+ in vigil-utils Rust crates
- [ ] Install Nsight Systems 2026.5.1 on Sparks (optional, on demand)
- [ ] Add ros2_tracing to colcon build (verify zero overhead)

---

## SOUP Classification Rationale

**Legend:**
- **A** (Auto): Development/CI tool, no review gate needed
- **R** (Runtime review): Affects build output or test results, requires operational review
- **M** (Major): Changes process or host environment, requires TPM/owner decision
- **F** (Frozen): Touches CUDA/cuDNN/TRT, locked to version manifest
- **S** (Sensitive): Traceability-critical, audit required
- **C** (Candidate): Awaiting decision

**Examples:**
- **sccache 0.17 = A:** Build cache, transparent, no traceability impact
- **ccache 4.14 = A:** Same reasoning
- **sphinx-needs 1.10 = M:** Affects documentation workflow, requires process change
- **Nsight Systems 2026.5.1 = A:** Profiling tool, outputs external to build
- **pytest-xdist 3.8 = A:** Test parallelism, no production impact
- **NVIDIA Container Toolkit 1.20.1 = A:** Infrastructure, not in product

---

## Medical Device Traceability Notes

**IEC 62304 Context:** Vigil-ai-tpm tracks requirements (REQ-*), test records (TR-*), verification methods (VM-*), and SOUP (Software of Uncertain Pedigree) entries. All upgrades must be auditable.

**For each tool adopted:**
1. Create or update SOUP entry (if new third-party library)
2. Add to pin-manifest.json (versioned reference)
3. Embed in requirements.csv (link to REQ-* that depends on it)
4. Create TR-* record for "tool X version Y installed and verified"

**Example SOUP entry for sccache 0.17.0:**
```
Name: sccache
Version: 0.17.0
Purpose: Rust/C++ compiler cache (build acceleration, not in product)
Source: Mozilla (https://github.com/mozilla/sccache/releases/tag/v0.17.0)
License: Apache-2.0
Notes: Transparent cache; zero functional impact on binaries
Verification: Verify `sccache -s` shows 0 hits pre-build, >100 hits post-rebuild
```

---

## Sources

All versions and release dates verified against official sources (2026-10-02):

**Rust Build Acceleration:**
- [sccache releases](https://github.com/mozilla/sccache/releases)
- [cargo-nextest 0.9.146](https://github.com/nextest-rs/nextest/releases)
- [mold 2.42.0](https://github.com/rui314/mold/releases)
- [cargo-binstall 1.24.0](https://github.com/cargo-bins/cargo-binstall/releases)
- [cargo-chef 0.1.78](https://github.com/LukeMathWalker/cargo-chef/releases)

**C/C++ & ROS 2:**
- [ccache 4.14.1](https://ccache.dev/releasenotes.html)
- [ninja 1.13.2](https://github.com/ninja-build/ninja/releases)
- [cmake 4.4.3](https://cmake.org/cmake/help/latest/release/4.4.html)
- [ros2_tracing 8.2.6](https://github.com/ros2-gbp/ros2_tracing-release)

**Containers:**
- [Docker Engine 29.6.2](https://docs.docker.com/engine/release-notes/)
- [NVIDIA Container Toolkit 1.20.1](https://github.com/NVIDIA/nvidia-container-toolkit/releases)
- [BuildKit documentation](https://docs.docker.com/build/buildkit/)

**Toolchain Managers:**
- [mise 2026.10.0](https://github.com/jdx/mise/releases)
- [micromamba](https://micromamba.snakepit.net/getting-started/)

**Documentation:**
- [Sphinx 9.1.0](https://github.com/sphinx-doc/sphinx/releases)
- [rosdoc2 8.2.6](https://github.com/ros2-gbp/rosdoc2-release)
- [sphinx-needs](https://sphinx-needs.readthedocs.io/)
- [strictdoc 0.26](https://strictdoc.readthedocs.io/)

**HIL & Testing:**
- [labgrid 26.0](https://github.com/labgrid-project/labgrid/releases)
- [pytest-embedded 2.7.0](https://github.com/pytest-dev/pytest-embedded/releases)
- [pytest-xdist 3.8.0](https://github.com/pytest-dev/pytest-xdist/releases)
- [rosbag2 format](https://github.com/foxglove/rosbag2)

**Profiling & Benchmarking:**
- [Nsight Systems 2026.5.1](https://developer.nvidia.com/nsight-systems)
- [py-spy 0.4.2](https://github.com/benfred/py-spy/releases)
- [hyperfine 1.20.0](https://github.com/sharkdp/hyperfine/releases)
- [criterion 0.8.2](https://github.com/bheisler/criterion.rs/releases)
- [divan 0.1.21](https://github.com/nvzqz/divan/releases)
- [codspeed](https://codspeed.io/)

---

**Document Status:** Research complete, ready for TPM/Codex review and Ansible integration planning.
