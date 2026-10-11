# VIGIL-Eko development and qualification boundary

Reviewed 2026-10-11 against the separate `David-Martel/vigil-eko` checkout at
`9cac42c`, with source review refreshed through `9ee408f` and camera design at
`7fd6fa7`; the integrated source update below records subsequent fixes. This is
an integration plan, not a claim that Linux BLE or a clinical classifier has
shipped. The user-supplied `vigil-oko` path
resolves locally to `vigil-eko`; confirm the remote identity on other machines.

## Current source and deployment

The Rust workspace separates protocol decoding (`eko-proto`), Windows WinRT
transport (`eko-ble`), session/reconnect/recording (`eko-session`), DSP,
reporting, CPAL playback, the CLI/app, and PyO3 bindings. Linux CI runs the
workspace's format, clippy and tests. Its WinRT transport is gated by
`cfg(windows)` and now has a separate Windows transport/session CI job.
Compilation and offline fixtures do not establish live Windows BLE reliability.
Linux live BLE requires a separate BlueZ backend; replay tests do not close it.

### Integrated source update — 2026-10-11

VIGIL-Eko PR4 merged as `60930f9`, preserving exact retransmissions without
decoding or advancing clocks twice, rejecting conflicting duplicates and
recording their integrity witness before subsequent samples. PR5 merged as
`89c0e0d`, handling preview startup/shutdown races and suppressing stale
landmarks/previews after inference. Its Windows/Linux Python CI joins the
Windows Rust transport/session and Linux Rust workspace checks. All four jobs
passed on the combined main commit `89c0e0d` in run `38176794919`.

Offline protocol/oracle parity and fake camera/engine/clock fixtures support
these source changes. No device capture or native pose model was executed for
them. Failed transport teardown quarantine, reliable terminal UI delivery,
explicit recording-finalization errors, camera identity after USB reorder and
Linux live capture remain separate acceptance work. Both owned proposal
branches and the clean Codex worktree were removed after upstream ancestry
verification; Claude's primary checkout and locked worktrees were preserved.

Keep application lockfiles and target configurations in their own repository.
PC_AI's isolated media venv is an inventory/qualification environment, not the
VIGIL runtime. A binary built or a wheel imported locally is not deployed on
another Windows or Linux host. Record source SHA, target triple, toolchain,
enabled features, artifact hash, dependency/model manifest and actual host checks
with each release. On the investigated NUC, use at most two build jobs and one
build at a time until the physical adapter is qualified.

## Dell UVC and pose guidance

Use the WB7022 as a video device; obtain PCG/ECG from the stethoscope and choose
speaker/output endpoints independently. Identify devices by enumerated stable
identity and explicitly negotiated format, rather than camera index zero or an
advertised name. Qualification must record USB topology, actual frame format,
resolution, frame rate and backend. Start with a bounded 720p/15-fps engineering
trial; it is a proposed workload budget, not a measured optimum. Windows and
Linux capture implementations must report unavailable/permission-denied/device-
lost states and release the device after cancellation or unplug.

VIGIL-Eko's accepted ADR-0002 specifies an isolated MediaPipe camera sidecar
as a time-limited exception to its Rust-first rule; a future Rust producer uses
the same packet protocol. Rust retains downstream placement/evidence, BLE,
recording and readiness authority. The
[official Pose Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
documents timestamped video/live-stream input and asynchronous callbacks.
Associate every result with its input timestamp and model hash. Use a bounded
latest-frame queue, and expire stale results; an old overlay must never prove
current placement. Camera guidance is advisory under ADR-0002: missing/stale
camera input does not block electrode-readiness auto-start. Keep mirroring and
image-coordinate transforms explicit.

Pose landmarks are inputs to an engineering placement aid. A visible torso
or hand does not prove chestpiece contact, ECG electrode quality, correct
auscultation anatomy or recording quality. Require operator confirmation and
independent signal-quality/integrity checks. Chestpiece tracking and reference
site mapping need their own labelled fixtures and acceptance criteria.

Use CPU processing first. Optional Intel QSV/D3D11 or NVIDIA encode paths must
be qualified on the actually present adapter. Encoding acceleration does not
prove MediaPipe, OpenCV or clinical-model GPU acceleration. A disconnected eGPU
must leave a usable CPU path, rather than trigger inference-library loading.

## Signal provenance and training data

Preserve original PCG/ECG streams, packet sequence/gap witnesses, native sample
rates, sensor layout, contact flags, host monotonic timestamps and clock-offset
uncertainty. Do not infer synchronization solely from matching sample counts.
Listening filters, gain, resampling and speaker frequency shifts belong to a
derived playback path and must not overwrite original recordings. Sensor values
with no established physical calibration stay raw, with unknown units.

Training/upskilling labels should distinguish site guidance, acquisition quality,
operator-confirmed completion and research assessment. Do not turn a software
readiness flag into a diagnosis. The
[manufacturer's CORE 500 capture instructions](https://support.ekohealth.com/hc/en-us/articles/16573698777755-Capturing-Sounds-and-ECG-with-the-CORE-500)
describe device acquisition and orientation; the host implementation needs its
own validation and does not inherit the vendor application's clinical status.
The [CirCor dataset](https://physionet.org/content/circor-heart-sound/1.0.3/)
is a research candidate, not a clinical qualification receipt for this device.
Before model work, record dataset/license/version provenance, patient-level
splits, reference labels and domain differences; prohibit patient overlap between
training and evaluation. Keep patient recordings and identifiers outside Git.

## Required evidence before release acceptance

| Lane | Evidence and acceptance boundary |
|---|---|
| Pure Rust | Locked format/clippy/unit/fixture tests; decoder parity against the independent oracle; corrupt/truncated/reordered/gapped fixtures. |
| Windows | Compile and test WinRT-specific paths on a Windows runner. Inject timeout, failed unsubscribe/close and cancellation. Reconnect cannot reuse an uncertain backend or deliver callbacks from the previous session. |
| Linux | Replay/CLI/app tests and packaging are separate from BlueZ pairing/notification/reconnect HIL. Install documented ALSA and GUI build dependencies on the runner. |
| Recording | Checkpoint/reopen after interruption; exact sample rate/channel metadata; gap accounting; disk-full and failed-finalization reported to the operator. |
| Playback | Synthetic signal and resampler tests; callback deadline/underrun/overflow accounting; no blocking UI/transport operations in the audio callback. Validate actual speaker output separately. |
| Camera | Explicit format negotiation; bounded preview; unplug/replug and permission failure; no retained raw frames unless recording is explicitly requested. |
| Pose | Mirroring, dropped/out-of-order results, occlusion and confidence fixtures; stale results cannot confirm placement, and advisory camera absence preserves the electrode-readiness gate. Labelled placement study remains separate from software tests. |
| Combined HIL | Bounded non-patient trial first, then existing VIGIL test procedures for identity, zero unexplained gaps, reconnect and multi-site workflow. Record actual device/firmware/host and per-run evidence privately. |

Claude owns the active Eko worktree. Coordinate file ownership before edits and
do not delete another agent's locked worktree. If the agent bus is unavailable,
use the local handoff request and label acknowledgment as pending until received.
No radio scan, pairing, patient capture, firmware change or medical validation is
implied by this source review.
