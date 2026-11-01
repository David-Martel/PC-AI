# Native Ollama residency contract

Reviewed October 10, 2026; owner: Codex integration lane.

The native configuration normalizer replaced an explicit
`ollama.keep_alive_seconds: 0` with 1800. This prevented the existing request
builder from sending Ollama's unload-on-completion value. The repair gives
omitted fields, omitted sections and `Default` an explicit 1800-second default,
then preserves configured zero, positive durations and negative indefinite
residency. Other configuration defaults and request options are unchanged.

Eight maintained controls call real configuration deserialization, file loading,
normalization and the locked ollama-rs request serializer. They cover missing,
zero, positive, negative and invalid values, repeated normalization, and
unchanged messages/options. The original implementation passed four and failed
four. The genuinely recompiled repair passed all eight; the complete binary
suite passed 17 with no failures, ignored tests or filtered tests. Focused
formatting and all-target package clippy with `-D warnings` passed.

QA used CargoTools' raw route, locked offline dependencies, two compiler workers
and private output/temp on D:. All 304 other preserved workspace/config/wrapper
files retained their hashes. The first rejected Cargo configuration, a guarded
output collision and a stale-binary candidate failure are preserved separately.
After detecting Cargo's reuse of the old executable, QA refreshed only the
owned source mtime, confirmed an actual compilation and changed binary hash,
then reran the unchanged controls. These are ordinary command exit receipts;
they do not assert native supervisor custody qualification.

Frozen actual evidence: `D:/pcai-relocation/ollama-keepalive-validation-r1/actual-manifest.json`,
SHA256 `816C1A9C6AD9B9D59527B5BE65D0C8192A1C5CAF7692C2AB700D700FEC1576F2`.
The accepted main source is `984E7D626D26CB9CB71DAB505665CB65A4B40946E56BB3CF73C54D567C5A3D0A`;
the eight controls are `5E78292553FB5B5612D12E0FF4FEADA048B84E45238E0DEC83868DA2752EE42A`.
Rust Guidelines CI now selects the binary separately because `--lib` excludes it.

Runtime and deployment remain open. After a supported idle-cache release,
observed system memory commit dropped from 87% to 79%; new requests later
reloaded the model. These operations did not stop the daemon and are not a
matched workload benchmark. A retained Vigil Audio camera-guidance
process, matching source endpoint/model and current route logs, is a strong
consumer lead; individual requests are not yet correlated to its PID. The
OpenClaw catalogue alone did not identify an active requester. Preserve the live
camera session and obtain its owner handoff before changing its lifecycle.
This source repair does not establish that the requesting consumer uses the
native runner, or demonstrate lasting memory or responsiveness improvements.
