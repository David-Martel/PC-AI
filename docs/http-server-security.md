# HTTP server security: Host allow-list, CORS, bearer token and bind rules

This page covers `pcai-inference` (the OpenAI-compatible LLM server,
`Native/pcai_core/pcai_inference/src/http/mod.rs`) and `pcai-media`
(`Native/pcai_core/pcai_media_server/src/main.rs`). Both are wrapped by one shared crate,
`Native/pcai_core/pcai_http_guard`, so they enforce the same rules. The FunctionGemma runtime
(`Deploy/rust-functiongemma-runtime`) is a separate service and is not covered here.

## Why these servers need it

Both servers listen on loopback and have no login. Loopback is not a boundary against the browser:

- **CSRF.** Any web page the user visits can send requests to `http://127.0.0.1:<port>`. The old
  permissive CORS layer answered every origin with `Access-Control-Allow-Origin: *`, so the page
  could also read the responses, and it answered the preflight that lets a page send a JSON POST.
- **DNS rebinding.** An attacker's domain can be re-pointed at `127.0.0.1` after the page loaded.
  The browser then treats the API as same-origin and CORS never applies. The only trace is the
  `Host` header, which still names the attacker's domain.
- **Network exposure.** `server.host` could be set to `0.0.0.0` with no credential at all.

The fix is three independent checks plus strict bind handling. Each stands on its own.

## What is enforced

A request passes these layers, outermost first (`SecurityPolicy::apply`):

| Layer | Rejects with | Detail |
|---|---|---|
| `Host` allow-list | `421` `host_not_allowed`; `400` `invalid_host` | Compares the host part, case-insensitively, ignoring the port. Allowed: `localhost`, `127.0.0.1`, `::1` (also `[::1]`), the bind address, and `allowed_hosts`. Any port is accepted, because the HVSocket proxy forwards 18080 to the server port (`Config/hvsock-proxy.conf`). |
| `Origin` check | `403` `origin_not_allowed` | A request that carries an `Origin` header comes from a web page. It is refused unless the origin is in `cors_allowed_origins`. This also stops cross-site "simple" requests, which are delivered without a preflight. |
| CORS | no CORS headers | Added only when `cors_allowed_origins` is non-empty. It echoes a listed origin and answers its preflight; it never sends `*`. |
| Bearer token | `401` `invalid_token` | Only when a token is configured. Every route except `GET /health` needs `Authorization: Bearer <token>`. The comparison is constant-time (`subtle`). |

Details that are easy to get wrong:

- **Missing `Host`.** HTTP/1.1 requires the header, so its absence is a malformed request (`400`).
  An HTTP/1.0 request without `Host` is accepted: browsers never speak HTTP/1.0, so it cannot be a
  rebinding page, and old tools use it. For HTTP/2 and absolute-form requests the URI authority is
  checked as well, and both values must be allowed. Two `Host` headers are refused.
- **Wildcard binds.** `0.0.0.0` and `::` are never accepted as a `Host`, and are refused in
  `allowed_hosts`. Binding a wildcard requires `allowed_hosts` that names what clients use.
- **JSON content type.** The JSON routes also reject non-JSON bodies with `415`. That is a second
  barrier against form posts and `text/plain` fetches, pinned by a test.
- **Preflight sits outside the token check**, because a browser sends no credentials on it.

## Configuration

`pcai-inference` reads these from the `server` object of its JSON config:

| Key | Default | Meaning |
|---|---|---|
| `host` | `127.0.0.1` | An IP literal: IPv4, IPv6 with or without brackets, or scoped link-local such as `fe80::1%12` or `fe80::1%eth0`. `localhost` means `127.0.0.1`. Other names are refused. |
| `port` | `8080` | Listening port. |
| `cors_allowed_origins` | `[]` | Browser origins allowed to call the API, e.g. `["https://tools.example:3000"]`. Empty: no CORS headers, browser callers refused. `*`, paths and other schemes are refused at startup. |
| `allowed_hosts` | `[]` | Extra `Host` values. Required for a wildcard bind. |
| `auth_token` | none | Bearer token. Never written by `to_file` and redacted in `Debug`. |
| `cors` | ignored | Legacy key. Old files still load, but `true` no longer enables anything; a warning is logged. |

The environment variable `PCAI_INFERENCE_TOKEN` overrides `auth_token`. Blank values (empty or
whitespace) count as unset, so an empty variable can never switch authentication off or unlock a
network bind.

`pcai-media` has no config file. It takes `--host`, `--port`, `--allowed-host <host>` (repeatable)
and `--cors-origin <origin>` (repeatable), and reads the token only from `PCAI_MEDIA_TOKEN`. There
is deliberately no `--token` flag: arguments show up in process listings.

## Bind rules

- The address is parsed, never formatted: `127.0.0.1`, `::1`, `[::1]` and `fe80::1%12` all work.
  A numeric scope becomes the socket `scope_id`; an interface name is resolved with
  `if_nametoindex`. Prefer the numeric form on Windows.
- **A non-loopback bind without a token refuses to start**, with a message naming the address.
  Loopback means `127.0.0.0/8`, `::1` and IPv4-mapped loopback.
- **A bind failure exits nonzero naming the address.** There is no fallback to another port (the
  same rule FunctionGemma adopted in #408).
- `pcai-media` validates all of this before loading the model.

## Client compatibility

In-repo clients send `Host: 127.0.0.1:<port>` and no `Origin`, so they need no change when no
token is configured. When a token is configured:

- One helper, `Get-PcaiOriginBearerHeader` (`Modules/PC-AI.Common/Public`), reads the token from the
  named environment variable and returns the `Authorization: Bearer` header only when the request
  URL has the same origin (scheme, host, port) as the server's configured URL. A request to any
  other origin, such as Ollama, LM Studio or vLLM, gets no header. Three callers use it:
  - `Modules/PC-AI.LLM` (`PCAI_INFERENCE_TOKEN`): the `pcai-inference` connection probe and
    `Invoke-OpenAIChat` / `Invoke-OpenAIChatWithProgress`, via `Get-PcaiInferenceAuthHeader`.
  - The evaluation harness (`Invoke-SingleTestCase`, `PCAI_INFERENCE_TOKEN`): the `/v1/completions`
    and `/v1/chat/completions` requests, for the configured pcai-inference URL and for the
    `llamacpp-bin` / `mistralrs-bin` servers it starts itself. An `http` backend pointed at any
    other origin, and the `ollama` backend, get no header.
  - The media CUDA hardware runner (`Tests/Hardware/Invoke-PcaiMediaCudaFunctionalTests.ps1`,
    `PCAI_MEDIA_TOKEN`): its generate and understand POSTs to the child server it launches. The
    child inherits the variable, and its `/health` readiness probe stays unauthenticated.
- The streaming helpers (`Invoke-OpenAIChatStream`, `Invoke-OpenAICompletionStream`), the
  `PcaiNative` C# clients and `PcaiChatTui` do not send a token yet. Use a token-free loopback
  server with them for now.

## Verification

```powershell
cd Native\pcai_core
cargo test -p pcai_http_guard
cargo test --no-default-features --features server,ffi -p pcai-inference --lib hardening
cargo test -p pcai-media-server hardening
```

Related: [CLAUDE.md](../CLAUDE.md) (endpoints), `Config/hvsock-proxy.conf`,
`Native/pcai_core/pcai_http_guard/src/lib.rs`.
