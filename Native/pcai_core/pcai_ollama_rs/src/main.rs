use std::fs;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Resolve the Ollama API base URL from environment or well-known default.
fn default_ollama_url() -> String {
    std::env::var("OLLAMA_HOST").unwrap_or_else(|_| {
        // Ollama's documented default; overridden via OLLAMA_HOST env var or --base-url flag.
        format!("http://{}:{}", "127.0.0.1", 11434)
    })
}

use anyhow::{anyhow, bail, Context, Result};
use mimalloc::MiMalloc;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::chat::{ChatMessage, MessageRole};
use ollama_rs::generation::parameters::{KeepAlive, TimeUnit};
use ollama_rs::generation::tools::{ToolCall, ToolCallFunction, ToolInfo};
use ollama_rs::models::{LocalModel, ModelOptions};
use ollama_rs::Ollama;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::{Host, Url};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, Deserialize, Default)]
struct RootConfig {
    #[serde(default)]
    ollama: OllamaConfig,
    #[serde(default)]
    providers: ProvidersConfig,
    #[serde(default)]
    router: RouterConfig,
}

#[derive(Debug, Deserialize, Default)]
struct ProvidersConfig {
    #[serde(default, rename = "ollama")]
    ollama_provider: ProviderConfig,
}

#[derive(Debug, Deserialize, Default)]
struct ProviderConfig {
    #[serde(default, rename = "baseUrl")]
    base_url: String,
    #[serde(default, rename = "defaultModel")]
    default_model: String,
    #[serde(default)]
    timeout: u64,
}

#[derive(Debug, Deserialize, Default)]
struct RouterConfig {
    #[serde(default, rename = "toolsPath")]
    tools_path: String,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct OllamaConfig {
    #[serde(default)]
    model: String,
    #[serde(default, rename = "base_url")]
    base_url: String,
    #[serde(default, rename = "timeout_ms")]
    timeout_ms: u64,
    #[serde(default)]
    temperature: f32,
    #[serde(default, rename = "num_ctx")]
    num_ctx: u64,
    #[serde(default, rename = "num_gpu")]
    num_gpu: i32,
    #[serde(default, rename = "num_thread")]
    num_thread: u32,
    #[serde(default, rename = "num_predict")]
    num_predict: i32,
    #[serde(default, rename = "keep_alive_seconds")]
    keep_alive_seconds: i64,
    #[serde(default, rename = "adaptive_ctx_enabled")]
    adaptive_ctx_enabled: bool,
    #[serde(default, rename = "adaptive_ctx_min")]
    adaptive_ctx_min: u64,
    #[serde(default, rename = "adaptive_ctx_max")]
    adaptive_ctx_max: u64,
    #[serde(default, rename = "adaptive_ctx_chars_per_token")]
    adaptive_ctx_chars_per_token: usize,
    #[serde(default, rename = "adaptive_ctx_base_headroom")]
    adaptive_ctx_base_headroom: usize,
    #[serde(default, rename = "adaptive_ctx_step_tokens")]
    adaptive_ctx_step_tokens: u64,
    #[serde(default, rename = "top_p")]
    top_p: f32,
    #[serde(default, rename = "top_k")]
    top_k: u32,
    #[serde(default, rename = "repeat_last_n")]
    repeat_last_n: i32,
    #[serde(default, rename = "repeat_penalty")]
    repeat_penalty: f32,
    #[serde(default, rename = "tfs_z")]
    tfs_z: f32,
    #[serde(default, rename = "seed")]
    seed: i32,
    #[serde(default, rename = "auto_pull_missing_models")]
    auto_pull_missing_models: bool,
    #[serde(default, rename = "strict_model_selection")]
    strict_model_selection: bool,
    #[serde(default, rename = "toolInvokerPath")]
    tool_invoker_path: String,
}

impl OllamaConfig {
    fn apply_defaults(&mut self, providers: &ProvidersConfig) {
        if self.base_url.trim().is_empty() {
            self.base_url = if !providers.ollama_provider.base_url.trim().is_empty() {
                providers.ollama_provider.base_url.clone()
            } else {
                default_ollama_url()
            };
        }
        if self.model.trim().is_empty() {
            self.model = if !providers.ollama_provider.default_model.trim().is_empty() {
                providers.ollama_provider.default_model.clone()
            } else {
                "qwen2.5-coder:3b".to_string()
            };
        }
        if self.timeout_ms == 0 {
            self.timeout_ms = if providers.ollama_provider.timeout > 0 {
                providers.ollama_provider.timeout
            } else {
                90_000
            };
        }
        if self.temperature == 0.0 {
            self.temperature = 0.15;
        }
        if self.num_ctx == 0 {
            // Default context window. Ollama manages GPU memory — let it use
            // the model's full context if possible, fall back if OOM.
            self.num_ctx = 32768;
        }
        if self.num_predict == 0 {
            // Allow substantial output by default. -1 = unlimited (until EOS).
            self.num_predict = 16384;
        }
        if self.keep_alive_seconds == 0 {
            self.keep_alive_seconds = 1800;
        }
        if self.adaptive_ctx_min == 0 {
            self.adaptive_ctx_min = 8192;
        }
        if self.adaptive_ctx_max == 0 {
            self.adaptive_ctx_max = self.num_ctx;
        }
        if self.adaptive_ctx_chars_per_token == 0 {
            self.adaptive_ctx_chars_per_token = 4;
        }
        if self.adaptive_ctx_base_headroom == 0 {
            self.adaptive_ctx_base_headroom = 1024;
        }
        if self.adaptive_ctx_step_tokens == 0 {
            self.adaptive_ctx_step_tokens = 4096;
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct HealthResponse {
    ok: bool,
    base_url: String,
    configured_model: String,
    resolved_model: String,
    available_models: Vec<String>,
    runner: &'static str,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelsResponse {
    ok: bool,
    models: Vec<String>,
    runner: &'static str,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_tokens: Option<i32>,
    #[serde(default, rename = "numCtx")]
    num_ctx: Option<u64>,
    #[serde(default, rename = "numThread")]
    num_thread: Option<u32>,
    #[serde(default, rename = "topP")]
    top_p: Option<f32>,
    #[serde(default, rename = "topK")]
    top_k: Option<u32>,
    #[serde(default, rename = "repeatLastN")]
    repeat_last_n: Option<i32>,
    #[serde(default, rename = "repeatPenalty")]
    repeat_penalty: Option<f32>,
    #[serde(default, rename = "tfsZ")]
    tfs_z: Option<f32>,
    #[serde(default, rename = "seed")]
    seed: Option<i32>,
    #[serde(default)]
    enable_tools: bool,
    #[serde(default)]
    max_tool_rounds: usize,
    #[serde(default)]
    tools_path: Option<String>,
    messages: Vec<RequestMessage>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct RequestMessage {
    role: String,
    content: String,
    #[serde(default)]
    tool_calls: Vec<RequestToolCall>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct RequestToolCall {
    name: String,
    arguments: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ChatResponse {
    ok: bool,
    provider: &'static str,
    model: String,
    content: String,
    tool_calls: Vec<RequestToolCall>,
    executed_tools: Vec<ExecutedTool>,
    timing: Option<TimingSummary>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ExecutedTool {
    name: String,
    arguments: Value,
    result: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TimingSummary {
    total_duration_ns: u64,
    prompt_eval_count: u64,
    eval_count: u64,
    eval_duration_ns: u64,
}

fn read_json_text(path: &Path) -> Result<String> {
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Ok(content.trim_start_matches('\u{feff}').to_string())
}

#[tokio::main]
async fn main() {
    let exit_code = match run().await {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("{err:#}");
            1
        }
    };
    std::process::exit(exit_code);
}

async fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(String::as_str).unwrap_or("");
    if command.is_empty() {
        bail!("usage: pcai-ollama-rs <health|models|chat> --config <path> [--request-file <path>]");
    }

    let config_path = resolve_config_path(arg_value(&args, "--config"))?;
    let repo_root = config_path
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| anyhow!("failed to resolve repo root from config path"))?
        .to_path_buf();
    let config = load_config(&config_path)?;
    let client = build_client(&config.ollama.base_url)?;

    match command {
        "health" => {
            let models = client.list_local_models().await.unwrap_or_default();
            let model_names = model_names(&models);
            let resolved_model = resolve_model_name(&config.ollama, &model_names, None, false)
                .await
                .unwrap_or_else(|_| config.ollama.model.clone());
            let response = HealthResponse {
                ok: !model_names.is_empty(),
                base_url: config.ollama.base_url.clone(),
                configured_model: config.ollama.model.clone(),
                resolved_model,
                available_models: model_names,
                runner: "ollama-rs",
                error: None,
            };
            println!("{}", serde_json::to_string(&response)?);
        }
        "models" => {
            let models = client.list_local_models().await?;
            let response = ModelsResponse {
                ok: true,
                models: model_names(&models),
                runner: "ollama-rs",
            };
            println!("{}", serde_json::to_string(&response)?);
        }
        "chat" => {
            let request_path = arg_value(&args, "--request-file")
                .map(PathBuf::from)
                .ok_or_else(|| anyhow!("chat requires --request-file"))?;
            let request: ChatRequest = serde_json::from_str(
                &read_json_text(&request_path)
                    .with_context(|| format!("read chat request file: {}", request_path.display()))?,
            )?;

            let response = run_chat(&client, &config, &repo_root, request).await?;
            println!("{}", serde_json::to_string(&response)?);
        }
        other => bail!("unsupported command: {other}"),
    }

    Ok(())
}

fn resolve_config_path(arg: Option<String>) -> Result<PathBuf> {
    if let Some(path) = arg {
        return Ok(PathBuf::from(path));
    }

    let mut current = std::env::current_dir().context("resolve current directory")?;
    loop {
        let candidate = current.join("Config").join("llm-config.json");
        if candidate.exists() {
            return Ok(candidate);
        }
        if !current.pop() {
            break;
        }
    }

    bail!("could not locate Config/llm-config.json; pass --config explicitly")
}

fn load_config(path: &Path) -> Result<RootConfig> {
    let mut config: RootConfig =
        serde_json::from_str(&read_json_text(path)?).with_context(|| format!("parse {}", path.display()))?;
    config.ollama.apply_defaults(&config.providers);
    Ok(config)
}

fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|window| window[0] == name)
        .map(|window| window[1].clone())
}

/// Parse an Ollama endpoint, tolerating the scheme-less `host:port` form.
///
/// `OLLAMA_HOST` is conventionally written the way Ollama documents it --
/// `0.0.0.0:11434`, with no scheme -- which `Url::parse` rejects outright.
/// The `has_host` guard matters for the other direction: `localhost:11434`
/// *does* parse, as scheme `localhost` with path `11434` and no host at all,
/// so accepting it unchecked would yield a client pointed at nothing.
fn parse_ollama_url(raw: &str) -> Option<Url> {
    let accept = |url: Url| url.has_host().then_some(url);
    Url::parse(raw)
        .ok()
        .and_then(accept)
        .or_else(|| Url::parse(&format!("http://{raw}")).ok().and_then(accept))
}

/// `OLLAMA_HOST` does double duty: it tells the server what to bind and tells
/// clients where to connect. `0.0.0.0` is the unspecified address -- meaningful
/// as "bind every interface", not as a destination -- so a client reading it
/// verbatim is relying on the OS to reinterpret it. Resolve it to loopback.
fn client_host(host: Host<&str>) -> String {
    match host {
        Host::Ipv4(addr) if addr.is_unspecified() => Ipv4Addr::LOCALHOST.to_string(),
        Host::Ipv6(addr) if addr.is_unspecified() => format!("[{}]", Ipv6Addr::LOCALHOST),
        other => other.to_string(),
    }
}

/// Split an Ollama base URL into the `scheme://host` and port that the client
/// builder expects, falling back to the default URL if `base_url` is unusable.
///
/// Uses `Url::host()` rather than `host_str()`: the latter returns a bare `::1`
/// for an IPv6 literal, so pasting it into `"{}://{}"` yields `http://::1`,
/// which is not a valid URL. `Host`'s `Display` re-adds the brackets.
fn split_base_url(base_url: &str) -> Result<(String, u16)> {
    let parsed = parse_ollama_url(base_url)
        .or_else(|| parse_ollama_url(&default_ollama_url()))
        .ok_or_else(|| anyhow!("neither '{base_url}' nor OLLAMA_HOST is a usable Ollama endpoint"))?;
    let host = parsed
        .host()
        .map(|host| format!("{}://{}", parsed.scheme(), client_host(host)))
        .unwrap_or_else(|| format!("{}://{}", parsed.scheme(), Ipv4Addr::LOCALHOST));
    Ok((host, parsed.port().unwrap_or(11434)))
}

fn build_client(base_url: &str) -> Result<Ollama> {
    let (host, port) = split_base_url(base_url)?;
    // ollama-rs 0.3.6 deprecated `Ollama::new` in favour of the builder, and
    // `-D warnings` turns that deprecation into a build error. `build()`
    // returns `Ollama` directly, not a Result.
    Ok(Ollama::builder().host(host).port(port).build())
}

fn model_names(models: &[LocalModel]) -> Vec<String> {
    models.iter().map(|model| model.name.clone()).collect()
}

async fn resolve_model_name(
    config: &OllamaConfig,
    local_models: &[String],
    requested: Option<&str>,
    allow_pull: bool,
) -> Result<String> {
    let requested = requested
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(config.model.as_str());

    if let Some(found) = match_model_name(local_models, requested) {
        return Ok(found.to_string());
    }

    if config.auto_pull_missing_models && allow_pull {
        let client = build_client(&config.base_url)?;
        client
            .pull_model(requested.to_string(), false)
            .await
            .with_context(|| format!("pull missing ollama model {requested}"))?;
        let refreshed = model_names(&client.list_local_models().await?);
        if let Some(found) = match_model_name(&refreshed, requested) {
            return Ok(found.to_string());
        }
    }

    if config.strict_model_selection {
        bail!("configured ollama model '{requested}' is not installed locally")
    }

    if let Some(first) = local_models.first() {
        return Ok(first.clone());
    }

    bail!("no local Ollama models are installed")
}

fn match_model_name<'a>(local_models: &'a [String], requested: &str) -> Option<&'a str> {
    if let Some(found) = local_models
        .iter()
        .find(|candidate| candidate.eq_ignore_ascii_case(requested))
    {
        return Some(found.as_str());
    }

    let requested_family = requested.split(':').next().unwrap_or(requested);
    local_models
        .iter()
        .find(|candidate| {
            let family = candidate.split(':').next().unwrap_or(candidate.as_str());
            family.eq_ignore_ascii_case(requested_family)
        })
        .map(|candidate| candidate.as_str())
}

async fn run_chat(
    client: &Ollama,
    config: &RootConfig,
    repo_root: &Path,
    request: ChatRequest,
) -> Result<ChatResponse> {
    let local_models = model_names(&client.list_local_models().await?);
    let resolved_model = resolve_model_name(&config.ollama, &local_models, request.model.as_deref(), true).await?;
    let tools_path = request
        .tools_path
        .as_ref()
        .map(|path| resolve_repo_path(repo_root, path))
        .or_else(|| {
            if config.router.tools_path.trim().is_empty() {
                None
            } else {
                Some(resolve_repo_path(repo_root, &config.router.tools_path))
            }
        })
        .unwrap_or_else(|| repo_root.join("Config").join("pcai-tools.json"));
    let tool_invoker_path = resolve_repo_path(repo_root, &config.ollama.tool_invoker_path);
    let tools = if request.enable_tools && tools_path.exists() {
        load_tools(&tools_path)?
    } else {
        Vec::new()
    };

    let mut messages = convert_messages(request.messages.clone());
    let mut executed_tools = Vec::new();
    let max_tool_rounds = request.max_tool_rounds.max(1);

    for round in 0..max_tool_rounds {
        let prompt_chars = messages.iter().map(|message| message.content.len()).sum::<usize>();
        let mut chat_request = ChatMessageRequest::new(resolved_model.clone(), messages.clone())
            .options(build_options(config, &request, prompt_chars));
        chat_request = apply_keep_alive(chat_request, config.ollama.keep_alive_seconds);
        if !tools.is_empty() {
            chat_request = chat_request.tools(tools.clone());
        }

        let response = client
            .send_chat_messages(chat_request)
            .await
            .context("ollama-rs chat request failed")?;
        let tool_calls = response
            .message
            .tool_calls
            .iter()
            .map(|call| RequestToolCall {
                name: call.function.name.clone(),
                arguments: call.function.arguments.clone(),
            })
            .collect::<Vec<_>>();
        let timing = response.final_data.as_ref().map(|data| TimingSummary {
            total_duration_ns: data.total_duration,
            prompt_eval_count: data.prompt_eval_count,
            eval_count: data.eval_count,
            eval_duration_ns: data.eval_duration,
        });
        let msg = response.message;

        // Return early when no more tool rounds are needed.  Move `content`
        // out of `msg` instead of cloning — `messages` is not updated on this
        // path since the loop is exiting anyway.
        if tool_calls.is_empty() || tools.is_empty() || round + 1 >= max_tool_rounds {
            return Ok(ChatResponse {
                ok: true,
                provider: "ollama",
                model: resolved_model,
                content: msg.content,
                tool_calls,
                executed_tools,
                timing,
                error: None,
            });
        }

        // Only push the assistant turn when the loop continues with tool calls.
        messages.push(msg);

        for call in tool_calls {
            let result = invoke_pcai_tool(repo_root, &tool_invoker_path, &tools_path, &call.name, &call.arguments)?;
            executed_tools.push(ExecutedTool {
                name: call.name,
                arguments: call.arguments,
                result: result.clone(),
            });
            messages.push(ChatMessage {
                role: MessageRole::Tool,
                content: result,
                tool_calls: Vec::<ToolCall>::new(),
                images: None,
                thinking: None,
            });
        }
    }

    bail!("ollama tool loop exhausted without producing a final response")
}

fn convert_messages(messages: Vec<RequestMessage>) -> Vec<ChatMessage> {
    messages
        .into_iter()
        .map(|message| ChatMessage {
            role: match message.role.as_str() {
                "assistant" => MessageRole::Assistant,
                "system" => MessageRole::System,
                "tool" => MessageRole::Tool,
                _ => MessageRole::User,
            },
            content: message.content,
            tool_calls: message
                .tool_calls
                .into_iter()
                .map(|call| ToolCall {
                    function: ToolCallFunction {
                        name: call.name,
                        arguments: call.arguments,
                    },
                })
                .collect(),
            images: None,
            thinking: None,
        })
        .collect()
}

fn build_options(config: &RootConfig, request: &ChatRequest, prompt_chars: usize) -> ModelOptions {
    let mut options = ModelOptions::default();
    let num_ctx = request
        .num_ctx
        .unwrap_or_else(|| select_num_ctx(&config.ollama, prompt_chars));
    options = options.num_ctx(num_ctx);

    let temperature = request.temperature.unwrap_or(config.ollama.temperature);
    options = options.temperature(temperature);

    let max_tokens = request.max_tokens.unwrap_or(config.ollama.num_predict);
    options = options.num_predict(max_tokens);

    // num_gpu: -1 = all layers on GPU (Ollama convention), 0 = omit (Ollama
    // defaults to full offload), >0 = explicit layer count.  The ollama-rs
    // ModelOptions::num_gpu takes u32, so map -1 to 999 (Ollama caps at the
    // actual layer count).
    if config.ollama.num_gpu != 0 {
        let gpu_layers = if config.ollama.num_gpu < 0 {
            999_u32
        } else {
            config.ollama.num_gpu as u32
        };
        options = options.num_gpu(gpu_layers);
    }
    let num_thread = request.num_thread.unwrap_or(config.ollama.num_thread);
    if num_thread > 0 {
        options = options.num_thread(num_thread);
    }
    let top_p = request.top_p.unwrap_or(config.ollama.top_p);
    if top_p.is_finite() && top_p > 0.0 && top_p <= 1.0 {
        options = options.top_p(top_p);
    }
    let top_k = request.top_k.unwrap_or(config.ollama.top_k);
    if top_k > 0 {
        options = options.top_k(top_k);
    }
    let repeat_last_n = request.repeat_last_n.unwrap_or(config.ollama.repeat_last_n);
    if repeat_last_n != 0 {
        options = options.repeat_last_n(repeat_last_n);
    }
    let repeat_penalty = request.repeat_penalty.unwrap_or(config.ollama.repeat_penalty);
    if repeat_penalty.is_finite() && repeat_penalty > 0.0 {
        options = options.repeat_penalty(repeat_penalty);
    }
    let tfs_z = request.tfs_z.unwrap_or(config.ollama.tfs_z);
    if tfs_z.is_finite() && tfs_z > 0.0 {
        options = options.tfs_z(tfs_z);
    }
    let seed = request.seed.unwrap_or(config.ollama.seed);
    if seed != 0 {
        options = options.seed(seed);
    }

    options
}

fn select_num_ctx(config: &OllamaConfig, prompt_chars: usize) -> u64 {
    if !config.adaptive_ctx_enabled {
        return config.num_ctx.max(1024);
    }

    let chars_per_token = config.adaptive_ctx_chars_per_token.max(1);
    let prompt_tokens = prompt_chars.div_ceil(chars_per_token) as u64;
    let requested = prompt_tokens
        .saturating_add(config.num_predict.max(0) as u64)
        .saturating_add(config.adaptive_ctx_base_headroom as u64);
    let rounded = round_up_to_step(requested, config.adaptive_ctx_step_tokens.max(1));
    rounded.clamp(
        config.adaptive_ctx_min.max(1024),
        config.adaptive_ctx_max.max(config.adaptive_ctx_min.max(1024)),
    )
}

fn round_up_to_step(value: u64, step: u64) -> u64 {
    if step <= 1 {
        return value;
    }
    let rem = value % step;
    if rem == 0 {
        value
    } else {
        value + (step - rem)
    }
}

fn apply_keep_alive(request: ChatMessageRequest, keep_alive_seconds: i64) -> ChatMessageRequest {
    match keep_alive_seconds {
        i64::MIN..=-1 => request.keep_alive(KeepAlive::Indefinitely),
        0 => request.keep_alive(KeepAlive::UnloadOnCompletion),
        seconds => request.keep_alive(KeepAlive::Until {
            time: seconds as u64,
            unit: TimeUnit::Seconds,
        }),
    }
}

fn load_tools(path: &Path) -> Result<Vec<ToolInfo>> {
    let content = read_json_text(path)?;
    let catalog: Value =
        serde_json::from_str(&content).with_context(|| format!("parse tool catalog {}", path.display()))?;
    let mut tools_value = catalog
        .get("tools")
        .cloned()
        .ok_or_else(|| anyhow!("tool catalog missing top-level 'tools' array"))?;
    if let Some(items) = tools_value.as_array_mut() {
        for item in items {
            if let Some(raw_type) = item.get("type").and_then(Value::as_str) {
                if raw_type.eq_ignore_ascii_case("function") {
                    item["type"] = Value::String("Function".to_string());
                }
            }
        }
    }
    let tools: Vec<ToolInfo> =
        serde_json::from_value(tools_value).context("deserialize tools into ollama-rs format")?;
    Ok(tools)
}

fn invoke_pcai_tool(
    repo_root: &Path,
    tool_invoker_path: &Path,
    tools_path: &Path,
    tool_name: &str,
    arguments: &Value,
) -> Result<String> {
    let args_file = std::env::temp_dir().join(format!(
        "pcai-tool-{}.json",
        tool_name.replace(|ch: char| !ch.is_ascii_alphanumeric(), "_")
    ));
    fs::write(&args_file, serde_json::to_vec(arguments)?).with_context(|| format!("write {}", args_file.display()))?;

    let output = Command::new("pwsh")
        .arg("-NoProfile")
        .arg("-File")
        .arg(tool_invoker_path)
        .arg("-RepoRoot")
        .arg(repo_root)
        .arg("-ToolsPath")
        .arg(tools_path)
        .arg("-ToolName")
        .arg(tool_name)
        .arg("-ArgumentsPath")
        .arg(&args_file)
        .output()
        .with_context(|| format!("run tool invoker for {tool_name}"))?;

    let _ = fs::remove_file(&args_file);

    if !output.status.success() {
        bail!(
            "tool invoker failed for {}: {}",
            tool_name,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let response: Value = serde_json::from_slice(&output.stdout).context("parse tool invoker response json")?;
    if response.get("ok").and_then(Value::as_bool).unwrap_or(false) {
        Ok(response
            .get("result")
            .map(|value| {
                if let Some(text) = value.as_str() {
                    text.to_string()
                } else {
                    value.to_string()
                }
            })
            .unwrap_or_default())
    } else {
        bail!(
            "tool '{}' returned error: {}",
            tool_name,
            response.get("error").and_then(Value::as_str).unwrap_or("unknown error")
        )
    }
}

fn resolve_repo_path(repo_root: &Path, raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        repo_root.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::split_base_url;

    #[test]
    fn splits_ipv4_host_and_explicit_port() {
        let (host, port) = split_base_url("http://127.0.0.1:11434").unwrap();
        assert_eq!(host, "http://127.0.0.1");
        assert_eq!(port, 11434);
    }

    #[test]
    fn keeps_brackets_around_ipv6_literals() {
        // host_str() returns a bare `::1` here, which would produce `http://::1`.
        let (host, port) = split_base_url("http://[::1]:11434").unwrap();
        assert_eq!(host, "http://[::1]");
        assert_eq!(port, 11434);
    }

    #[test]
    fn defaults_the_port_when_the_url_omits_it() {
        let (host, port) = split_base_url("http://ollama.internal").unwrap();
        assert_eq!(host, "http://ollama.internal");
        assert_eq!(port, 11434);
    }

    #[test]
    fn preserves_a_non_default_scheme_and_port() {
        let (host, port) = split_base_url("https://gpu-box.lan:8443").unwrap();
        assert_eq!(host, "https://gpu-box.lan");
        assert_eq!(port, 8443);
    }

    #[test]
    fn accepts_the_scheme_less_form_ollama_documents() {
        // `OLLAMA_HOST=0.0.0.0:11434` is how Ollama documents it, and it is what
        // this workstation actually sets. `Url::parse` rejects it outright, so
        // before the http:// retry this returned Err and no client could be built.
        // 0.0.0.0 is a bind address, so it resolves to loopback for a client.
        let (host, port) = split_base_url("0.0.0.0:11434").unwrap();
        assert_eq!(host, "http://127.0.0.1");
        assert_eq!(port, 11434);
    }

    #[test]
    fn resolves_the_unspecified_ipv6_address_to_loopback() {
        let (host, port) = split_base_url("http://[::]:11434").unwrap();
        assert_eq!(host, "http://[::1]");
        assert_eq!(port, 11434);
    }

    #[test]
    fn leaves_a_routable_address_alone() {
        // Only the unspecified address is rewritten; a real remote host is not.
        let (host, port) = split_base_url("http://192.168.50.79:11434").unwrap();
        assert_eq!(host, "http://192.168.50.79");
        assert_eq!(port, 11434);
    }

    #[test]
    fn treats_localhost_port_as_a_host_not_a_scheme() {
        // `localhost:11434` parses cleanly as scheme `localhost`, path `11434`,
        // and *no host*. Without the has_host guard it would be accepted and
        // point the client at nothing.
        let (host, port) = split_base_url("localhost:11434").unwrap();
        assert_eq!(host, "http://localhost");
        assert_eq!(port, 11434);
    }

    #[test]
    fn falls_back_to_the_default_url_when_parsing_fails() {
        // Not a URL at all, and not rescuable by prefixing a scheme either,
        // so the OLLAMA_HOST/default fallback branch is taken.
        let (host, port) = split_base_url("not a url").unwrap();
        assert!(host.starts_with("http://"), "unexpected fallback host: {host}");
        assert!(port > 0);
    }
}
