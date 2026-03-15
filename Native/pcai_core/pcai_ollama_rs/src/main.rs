use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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
use url::Url;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, Deserialize, Default)]
struct RootConfig {
    #[serde(default, rename = "fallbackOrder")]
    fallback_order: Vec<String>,
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
    enabled: bool,
    #[serde(default)]
    model: String,
    #[serde(default, rename = "tool_model")]
    tool_model: String,
    #[serde(default, rename = "summary_model")]
    summary_model: String,
    #[serde(default, rename = "base_url")]
    base_url: String,
    #[serde(default, rename = "timeout_ms")]
    timeout_ms: u64,
    #[serde(default)]
    temperature: f32,
    #[serde(default, rename = "num_ctx")]
    num_ctx: u64,
    #[serde(default, rename = "num_gpu")]
    num_gpu: u32,
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
    #[serde(default, rename = "auto_detect_models")]
    auto_detect_models: bool,
    #[serde(default, rename = "required_models")]
    required_models: Vec<String>,
    #[serde(default, rename = "warm_models_on_start")]
    warm_models_on_start: bool,
    #[serde(default, rename = "auto_pull_missing_models")]
    auto_pull_missing_models: bool,
    #[serde(default, rename = "strict_model_selection")]
    strict_model_selection: bool,
    #[serde(default, rename = "cliSearchPaths")]
    cli_search_paths: Vec<String>,
    #[serde(default, rename = "toolInvokerPath")]
    tool_invoker_path: String,
}

impl OllamaConfig {
    fn apply_defaults(&mut self, providers: &ProvidersConfig) {
        if self.base_url.trim().is_empty() {
            self.base_url = if !providers.ollama_provider.base_url.trim().is_empty() {
                providers.ollama_provider.base_url.clone()
            } else {
                "http://127.0.0.1:11434".to_string()
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
            self.num_ctx = 131_072;
        }
        if self.num_predict == 0 {
            self.num_predict = 1024;
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

#[derive(Debug, Deserialize, Serialize)]
struct ToolCatalog {
    tools: Vec<ConfiguredTool>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ConfiguredTool {
    function: ConfiguredFunction,
}

#[derive(Debug, Deserialize, Serialize)]
struct ConfiguredFunction {
    name: String,
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

fn build_client(base_url: &str) -> Result<Ollama> {
    let parsed = Url::parse(base_url).or_else(|_| Url::parse("http://127.0.0.1:11434"))?;
    let host = format!("{}://{}", parsed.scheme(), parsed.host_str().unwrap_or("127.0.0.1"));
    let port = parsed.port().unwrap_or(11434);
    Ok(Ollama::new(host, port))
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
        let content = response.message.content.clone();
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

        messages.push(response.message.clone());

        if tool_calls.is_empty() || tools.is_empty() || round + 1 >= max_tool_rounds {
            return Ok(ChatResponse {
                ok: true,
                provider: "ollama",
                model: resolved_model,
                content,
                tool_calls,
                executed_tools,
                timing,
                error: None,
            });
        }

        for call in tool_calls {
            let result = invoke_pcai_tool(repo_root, &tool_invoker_path, &tools_path, &call.name, &call.arguments)?;
            executed_tools.push(ExecutedTool {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
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

    if config.ollama.num_gpu > 0 {
        options = options.num_gpu(config.ollama.num_gpu);
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
