use anyhow::{Context, Result};
use minijinja::Environment;
use serde_json::Value;
use std::{fs, path::Path};

pub fn load_chat_template(model_dir: &Path) -> Result<String> {
    let template_path = model_dir.join("chat_template.jinja");
    fs::read_to_string(&template_path)
        .with_context(|| format!("failed to read chat template: {}", template_path.display()))
}

pub fn render_chat_template(
    template: &str,
    messages: &Value,
    tools: &Value,
    add_generation_prompt: bool,
) -> Result<String> {
    let mut env = Environment::new();
    env.add_template("chat", template)?;
    let tmpl = env.get_template("chat")?;
    let context = serde_json::json!({
        "messages": messages,
        "tools": tools,
        "add_generation_prompt": add_generation_prompt
    });
    Ok(tmpl.render(context)?)
}

pub fn render_chat_prompt(
    model_dir: &Path,
    messages: &Value,
    tools: &Value,
    add_generation_prompt: bool,
) -> Result<String> {
    let template = load_chat_template(model_dir)?;
    render_chat_template(&template, messages, tools, add_generation_prompt)
}
