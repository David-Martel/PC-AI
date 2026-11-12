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
    Ok(tmpl.render(minijinja::value::Serde(context))?)
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

#[cfg(test)]
mod tests {
    use super::{render_chat_prompt, render_chat_template};
    use serde_json::json;

    // A template that reaches every part of the context the renderer builds: nested
    // message fields, a tool's nested JSON schema, and the generation-prompt flag.
    const TEMPLATE: &str = "{% for m in messages %}<{{ m.role }}>{{ m.content }}\
{% for c in m.tool_calls or [] %}[call {{ c.function.name }}({{ c.function.arguments.path }})]{% endfor %}\
</{{ m.role }}>{% endfor %}\
{% for t in tools %}{tool {{ t.function.name }}: {{ t.function.parameters.properties.path.type }}}{% endfor %}\
{% if add_generation_prompt %}<model>{% endif %}";

    fn messages() -> serde_json::Value {
        json!([
            {"role": "user", "content": "list the drive"},
            {"role": "model", "content": "", "tool_calls": [
                {"function": {"name": "list_dir", "arguments": {"path": "/mnt/c"}}}
            ]}
        ])
    }

    fn tools() -> serde_json::Value {
        json!([{"function": {
            "name": "list_dir",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
        }}])
    }

    // Protects: MiniJinja 3 rendering of the serde_json chat context. In 3.x a
    //   serde_json::Value is not Into<minijinja::Value>, so the context must go through
    //   value::Serde (feature "serde"); without it this crate does not compile (E0277).
    // Detects: nested message fields, tool calls, tool schemas or the generation
    //   flag being lost or flattened by the adapter.
    // Needs: pure rendering, no model or GPU. The crate itself builds only with its
    //   default `cuda` feature (cudarc and qlora_rs are used unconditionally), so a
    //   CPU-only run compiles this file alone against minijinja 3 (see the PR body).
    #[test]
    fn renders_nested_messages_tools_and_generation_prompt() {
        let out = render_chat_template(TEMPLATE, &messages(), &tools(), true).expect("render");
        assert_eq!(
            out,
            "<user>list the drive</user><model>[call list_dir(/mnt/c)]</model>\
{tool list_dir: string}<model>"
        );
        let no_prompt = render_chat_template(TEMPLATE, &messages(), &tools(), false).expect("render");
        assert!(!no_prompt.ends_with("<model>"), "{no_prompt}");
    }

    // A malformed template is an error, not an empty prompt.
    #[test]
    fn malformed_template_is_an_error() {
        assert!(render_chat_template("{% for m in messages %}", &messages(), &tools(), true).is_err());
    }

    // The file entry point reads chat_template.jinja from the model directory, and a
    // missing file names the path it tried.
    #[test]
    fn prompt_loads_the_model_directory_template() {
        let dir = std::env::temp_dir().join(format!("fg-chat-template-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        std::fs::write(dir.join("chat_template.jinja"), TEMPLATE).expect("write template");
        let out = render_chat_prompt(&dir, &messages(), &tools(), true).expect("render");
        assert!(out.starts_with("<user>list the drive</user>"), "{out}");
        std::fs::remove_dir_all(&dir).expect("cleanup");
        let err = render_chat_prompt(&dir, &messages(), &tools(), true).expect_err("missing file");
        assert!(format!("{err:#}").contains("chat_template.jinja"), "{err:#}");
    }
}
