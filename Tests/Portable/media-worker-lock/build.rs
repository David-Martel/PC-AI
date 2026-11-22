use std::env;
use std::fs;
use std::path::PathBuf;

/// Return the text of the one function whose signature starts with `signature`,
/// through its closing `}` at column zero.
fn function<'a>(text: &'a str, signature: &str) -> &'a str {
    assert_eq!(text.matches(signature).count(), 1, "exactly one {signature}");
    let start = text.find(signature).expect("function signature");
    let rest = &text[start..];
    let end = rest.find("\n}\n").expect("function end at column zero") + 3;
    &rest[..end]
}

/// Text strictly between two markers inside `haystack`; each must occur once, in order.
fn between<'a>(haystack: &'a str, from: &str, to: &str) -> &'a str {
    assert_eq!(haystack.matches(from).count(), 1, "one {from}");
    assert_eq!(haystack.matches(to).count(), 1, "one {to}");
    assert!(haystack.find(from) < haystack.find(to), "{from} must precede {to}");
    let after = haystack.split_once(from).expect("from marker").1;
    after.split_once(to).expect("to marker").0
}

fn main() {
    println!("cargo:rerun-if-env-changed=PCAI_MEDIA_FFI_SOURCE");
    let source = PathBuf::from(env::var_os("PCAI_MEDIA_FFI_SOURCE").expect("set an explicit media FFI source path"));
    println!("cargo:rerun-if-changed={}", source.display());
    let text = fs::read_to_string(&source).expect("read selected media FFI source");

    // 1. State acquisition recovers from poison by consuming the held guard.
    let lock = function(&text, "fn lock_state_recovering() -> MutexGuard<'static, MediaState> {");
    assert!(lock.contains("get_state().lock()"), "acquisition takes the one global state lock");
    assert!(lock.contains("PoisonError::into_inner"), "poisoned-lock recovery must use into_inner");
    assert!(
        !lock.contains(".expect(") && !lock.contains(".unwrap()"),
        "acquisition must not panic on poison"
    );

    // 2. The Pending -> Running transition returns without holding the lock.
    let begin = function(&text, "fn begin_async_request(id: i64) -> bool {");
    assert!(
        begin.contains("lock_state_recovering()"),
        "transition acquires state through the recovering helper"
    );

    // 3. The worker never holds the state/request lock across generation or the
    //    publish step, so poll/cancel stay responsive for every request.
    let worker = function(&text, "fn execute_async_request_staged(");
    let before_generate = between(worker, "begin_async_request(id)", "let result = generate();");
    for forbidden in ["lock_state_recovering()", "get_state()", ".lock("] {
        assert!(
            !before_generate.contains(forbidden),
            "no state lock may be taken or held into generation: {forbidden}"
        );
    }
    let during_publish = between(worker, "begin_publish(id)", "finish_publish(id, terminal);");
    assert!(
        during_publish.contains("publish(&staged, dest)"),
        "the publish step sits between begin and finish"
    );
    for forbidden in ["lock_state_recovering()", "get_state()", ".lock("] {
        assert!(
            !during_publish.contains(forbidden),
            "no global lock may be held across the publish step: {forbidden}"
        );
    }

    // The two unit-testable functions are copied verbatim into the probe.
    let generated = format!("{lock}\n{begin}");
    let target = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo output directory"));
    fs::write(target.join("worker_acquisition.rs"), generated).expect("write source-bound probe");
}
