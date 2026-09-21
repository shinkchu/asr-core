// Every example compiles this module wholesale; only the helpers a given
// example calls are live in its binary, so unused entries are expected.
#![allow(dead_code)]

use asr_core::{EngineConfig, Secret, SessionFailure, SessionOutcome};
use std::error::Error;
/// Reads and validates `ASR_API_KEY`, the credential every smoke probe
/// needs.
pub fn api_key() -> Result<Secret, Box<dyn Error>> {
    let api_key = std::env::var("ASR_API_KEY").unwrap_or_default();
    if api_key.trim().is_empty() {
        return Err("ASR_API_KEY is required".into());
    }
    Ok(Secret::new(api_key))
}

/// Prints the shared PASS line; `context` (if non-empty) describes the
/// probe-specific condition before the frame count.
pub fn report_pass(outcome: &SessionOutcome, context: &str) {
    if context.is_empty() {
        println!(
            "PASS: received={} frames, text={:?}",
            outcome.received_frames,
            outcome.transcript.text()
        );
    } else {
        println!(
            "PASS: {context}, received={} frames, text={:?}",
            outcome.received_frames,
            outcome.transcript.text()
        );
    }
}

/// Prints the shared FAIL line with the structured error rendered the same
/// way in every probe.
pub fn fail_line(failure: &SessionFailure) {
    eprintln!(
        "FAIL: {:?}/{}: {}",
        failure.error.kind, failure.error.stage, failure.error.message
    );
}

/// Prints the FAIL line and exits nonzero.
pub fn report_fail(failure: &SessionFailure) -> ! {
    fail_line(failure);
    std::process::exit(1)
}

/// Loads an EngineConfig from a JSON file. Cloud variants get `ASR_API_KEY`
/// injected through [`api_key`] (same validation as the smoke probes);
/// local variants take no credential, so an unset `ASR_API_KEY` is fine
/// for them.
pub fn config(path: &str) -> Result<EngineConfig, Box<dyn Error>> {
    let mut config: EngineConfig = serde_json::from_slice(&std::fs::read(path)?)?;
    if matches!(
        config,
        EngineConfig::DashScope(_) | EngineConfig::OpenAiHttp(_) | EngineConfig::OpenAiRealtime(_)
    ) {
        let key = api_key()?;
        match &mut config {
            EngineConfig::DashScope(c) => c.api_key = key,
            EngineConfig::OpenAiHttp(c) => c.api_key = key,
            EngineConfig::OpenAiRealtime(c) => c.api_key = key,
            _ => unreachable!("cloud variant matched above"),
        }
    }
    Ok(config)
}
