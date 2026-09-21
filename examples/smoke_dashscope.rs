//! DashScope long-silence probe (TOFIX P2-5). Pushes `SMOKE_SILENCE_SECONDS`
//! of silence paced in real time so a server-side silence timeout can trigger;
//! after the `heartbeat: true` fix the session must complete instead of dying
//! with "connection closed before completion".
//!
//! ```text
//! ASR_API_KEY=sk-... cargo run --features backend-dashscope --example smoke_dashscope
//! ```
//!
//! Environment: `ASR_API_KEY` (required), `DASHSCOPE_ENDPOINT`
//! (default: the mainland endpoint; set the intl one for Singapore
//! workspaces), `DASHSCOPE_MODEL`, `SMOKE_SILENCE_SECONDS` (default 90).

mod support;

use asr_core::{
    AudioChunk, AudioSpec, DashScopeConfig, Engine, EngineConfig, EngineOptions, ErrorKind,
    SessionOptions,
};
use std::{
    error::Error,
    time::{Duration, Instant},
};

fn main() -> Result<(), Box<dyn Error>> {
    let api_key = support::api_key()?;
    let endpoint = std::env::var("DASHSCOPE_ENDPOINT")
        .unwrap_or_else(|_| "wss://dashscope.aliyuncs.com/api-ws/v1/inference".into());
    let model =
        std::env::var("DASHSCOPE_MODEL").unwrap_or_else(|_| "paraformer-realtime-v2".into());
    let seconds: u64 = std::env::var("SMOKE_SILENCE_SECONDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(90);

    let config = EngineConfig::DashScope(DashScopeConfig::new(endpoint, model, api_key));
    let engine = Engine::prepare(config, EngineOptions::default())?;
    let session = engine.start(SessionOptions::new(AudioSpec {
        sample_rate: 16_000,
    }))?;
    let input = session.input();

    // 100ms chunks pushed at real-time pace: the probe must reflect the
    // production condition, not how fast the local queue can drain. An early
    // push failure means the server already failed the session; the terminal
    // error carries the rendered task-failed detail, so fall through to it.
    let chunk = vec![0.0f32; 1_600];
    println!("pushing {seconds}s of silence ...");
    let mut push_error = None;
    for _ in 0..seconds * 10 {
        if let Err(error) = input.push_wait(
            AudioChunk::mono(chunk.clone(), 16_000)?,
            Instant::now() + Duration::from_secs(5),
        ) {
            push_error = Some(error);
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    match session.finish(Instant::now() + Duration::from_secs(30)) {
        Ok(outcome) => {
            support::report_pass(&outcome, &format!("completed over {seconds}s of silence"));
            Ok(())
        }
        Err(failure) => {
            if let Some(push_error) = push_error {
                eprintln!("push stopped early: {}", push_error.error.message);
            }
            support::fail_line(&failure);
            // The heartbeat hint only applies to the P2-2 failure signature
            // (silence killing the connection); a task-failed session is a
            // server-side rejection, not a keep-alive problem.
            if failure.error.kind == ErrorKind::Protocol
                && failure.error.message.contains("connection closed")
            {
                eprintln!("heartbeat keep-alive did not hold the connection (TOFIX P2-2)");
            }
            std::process::exit(1);
        }
    }
}
