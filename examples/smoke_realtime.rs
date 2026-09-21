//! OpenAI Realtime transcription probe (TOFIX P2-5). Runs a GA
//! transcription session against the live endpoint. The silence probe
//! (no argument) exercises the finish path end to end; with
//! `SMOKE_SERVER_VAD=1` it also probes the empty-tail commit, whose real
//! server error code is rendered in the failure message — run this once
//! before trusting the `input_audio_buffer_commit_empty` literal.
//!
//! ```text
//! ASR_API_KEY=sk-... cargo run --features backend-openai-realtime --example smoke_realtime
//! ASR_API_KEY=sk-... cargo run --features backend-openai-realtime --example smoke_realtime -- speech.wav
//! ```
//!
//! Environment: `ASR_API_KEY` (required), `OPENAI_REALTIME_ENDPOINT`
//! (default: the GA transcription endpoint), `OPENAI_REALTIME_MODEL`
//! (default gpt-4o-transcribe), `SMOKE_SERVER_VAD` (default 1),
//! `SMOKE_SILENCE_SECONDS` (default 10).

mod support;

use asr_core::{
    AudioChunk, AudioSpec, Engine, EngineConfig, EngineOptions, OpenAiRealtimeConfig,
    SessionOptions,
};
use std::{
    error::Error,
    time::{Duration, Instant},
};

fn report(result: Result<asr_core::SessionOutcome, Box<asr_core::SessionFailure>>) {
    match result {
        Ok(outcome) => support::report_pass(&outcome, "completed"),
        Err(failure) => support::report_fail(&failure),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let api_key = support::api_key()?;
    let endpoint = std::env::var("OPENAI_REALTIME_ENDPOINT")
        .unwrap_or_else(|_| "wss://api.openai.com/v1/realtime?intent=transcription".into());
    let model =
        std::env::var("OPENAI_REALTIME_MODEL").unwrap_or_else(|_| "gpt-4o-transcribe".into());
    let server_vad = std::env::var("SMOKE_SERVER_VAD").as_deref() != Ok("0");
    let seconds: u64 = std::env::var("SMOKE_SILENCE_SECONDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10);

    let mut config = OpenAiRealtimeConfig::new(endpoint, model, api_key);
    config.server_vad = server_vad;
    let engine = Engine::prepare(
        EngineConfig::OpenAiRealtime(config),
        EngineOptions::default(),
    )?;

    if let Some(path) = std::env::args().nth(1) {
        let audio = asr_core::audio::read_wav_pcm16(&path)?;
        let mut options = SessionOptions::new(audio.spec);
        options.max_duration = Duration::from_secs(60 * 60);
        println!("transcribing {path} (server_vad={server_vad}) ...");
        report(engine.transcribe(&audio, options, Instant::now() + Duration::from_secs(120)));
        return Ok(());
    }

    let session = engine.start(SessionOptions::new(AudioSpec {
        sample_rate: 16_000,
    }))?;
    let input = session.input();
    let chunk = vec![0.0f32; 1_600];
    println!("pushing {seconds}s of silence (server_vad={server_vad}) ...");
    for _ in 0..seconds * 10 {
        input.push_wait(
            AudioChunk::mono(chunk.clone(), 16_000)?,
            Instant::now() + Duration::from_secs(5),
        )?;
        std::thread::sleep(Duration::from_millis(100));
    }
    report(session.finish(Instant::now() + Duration::from_secs(30)));
    Ok(())
}
