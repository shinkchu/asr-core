//! Any configured backend, with credentials supplied separately through ASR_API_KEY.
mod support;
use asr_core::*;
use std::{
    error::Error,
    time::{Duration, Instant},
};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: transcribe CONFIG.json AUDIO.wav".into());
    }
    let audio = audio::read_wav_pcm16(&args[2])?;
    let engine = Engine::prepare(support::config(&args[1])?, EngineOptions::default())?;
    let mut options = SessionOptions::new(audio.spec);
    options.max_duration = Duration::from_secs(60 * 60);
    options.max_transcript_bytes = 2 * 1024 * 1024;
    let outcome = engine.transcribe(&audio, options, Instant::now() + Duration::from_secs(60))?;
    println!("{}", serde_json::to_string_pretty(&outcome.transcript)?);
    println!("{}", outcome.transcript.text());
    Ok(())
}
