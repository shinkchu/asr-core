//! OpenAI HTTP transcription probe (TOFIX P2-5). Uploads a WAV through the
//! configured response mode so the live stream can be compared with the
//! parser's assumptions (SSE `transcript.text.delta`/`done` event names and
//! `event:` + `data:` framing).
//!
//! ```text
//! ASR_API_KEY=sk-... cargo run --features backend-openai-http --example smoke_http -- speech.wav
//! ASR_API_KEY=sk-... SMOKE_HTTP_MODE=utterances SMOKE_VAD_MODEL=silero_vad.onnx \
//!     cargo run --features backend-openai-http,vad-silero --example smoke_http -- speech.wav
//! ```
//!
//! Environment: `ASR_API_KEY` (required), `OPENAI_HTTP_API_ROOT`
//! (default https://api.openai.com/v1), `OPENAI_HTTP_MODEL`
//! (default gpt-4o-transcribe), `SMOKE_HTTP_MODE` (`whole` | `utterances`,
//! default `whole`; utterances additionally needs the `vad-silero` feature).

mod support;

use asr_core::{
    Engine, EngineConfig, EngineOptions, HttpMode, OpenAiHttpConfig, SessionOptions, VadConfig,
};
use std::{
    error::Error,
    time::{Duration, Instant},
};

fn main() -> Result<(), Box<dyn Error>> {
    let api_key = support::api_key()?;
    let Some(path) = std::env::args().nth(1) else {
        return Err("usage: smoke_http AUDIO.wav".into());
    };
    let api_root = std::env::var("OPENAI_HTTP_API_ROOT")
        .unwrap_or_else(|_| "https://api.openai.com/v1".into());
    let model = std::env::var("OPENAI_HTTP_MODEL").unwrap_or_else(|_| "gpt-4o-transcribe".into());

    let mut config = OpenAiHttpConfig::new(api_root, model, api_key);
    if std::env::var("SMOKE_HTTP_MODE").as_deref() == Ok("utterances") {
        let vad_model = std::env::var("SMOKE_VAD_MODEL")?;
        config.mode = HttpMode::Utterances(VadConfig::new(vad_model));
    }

    let engine = Engine::prepare(EngineConfig::OpenAiHttp(config), EngineOptions::default())?;
    let audio = asr_core::audio::read_wav_pcm16(&path)?;
    let mut options = SessionOptions::new(audio.spec);
    options.max_duration = Duration::from_secs(60 * 60);

    println!("uploading {path} ...");
    match engine.transcribe(&audio, options, Instant::now() + Duration::from_secs(120)) {
        Ok(outcome) => {
            support::report_pass(&outcome, "");
            Ok(())
        }
        Err(failure) => support::report_fail(&failure),
    }
}
