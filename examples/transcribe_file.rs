use asr_core::{
    audio::read_wav_pcm16, BiasPhrase, Engine, EngineConfig, EngineOptions, PunctConfig,
    SessionOptions, StreamingConfig, TransducerBiasConfig,
};
use std::{
    error::Error,
    time::{Duration, Instant},
};
fn main() -> Result<(), Box<dyn Error>> {
    let mut positional: Vec<String> = Vec::new();
    let mut hotwords: Vec<BiasPhrase> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--hotwords" {
            let list = args.next().ok_or("--hotwords requires a value")?;
            for word in list.split(',').map(str::trim).filter(|w| !w.is_empty()) {
                hotwords.push(BiasPhrase::new(word));
            }
        } else {
            positional.push(arg);
        }
    }
    if positional.len() != 2 && positional.len() != 3 {
        return Err(
            "usage: transcribe_file MODEL_DIRECTORY AUDIO.wav [PUNCTUATION_MODEL_DIRECTORY] [--hotwords WORD1,WORD2,...]"
                .into(),
        );
    }
    let audio = read_wav_pcm16(&positional[1])?;
    let engine = Engine::prepare(
        EngineConfig::Streaming(StreamingConfig {
            model_dir: positional[0].clone().into(),
            punctuation: positional.get(2).map(|dir| PunctConfig::new(dir.clone())),
            // 引擎级热词：启用 modified_beam_search 并对所有会话生效。
            bias: (!hotwords.is_empty()).then(|| TransducerBiasConfig::new(hotwords)),
        }),
        EngineOptions::default(),
    )?;
    let mut options = SessionOptions::new(audio.spec);
    options.max_duration = Duration::from_secs(60 * 60);
    options.max_transcript_bytes = 2 * 1024 * 1024;
    let result = engine.transcribe(&audio, options, Instant::now() + Duration::from_secs(30))?;
    println!("{}", result.transcript.text());
    Ok(())
}
