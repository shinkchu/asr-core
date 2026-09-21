use asr_core::{
    audio::read_wav_pcm16, AudioChunk, Engine, EngineConfig, EngineOptions, ErrorKind,
    SessionOptions, StreamingConfig,
};
use std::{
    error::Error,
    time::{Duration, Instant},
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: streaming MODEL_DIRECTORY AUDIO.wav".into());
    }
    let audio = read_wav_pcm16(&args[2])?;
    let engine = Engine::prepare(
        EngineConfig::Streaming(StreamingConfig::new(&args[1])),
        EngineOptions::default(),
    )?;
    let session = engine.start(SessionOptions::new(audio.spec))?;
    let input = session.input();
    let deadline = Instant::now() + Duration::from_secs(30);

    for samples in audio
        .samples
        .chunks((audio.spec.sample_rate as usize / 50).max(1))
    {
        let mut chunk = AudioChunk::mono(samples.to_vec(), audio.spec.sample_rate)?;
        loop {
            if Instant::now() >= deadline {
                return Err("streaming submission deadline exceeded".into());
            }
            match input.try_push(chunk) {
                Ok(()) => break,
                Err(error) if error.error.kind == ErrorKind::WouldBlock => {
                    chunk = error.chunk;
                    match input.push_wait(chunk, deadline) {
                        Ok(()) => break,
                        Err(error) if error.error.kind == ErrorKind::Timeout => {
                            if Instant::now() >= deadline {
                                return Err(error.into());
                            }
                            chunk = error.chunk;
                            continue;
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
                Err(error) => return Err(error.into()),
            }
        }
    }

    println!("{}", session.finish(deadline)?.transcript.text());
    Ok(())
}
