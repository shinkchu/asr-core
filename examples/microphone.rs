mod support;
use asr_core::*;
use std::{
    collections::BTreeMap,
    error::Error,
    time::{Duration, Instant},
};
fn main() -> Result<(), Box<dyn Error>> {
    let config = std::env::args()
        .nth(1)
        .ok_or("usage: microphone CONFIG.json")?;
    let engine = Engine::prepare(support::config(&config)?, EngineOptions::default())?;
    let mut capture = CaptureSession::start(&engine, Default::default())?;
    let mut subscription = capture.subscribe().unwrap();
    let display = std::thread::spawn(move || {
        let mut first = true;
        let mut partials = BTreeMap::new();
        loop {
            match subscription.recv_timeout(Duration::from_secs(1)) {
                Ok(Update::Reset(view)) => {
                    let terminal = matches!(
                        view.phase,
                        SessionPhase::Completed | SessionPhase::Failed | SessionPhase::Cancelled
                    );
                    eprintln!(
                        "{} Reset: phase={:?}, text={}",
                        if first {
                            "initial"
                        } else if terminal {
                            "final"
                        } else {
                            "overflow recovery"
                        },
                        view.phase,
                        view.transcript.text()
                    );
                    partials = view
                        .partials
                        .into_iter()
                        .map(|partial| (partial.utterance_id, partial.text))
                        .collect();
                    first = false;
                }
                Ok(Update::Partial {
                    utterance_id,
                    revision,
                    text,
                }) => {
                    eprintln!("partial {utterance_id}@{revision}: {text}");
                    partials.insert(utterance_id, text);
                }
                Ok(Update::Segment(segment)) => {
                    partials.remove(&segment.id);
                    eprintln!("final {}: {}", segment.id, segment.text);
                }
                Ok(Update::Phase(phase)) => eprintln!("phase: {phase:?}"),
                Ok(_) | Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    eprintln!("subscription disconnected after final Reset");
                    break;
                }
            }
        }
    });
    eprintln!("Recording. Press Enter to finish.");
    std::io::stdin().read_line(&mut String::new())?;
    let result = capture.finish(Instant::now() + Duration::from_secs(30));
    display.join().map_err(|_| "event display failed")?;
    println!("{}", result?.transcript.text());
    Ok(())
}
