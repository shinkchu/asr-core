use super::*;
use crate::{config::validate_timeouts, Secret, SpeechHints, Timeouts};
use std::time::Duration;

fn prepare(config: EngineConfig) -> Result<Engine, AsrError> {
    Engine::prepare(config, EngineOptions::default())
}

fn http_config() -> EngineConfig {
    EngineConfig::OpenAiHttp(OpenAiHttpConfig::new(
        "https://example.com/v1",
        "model",
        Secret::new("key"),
    ))
}

#[test]
fn engine_options_reject_zero_and_clones_share_the_session_limit() {
    assert_eq!(
        EngineOptions::new(0).unwrap_err().kind,
        ErrorKind::InvalidInput
    );
    let engine = Engine::prepare(http_config(), EngineOptions::new(1).unwrap()).unwrap();
    let clone = engine.clone();
    let first = engine.start(Default::default()).unwrap();
    assert_eq!(engine.active_sessions(), 1);
    assert_eq!(
        clone.start(Default::default()).err().unwrap().kind,
        ErrorKind::Busy
    );
    first.cancel();
    let deadline = Instant::now() + Duration::from_secs(2);
    while engine.active_sessions() != 0 && Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert_eq!(engine.active_sessions(), 0);
    let second = clone.start(Default::default()).unwrap();
    second.cancel();
}

#[test]
fn prepare_calls_have_independent_session_limits() {
    let first = Engine::prepare(http_config(), EngineOptions::new(1).unwrap()).unwrap();
    let second = Engine::prepare(http_config(), EngineOptions::new(1).unwrap()).unwrap();
    let first_session = first.start(Default::default()).unwrap();
    let second_session = second.start(Default::default()).unwrap();
    assert_eq!(first.active_sessions(), 1);
    assert_eq!(second.active_sessions(), 1);
    first_session.cancel();
    second_session.cancel();
}

#[test]
fn transcribe_startup_validation_returns_an_empty_failure() {
    let engine = prepare(http_config()).unwrap();
    let audio = AudioBuffer::mono(vec![0.0; 160], 16000).unwrap();

    let expired = engine
        .transcribe(
            &audio,
            SessionOptions::new(audio.spec),
            Instant::now() - Duration::from_millis(1),
        )
        .unwrap_err();
    assert_eq!(expired.error.kind, ErrorKind::Timeout);
    assert_eq!(expired.outcome.received_frames, 0);
    assert_eq!(engine.active_sessions(), 0);

    let mismatch = engine
        .transcribe(
            &audio,
            SessionOptions::new(AudioSpec::mono(8000).unwrap()),
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap_err();
    assert_eq!(mismatch.error.kind, ErrorKind::InvalidInput);
    assert_eq!(mismatch.outcome.received_frames, 0);

    let too_long = engine
        .transcribe(
            &audio,
            SessionOptions {
                max_duration: Duration::from_millis(5),
                ..SessionOptions::new(audio.spec)
            },
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap_err();
    assert_eq!(too_long.error.kind, ErrorKind::ResourceLimit);
    assert_eq!(too_long.outcome.received_frames, 0);
}

#[test]
fn transcribe_rejects_non_finite_samples_before_start() {
    let engine = prepare(http_config()).unwrap();
    let mut samples = vec![0.0; 160];
    samples[0] = f32::NAN;
    let audio = AudioBuffer::mono(samples, 16000).unwrap();

    let failure = engine
        .transcribe(
            &audio,
            SessionOptions::new(audio.spec),
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap_err();

    assert_eq!(failure.error.kind, ErrorKind::InvalidInput);
    assert_eq!(failure.error.stage, "transcribe");
    assert_eq!(failure.outcome.received_frames, 0);
    assert_eq!(engine.active_sessions(), 0);
}

#[test]
fn empty_transcribe_uses_the_normal_backend_contract() {
    let engine = prepare(http_config()).unwrap();
    let audio = AudioBuffer::mono(Vec::new(), 16000).unwrap();
    let outcome = engine
        .transcribe(
            &audio,
            SessionOptions::new(audio.spec),
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
    assert_eq!(outcome.received_frames, 0);
    assert_eq!(outcome.processed_frames, 0);
    assert!(outcome.transcript.segments.is_empty());
}

#[test]
fn excessive_network_timeouts_are_rejected() {
    let timeouts = Timeouts {
        connect: Duration::MAX,
        send: Duration::from_secs(1),
        response: Duration::from_secs(1),
    };

    let error = validate_timeouts(&timeouts).expect_err("huge timeout must be rejected");

    assert_eq!(error.kind, ErrorKind::InvalidInput);
}

#[test]
fn session_hints_on_an_unsupported_backend_fail_fast() {
    let engine = prepare(EngineConfig::OpenAiHttp(OpenAiHttpConfig::new(
        "https://example.com/v1",
        "model",
        Secret::new("key"),
    )))
    .unwrap();
    assert!(!engine.capabilities().supports_session_hints);
    let error = engine
        .start(SessionOptions {
            hints: Some(SpeechHints::new(vec!["语音识别".into()])),
            ..Default::default()
        })
        .err()
        .expect("unsupported session hints must fail");
    assert_eq!(error.kind, ErrorKind::UnsupportedCapability);
    assert_eq!(error.stage, "start");
}

#[test]
fn empty_session_hints_start_like_no_hints_on_an_unsupported_backend() {
    let engine = prepare(http_config()).unwrap();
    assert!(!engine.capabilities().supports_session_hints);

    // 零短语 hints 等价于无 hints:不再报 UnsupportedCapability。
    let session = engine
        .start(SessionOptions {
            hints: Some(SpeechHints::default()),
            ..Default::default()
        })
        .expect("empty session hints must be treated as no hints");
    session.cancel();

    // transcribe 路径同样归一化,契约与 None 完全一致。
    let audio = AudioBuffer::mono(Vec::new(), 16000).unwrap();
    let outcome = engine
        .transcribe(
            &audio,
            SessionOptions {
                hints: Some(SpeechHints::default()),
                ..SessionOptions::new(audio.spec)
            },
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
    assert_eq!(outcome.received_frames, 0);
    assert_eq!(outcome.processed_frames, 0);
    assert!(outcome.transcript.segments.is_empty());
}
