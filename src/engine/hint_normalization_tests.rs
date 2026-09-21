use super::normalize_session_hints;
use crate::{SessionOptions, SpeechHints};

#[test]
fn empty_phrase_hints_are_normalized_to_none() {
    let normalized = normalize_session_hints(SessionOptions {
        hints: Some(SpeechHints::default()),
        ..Default::default()
    });
    assert!(normalized.hints.is_none());

    // 反序列化/配置驱动的调用方产生的空短语 hints 走同一条归一化路径。
    let deserialized: SpeechHints = serde_json::from_str(r#"{"phrases": []}"#).unwrap();
    let normalized = normalize_session_hints(SessionOptions {
        hints: Some(deserialized),
        ..Default::default()
    });
    assert!(normalized.hints.is_none());

    // None 保持 None。
    let normalized = normalize_session_hints(SessionOptions::default());
    assert!(normalized.hints.is_none());
}

#[test]
fn non_empty_hints_are_preserved_by_the_normalization() {
    let hints = SpeechHints::new(vec!["语音识别".into()]);
    let normalized = normalize_session_hints(SessionOptions {
        hints: Some(hints.clone()),
        ..Default::default()
    });
    assert_eq!(normalized.hints, Some(hints));
}
