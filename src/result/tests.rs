use super::*;

#[test]
fn transcript_join_keeps_repeated_utterances() {
    let transcript = Transcript {
        segments: ["你好。", "你好。", "Hello.", "Hello."]
            .iter()
            .enumerate()
            .map(|(index, text)| Segment {
                id: index.to_string(),
                index: index as u64,
                text: text.to_string(),
                start_seconds: None,
                end_seconds: None,
            })
            .collect(),
    };
    assert_eq!(transcript.text(), "你好。你好。Hello. Hello.");
}

#[test]
fn transcript_join_mixed_cjk_and_ascii_matches_previous_semantics() {
    let transcript = Transcript {
        segments: [
            "你好",    // CJK end: never a space, even before ASCII
            "world",   // "你好world"
            "!",       // punctuation start: no space -> "你好world!"
            " 测试 ",  // trimmed; CJK start: no space -> "你好world!测试"
            "done.",   // CJK end: no space -> "你好world!测试done."
            " ",       // whitespace-only segment is skipped entirely
            "OK",      // '.' + ASCII start: space -> "...done. OK"
            "，中文",  // CJK punctuation start: no space
            "next 42", // CJK end: no space
        ]
        .iter()
        .enumerate()
        .map(|(index, text)| Segment {
            id: index.to_string(),
            index: index as u64,
            text: text.to_string(),
            start_seconds: None,
            end_seconds: None,
        })
        .collect(),
    };
    assert_eq!(transcript.text(), "你好world!测试done. OK，中文next 42");
}

#[test]
fn boxed_failure_keeps_session_result_smaller() {
    type DirectResult = Result<SessionOutcome, SessionFailure>;
    assert!(std::mem::size_of::<SessionResult>() < std::mem::size_of::<DirectResult>());
    eprintln!(
        "SessionFailure={} boxed SessionResult={} direct SessionResult={}",
        std::mem::size_of::<SessionFailure>(),
        std::mem::size_of::<SessionResult>(),
        std::mem::size_of::<DirectResult>(),
    );
}

#[test]
fn session_failure_exposes_the_structured_error_as_its_source() {
    let failure = SessionFailure {
        error: AsrError::new(crate::ErrorKind::Backend, "fixture", "failed"),
        outcome: SessionOutcome::default(),
    };
    let source = std::error::Error::source(&failure).unwrap();
    assert!(source.downcast_ref::<AsrError>().is_some());
}
