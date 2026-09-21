use super::*;
use crate::{ErrorKind, Segment, Transcript};

#[test]
fn failure_result_preserves_the_original_error_and_current_outcome() {
    let error = AsrError::new(ErrorKind::InvalidInput, "input", "fixture input error");
    let outcome = SessionOutcome {
        transcript: Transcript {
            segments: vec![Segment {
                id: "confirmed".into(),
                index: 0,
                text: "kept".into(),
                start_seconds: None,
                end_seconds: None,
            }],
        },
        received_frames: 320,
        processed_frames: 160,
    };

    let failure = failure_result(error.clone(), outcome.clone()).unwrap_err();
    assert_eq!(failure.error, error);
    assert_eq!(failure.outcome.transcript, outcome.transcript);
    assert_eq!(failure.outcome.received_frames, 320);
    assert_eq!(failure.outcome.processed_frames, 160);
}
