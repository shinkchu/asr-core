use crate::AsrError;
use serde::{Deserialize, Serialize};
use std::fmt;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionPhase {
    Starting,
    Running,
    Finishing,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub id: String,
    pub index: u64,
    pub text: String,
    pub start_seconds: Option<f64>,
    pub end_seconds: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    pub segments: Vec<Segment>,
}

impl Transcript {
    pub fn text(&self) -> String {
        let mut out = String::new();
        for segment in &self.segments {
            let text = segment.text.trim();
            if text.is_empty() {
                continue;
            }
            // The separator decision only depends on whether the last character
            // is ASCII, and every ASCII character occupies exactly one byte in
            // UTF-8. A trailing non-ASCII byte can only belong to a multi-byte
            // character, which never satisfies either check, so inspecting just
            // the last byte is equivalent to `chars().last()` and stays O(1).
            if out
                .as_bytes()
                .last()
                .is_some_and(|b| b.is_ascii_alphanumeric() || b".!?;:,".contains(b))
                && text.starts_with(|c: char| c.is_ascii_alphanumeric())
            {
                out.push(' ');
            }
            out.push_str(text);
        }
        out
    }
}

/// Canonical partial transcript snapshot, shared by the session's internal
/// result storage/observation queue and the public observation APIs
/// ([`SessionView::partials`], [`Update::Partial`]'s payload shape), so the
/// two can never drift.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartialView {
    pub utterance_id: String,
    pub revision: u64,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionView {
    pub phase: SessionPhase,
    pub transcript: Transcript,
    pub partials: Vec<PartialView>,
    pub received_frames: u64,
    pub processed_frames: u64,
    pub queued_frames: usize,
    pub error: Option<AsrError>,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Update {
    Reset(SessionView),
    Partial {
        utterance_id: String,
        revision: u64,
        text: String,
    },
    Segment(Segment),
    Phase(SessionPhase),
}

#[derive(Debug, Clone, Default)]
pub struct SessionOutcome {
    pub transcript: Transcript,
    pub received_frames: u64,
    pub processed_frames: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub name: String,
    pub audio: crate::AudioSpec,
    pub realtime_audio: bool,
    pub partial_results: bool,
    pub remote: bool,
    /// Whether configured final text is automatically punctuated.
    pub punctuation: bool,
    /// Whether this prepared engine accepts per-session [`crate::SpeechHints`].
    pub supports_session_hints: bool,
}

#[derive(Debug, Clone)]
pub struct SessionFailure {
    pub error: AsrError,
    pub outcome: SessionOutcome,
}

impl fmt::Display for SessionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for SessionFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// The failure remains boxed to keep this frequently returned result compact.
pub type SessionResult = Result<SessionOutcome, Box<SessionFailure>>;

#[cfg(test)]
mod tests;
