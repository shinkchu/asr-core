use super::results::ResultStore;
use crate::{
    AsrError, AudioChunk, SessionFailure, SessionOutcome, SessionPhase as Phase, SessionResult,
    Update,
};
use std::{collections::VecDeque, time::Instant};

pub(crate) struct Lifecycle {
    pub(crate) phase: Phase,
    pub(crate) accepting: bool,
    pub(crate) finishing: bool,
    pub(crate) deadline: Option<Instant>,
    pub(crate) outcome: Option<SessionResult>,
}

pub(crate) struct InputBuffer {
    pub(crate) queue: VecDeque<AudioChunk>,
    pub(crate) queued: usize,
    pub(crate) received: u64,
    pub(crate) processed: u64,
}

pub(crate) struct ObserverState {
    pub(crate) updates: VecDeque<Update>,
    pub(crate) subscription_taken: bool,
    pub(crate) subscription_enabled: bool,
    pub(crate) needs_reset: bool,
}

pub(crate) struct SessionState {
    pub(crate) lifecycle: Lifecycle,
    pub(crate) input: InputBuffer,
    pub(crate) results: ResultStore,
    pub(crate) observer: ObserverState,
}

impl SessionState {
    pub(crate) fn new(max_transcript_bytes: usize) -> Self {
        Self {
            lifecycle: Lifecycle {
                phase: Phase::Starting,
                accepting: true,
                finishing: false,
                deadline: None,
                outcome: None,
            },
            input: InputBuffer {
                queue: VecDeque::new(),
                queued: 0,
                received: 0,
                processed: 0,
            },
            results: ResultStore::new(max_transcript_bytes),
            observer: ObserverState {
                updates: VecDeque::new(),
                subscription_taken: false,
                subscription_enabled: false,
                needs_reset: false,
            },
        }
    }
}

/// Builds the single failure shape used by established sessions, as well as
/// failures that happen before a session worker can be started.
pub(crate) fn failure_result(error: AsrError, outcome: SessionOutcome) -> SessionResult {
    Err(Box::new(SessionFailure { error, outcome }))
}

#[cfg(test)]
mod tests;
