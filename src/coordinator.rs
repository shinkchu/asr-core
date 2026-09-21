#![cfg_attr(
    not(any(
        feature = "backend-sherpa",
        feature = "backend-dashscope",
        feature = "backend-openai-http",
        feature = "backend-openai-realtime"
    )),
    allow(dead_code)
)]
use crate::session::{
    driver::{Driver, FinalTextProcessor},
    observe,
    results::CommitPreflight,
    state::{failure_result, SessionState as State},
};
use crate::{
    AsrError, AudioChunk, AudioSpec, ErrorKind, PartialView, SessionOptions, SessionOutcome,
    SessionPhase, SessionResult, Update,
};
use std::{
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

const NANOS_PER_SECOND: u128 = 1_000_000_000;

// f32 PCM sample budget.
//
// queue:
// 16M frames × 4 bytes ≈ 64 MiB
//
const MAX_QUEUE_FRAMES: usize = 16 * 1024 * 1024;
const MAX_QUEUE_CHUNKS: usize = 4096;

fn duration_frames_raw(duration: Duration, rate: u32) -> u128 {
    let rate = u128::from(rate);

    u128::from(duration.as_secs()) * rate
        + u128::from(duration.subsec_nanos()) * rate / NANOS_PER_SECOND
}

pub(crate) fn duration_frames_usize(
    duration: Duration,
    rate: u32,
    field: &str,
) -> Result<usize, AsrError> {
    let frames = duration_frames_raw(duration, rate);

    if frames > usize::MAX as u128 {
        return Err(AsrError::invalid(format!(
            "{field} is too large for this platform"
        )));
    }

    Ok(frames as usize)
}

fn duration_frames_u64(duration: Duration, rate: u32, field: &str) -> Result<u64, AsrError> {
    let frames = duration_frames_raw(duration, rate);

    if frames > u64::MAX as u128 {
        return Err(AsrError::invalid(format!("{field} is too large")));
    }

    Ok(frames as u64)
}

/// Session geometry derived from [`SessionOptions`], in frames at the input
/// sample rate: the input-queue capacity, the largest chunk a single push
/// accepts, and the total session length cap.
///
/// `spawn` validates the full set; `Engine::transcribe`'s pre-check and the
/// capture worker derive the same numbers through this single home so the
/// duration-to-frames arithmetic cannot drift between entry points.
pub(crate) struct SessionGeometry {
    /// Input queue capacity, in frames at the input sample rate.
    pub(crate) capacity: usize,
    /// Largest number of frames a single push may accept, in the same unit.
    pub(crate) max_chunk_frames: usize,
    /// Total session length cap, in frames (may exceed `usize` on 32-bit
    /// targets, hence `u64`).
    pub(crate) max_session_frames: u64,
}

pub(crate) fn session_geometry(options: &SessionOptions) -> Result<SessionGeometry, AsrError> {
    let rate = options.input.sample_rate;
    let capacity = duration_frames_usize(options.queue_duration, rate, "queue_duration")?;
    let max_chunk_frames =
        duration_frames_usize(options.max_chunk_duration, rate, "max_chunk_duration")?;
    let max_session_frames = duration_frames_u64(options.max_duration, rate, "max_duration")?;

    if capacity == 0 {
        return Err(AsrError::invalid(
            "queue_duration is shorter than one input frame",
        ));
    }

    if max_chunk_frames == 0 {
        return Err(AsrError::invalid(
            "max_chunk_duration is shorter than one input frame",
        ));
    }

    if max_session_frames == 0 {
        return Err(AsrError::invalid(
            "max_duration is shorter than one input frame",
        ));
    }

    Ok(SessionGeometry {
        capacity,
        max_chunk_frames,
        max_session_frames,
    })
}

/// Shared session state guarded by `state`.
///
/// Invariant: every access to `state` in this crate goes through
/// `lock().unwrap()`, which is only sound because the critical sections are
/// panic-free. If a panic ever happens while the lock is held, the mutex is
/// poisoned and every later `lock().unwrap()` — on all `finish`, `push` and
/// `recv` paths — panics in cascade. When adding code to a critical section,
/// do not introduce panicking operations (indexing, `unwrap`, `expect`,
/// arithmetic that can overflow under debug assertions, ...): return an error
/// or move the work outside the lock instead. Inference must also stay out of
/// the lock, as pinned by
/// `punctuation_inference_does_not_hold_the_session_lock`.
pub(crate) struct Shared {
    pub(crate) state: Mutex<State>,
    pub(crate) changed: Condvar,
    pub(crate) options: SessionOptions,

    capacity: usize,
    max_chunk_frames: usize,
    max_session_frames: u64,

    #[cfg(any(
        feature = "backend-dashscope",
        feature = "backend-openai-http",
        feature = "backend-openai-realtime"
    ))]
    pub(crate) wake: tokio::sync::Notify,
}
#[derive(Clone)]
pub(crate) struct Control(pub(crate) Arc<Shared>);
impl Control {
    pub(crate) fn signal(&self) {
        self.0.changed.notify_all();
        #[cfg(any(
            feature = "backend-dashscope",
            feature = "backend-openai-http",
            feature = "backend-openai-realtime"
        ))]
        self.0.wake.notify_one();
    }
    pub(crate) fn check(&self) -> Result<(), AsrError> {
        let s = self.0.state.lock().unwrap();
        if let Some(result) = &s.lifecycle.outcome {
            return Err(match result {
                Err(e) => e.error.clone(),
                Ok(_) => AsrError::new(
                    ErrorKind::InputClosed,
                    "session",
                    "session already completed",
                ),
            });
        }
        if s.lifecycle.deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(AsrError::new(
                ErrorKind::Timeout,
                "session",
                "session deadline exceeded",
            ));
        }
        Ok(())
    }
    #[cfg(any(
        feature = "backend-dashscope",
        feature = "backend-openai-http",
        feature = "backend-openai-realtime"
    ))]
    pub(crate) fn deadline(&self) -> Option<Instant> {
        self.0.state.lock().unwrap().lifecycle.deadline
    }
    fn outcome_locked(&self, s: &mut State) -> SessionOutcome {
        SessionOutcome {
            transcript: s.results.transcript().clone(),
            received_frames: s.input.received,
            processed_frames: s.input.processed,
        }
    }
    fn settle_once_locked(&self, s: &mut State, error: Option<AsrError>) {
        if s.lifecycle.outcome.is_some() {
            return;
        }
        s.lifecycle.phase = match &error {
            None => SessionPhase::Completed,
            Some(e) if e.kind == ErrorKind::Cancelled => SessionPhase::Cancelled,
            _ => SessionPhase::Failed,
        };
        // A terminal failure can leave indexed gaps. ResultStore retains those
        // segments with their original indices while discarding unindexed text.
        s.results.settle();
        let outcome = self.outcome_locked(s);
        s.lifecycle.outcome = Some(match error {
            Some(error) => failure_result(error, outcome),
            None => Ok(outcome),
        });
        s.lifecycle.accepting = false;
        s.input.queue.clear();
        s.input.queued = 0;
        observe::terminal(s);
    }
    pub(crate) fn fail(&self, error: AsrError) {
        let mut s = self.0.state.lock().unwrap();
        self.settle_once_locked(&mut s, Some(error));
        drop(s);
        self.signal();
    }
    fn complete(&self) {
        let mut s = self.0.state.lock().unwrap();
        let error = s.results.completion_error();
        self.settle_once_locked(&mut s, error);
        drop(s);
        self.signal();
    }
    fn publish_partial_locked(&self, s: &mut State, update: PartialView) {
        observe::partial(s, &update, self.0.options.event_capacity);
    }
    #[cfg_attr(
        not(any(
            feature = "backend-sherpa",
            feature = "backend-dashscope",
            feature = "backend-openai-http"
        )),
        allow(dead_code)
    )]
    pub(crate) fn partial(
        &self,
        id: impl Into<String>,
        text: impl Into<String>,
    ) -> Result<(), AsrError> {
        self.check()?;
        let id = id.into();
        let text = text.into();
        let mut s = self.0.state.lock().unwrap();
        if s.lifecycle.outcome.is_some() {
            return Ok(());
        }
        let emit_update = s.observer.subscription_enabled;
        let Some(update) = s.results.set_partial(id, text, emit_update)? else {
            return Ok(());
        };
        self.publish_partial_locked(&mut s, update);
        drop(s);
        self.signal();
        Ok(())
    }
    #[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
    pub(crate) fn append_delta(&self, id: impl Into<String>, delta: &str) -> Result<(), AsrError> {
        self.check()?;
        let mut s = self.0.state.lock().unwrap();
        if s.lifecycle.outcome.is_some() {
            return Ok(());
        }
        let emit_update = s.observer.subscription_enabled;
        let Some(update) = s.results.append_delta(id.into(), delta, emit_update)? else {
            return Ok(());
        };
        self.publish_partial_locked(&mut s, update);
        drop(s);
        self.signal();
        Ok(())
    }
    #[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
    pub(crate) fn register_id(&self, id: &str) -> Result<(), AsrError> {
        self.check()?;
        let mut s = self.0.state.lock().unwrap();
        if s.lifecycle.outcome.is_some() {
            return Ok(());
        }
        s.results.register_id(id)
    }
    #[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
    pub(crate) fn preflight_unindexed(&self, id: &str) -> Result<bool, AsrError> {
        self.check()?;
        let s = self.0.state.lock().unwrap();
        if s.lifecycle.outcome.is_some() {
            return Ok(false);
        }
        s.results.preflight_unindexed(id)
    }
    #[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
    pub(crate) fn complete_unindexed(
        &self,
        id: impl Into<String>,
        text: impl Into<String>,
    ) -> Result<(), AsrError> {
        self.check()?;
        let id = id.into();
        let mut s = self.0.state.lock().unwrap();
        if s.lifecycle.outcome.is_some() {
            return Ok(());
        }
        let removed_partial = s.results.has_partial(&id);
        if s.results.complete_unindexed(id.clone(), text.into())? && removed_partial {
            observe::partial_removed_without_segment(&mut s, &id);
        }
        drop(s);
        self.signal();
        Ok(())
    }
    #[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
    pub(crate) fn index_unindexed(
        &self,
        index: u64,
        id: &str,
        start: Option<f64>,
        end: Option<f64>,
    ) -> Result<(), AsrError> {
        self.check()?;
        let mut s = self.0.state.lock().unwrap();
        if s.lifecycle.outcome.is_some() {
            return Ok(());
        }
        let update = s.results.index_unindexed(index, id, start, end)?;
        observe::final_segments(
            &mut s,
            id,
            update.removed_partial,
            &update.segments,
            self.0.options.event_capacity,
        );
        drop(s);
        self.signal();
        Ok(())
    }
    pub(crate) fn commit(
        &self,
        index: u64,
        id: impl Into<String>,
        text: impl Into<String>,
        start: Option<f64>,
        end: Option<f64>,
    ) -> Result<(), AsrError> {
        self.check()?;
        let id = id.into();
        let text = text.into();
        let mut s = self.0.state.lock().unwrap();
        if s.lifecycle.outcome.is_some() {
            return Ok(());
        }
        let update = s.results.commit(index, id.clone(), text, start, end)?;
        observe::final_segments(
            &mut s,
            &id,
            update.removed_partial,
            &update.segments,
            self.0.options.event_capacity,
        );
        drop(s);
        self.signal();
        Ok(())
    }

    pub(crate) fn preflight_commit(
        &self,
        index: u64,
        id: &str,
    ) -> Result<CommitPreflight, AsrError> {
        self.check()?;
        let s = self.0.state.lock().unwrap();
        if s.lifecycle.outcome.is_some() {
            return Ok(CommitPreflight::Duplicate);
        }
        s.results.preflight_commit(index, id)
    }
}

pub struct Session {
    control: Control,
    // Keep one input owner for the lifetime of the session. Callers can safely
    // use a temporary returned by `input()` without cancelling the session.
    input: AudioInput,
}
impl Session {
    pub fn input(&self) -> AudioInput {
        self.input.clone()
    }
    /// Returns the session's single self-healing live view subscription.
    ///
    /// The first read on the returned subscription is always
    /// [`Update::Reset`] carrying the current session state; afterwards
    /// updates are buffered up to `event_capacity` and a slow reader
    /// self-heals with a fresh [`Update::Reset`] instead of losing the
    /// transcript.
    ///
    /// # Guarantees
    ///
    /// - At most one `subscribe()` call succeeds per session. The slot is
    ///   claimed atomically under the session lock and is never released.
    /// - Every later call returns [`None`]; `subscribe()` does not return an
    ///   error and does not panic on repeated calls.
    /// - Dropping the [`Subscription`] disables the update queue (updates are
    ///   no longer buffered; the session itself keeps running and is
    ///   unaffected), but it does **not** reset the subscription slot, so
    ///   `subscribe()` cannot succeed again afterwards. A host that needs to
    ///   keep receiving updates must keep its `Subscription` alive for the
    ///   rest of the session.
    pub fn subscribe(&self) -> Option<Subscription> {
        let mut state = self.control.0.state.lock().unwrap();
        if state.observer.subscription_taken {
            return None;
        }
        state.observer.subscription_taken = true;
        observe::enable(&mut state);
        drop(state);
        self.control.signal();
        Some(Subscription {
            control: self.control.clone(),
        })
    }
    pub fn cancel(&self) {
        self.control.fail(AsrError::new(
            ErrorKind::Cancelled,
            "session",
            "session cancelled",
        ));
    }
    pub fn finish(&self, deadline: Instant) -> SessionResult {
        self.request_finish(deadline);
        let mut s = self.control.0.state.lock().unwrap();
        loop {
            if let Some(result) = &s.lifecycle.outcome {
                return result.clone();
            }
            // Unreachable as None: `outcome` is only ever set, never cleared,
            // so a None outcome here implies `request_finish` already stored
            // `Some(deadline)` under the same lock. Fall back to the caller's
            // finish deadline so a future settle-order refactor degrades to a
            // timeout instead of panicking in this public API.
            let deadline = s.lifecycle.deadline.unwrap_or(deadline);
            let now = Instant::now();
            if now >= deadline {
                self.control.settle_once_locked(
                    &mut s,
                    Some(AsrError::new(
                        ErrorKind::Timeout,
                        "finish",
                        "session finish deadline exceeded",
                    )),
                );
                self.control.signal();
                continue;
            }
            s = self
                .control
                .0
                .changed
                .wait_timeout(s, deadline - now)
                .unwrap()
                .0;
        }
    }
    fn request_finish(&self, deadline: Instant) {
        let mut s = self.control.0.state.lock().unwrap();
        if s.lifecycle.outcome.is_none() {
            s.lifecycle.accepting = false;
            s.lifecycle.finishing = true;
            s.lifecycle.deadline = Some(
                s.lifecycle
                    .deadline
                    .map_or(deadline, |current| current.min(deadline)),
            );
            s.lifecycle.phase = SessionPhase::Finishing;
            observe::phase(&mut s, self.control.0.options.event_capacity);
        }
        drop(s);
        self.control.signal();
    }
    pub(crate) fn fail_input(&self, error: AsrError) -> SessionResult {
        let mut state = self.control.0.state.lock().unwrap();
        self.control.settle_once_locked(&mut state, Some(error));
        let result = state.lifecycle.outcome.as_ref().unwrap().clone();
        drop(state);
        self.control.signal();
        result
    }
    #[cfg(feature = "capture-cpal")]
    pub(crate) fn control(&self) -> Control {
        self.control.clone()
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[derive(Debug)]
pub struct PushError {
    pub error: AsrError,
    pub chunk: AudioChunk,
}
impl std::fmt::Display for PushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}
impl std::error::Error for PushError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
#[derive(Clone)]
pub struct AudioInput {
    control: Control,
}
impl AudioInput {
    pub fn try_push(&self, chunk: AudioChunk) -> Result<(), PushError> {
        self.try_push_inner(chunk, None, false)
    }
    /// Internal feed for `Engine::transcribe`: the engine's whole-recording
    /// pre-scan (`Engine::validate_transcribe_input`) already rejected
    /// non-finite / out-of-range samples, so the redundant per-chunk sample
    /// re-scan is skipped. Every other check still applies: input spec,
    /// chunk duration, session state, receive deadline, the session audio
    /// limit, empty-chunk early exit and queue capacity.
    pub(crate) fn push_wait_validated(
        &self,
        chunk: AudioChunk,
        deadline: Instant,
    ) -> Result<(), PushError> {
        self.push_wait_inner(chunk, deadline, true)
    }
    fn try_push_inner(
        &self,
        chunk: AudioChunk,
        receive_deadline: Option<Instant>,
        samples_validated: bool,
    ) -> Result<(), PushError> {
        let options = &self.control.0.options;
        let error = if chunk.spec != options.input {
            Some(AsrError::new(
                ErrorKind::InvalidInput,
                "input",
                "input sample rate changed",
            ))
        } else if chunk.samples.len() > self.control.0.max_chunk_frames {
            Some(AsrError::new(
                ErrorKind::InvalidInput,
                "input",
                "audio chunk exceeds maximum duration",
            ))
        } else if !samples_validated
            && chunk
                .samples
                .iter()
                .any(|s| !s.is_finite() || s.abs() > 1.0)
        {
            Some(AsrError::new(
                ErrorKind::InvalidInput,
                "input",
                "audio must contain finite normalized samples in -1..=1",
            ))
        } else {
            None
        };
        if let Some(error) = error {
            return Err(PushError { error, chunk });
        }
        let mut s = self.control.0.state.lock().unwrap();
        let error = if !s.lifecycle.accepting {
            Some(AsrError::new(
                ErrorKind::InputClosed,
                "input",
                "session input is closed",
            ))
        } else if receive_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            Some(AsrError::new(
                ErrorKind::Timeout,
                "input",
                "waiting for audio capacity timed out",
            ))
        } else if s.input.received.saturating_add(chunk.samples.len() as u64)
            > self.control.0.max_session_frames
        {
            Some(AsrError::new(
                ErrorKind::ResourceLimit,
                "input",
                "maximum session audio duration exceeded",
            ))
        } else {
            None
        };
        if let Some(error) = error {
            return Err(PushError { error, chunk });
        }
        // Empty chunks are no-ops: they never occupy queue capacity, so they
        // succeed even when the queue is full instead of making `push_wait`
        // block on capacity it does not need.
        if chunk.samples.is_empty() {
            return Ok(());
        }
        if s.input.queue.len() >= MAX_QUEUE_CHUNKS
            || chunk.samples.len() > self.control.0.capacity - s.input.queued
        {
            return Err(PushError {
                error: AsrError::new(ErrorKind::WouldBlock, "input", "audio queue is full"),
                chunk,
            });
        }
        s.input.received += chunk.samples.len() as u64;
        s.input.queued += chunk.samples.len();
        s.input.queue.push_back(chunk);
        drop(s);
        self.control.signal();
        Ok(())
    }
    pub fn push_wait(&self, chunk: AudioChunk, deadline: Instant) -> Result<(), PushError> {
        self.push_wait_inner(chunk, deadline, false)
    }
    fn push_wait_inner(
        &self,
        mut chunk: AudioChunk,
        deadline: Instant,
        samples_validated: bool,
    ) -> Result<(), PushError> {
        loop {
            match self.try_push_inner(chunk, Some(deadline), samples_validated) {
                Ok(()) => return Ok(()),
                Err(e) if e.error.kind == ErrorKind::WouldBlock => {
                    chunk = e.chunk;
                }
                Err(e) => return Err(e),
            }
            let s = self.control.0.state.lock().unwrap();
            if !s.lifecycle.accepting || Instant::now() >= deadline {
                drop(s);
                continue;
            }
            // Check the predicate under the same lock as the consumer to avoid lost wakes.
            if s.lifecycle.accepting
                && (s.input.queue.len() >= MAX_QUEUE_CHUNKS
                    || chunk.samples.len() > self.control.0.capacity - s.input.queued)
            {
                let _ = self
                    .control
                    .0
                    .changed
                    .wait_timeout(s, deadline.saturating_duration_since(Instant::now()))
                    .unwrap();
            }
        }
    }
    #[cfg(feature = "capture-cpal")]
    pub(crate) fn fail(&self, error: AsrError) {
        self.control.fail(error);
    }
}
pub struct Subscription {
    control: Control,
}
impl Subscription {
    pub fn recv(&mut self) -> Option<Update> {
        let mut state = self.control.0.state.lock().unwrap();
        loop {
            match observe::take_next(&mut state) {
                observe::Delivery::Update(update) => return Some(update),
                observe::Delivery::Disconnected => return None,
                observe::Delivery::Pending => {
                    state = self.control.0.changed.wait(state).unwrap();
                }
            }
        }
    }

    pub fn recv_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Update, std::sync::mpsc::RecvTimeoutError> {
        // A duration too large for an `Instant` deadline degrades to waiting
        // without a deadline instead of overflowing.
        let deadline = Instant::now().checked_add(timeout);
        let mut state = self.control.0.state.lock().unwrap();
        loop {
            match observe::take_next(&mut state) {
                observe::Delivery::Update(update) => return Ok(update),
                observe::Delivery::Disconnected => {
                    return Err(std::sync::mpsc::RecvTimeoutError::Disconnected)
                }
                observe::Delivery::Pending => {}
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Err(std::sync::mpsc::RecvTimeoutError::Timeout);
            }
            state = match deadline {
                Some(deadline) => {
                    self.control
                        .0
                        .changed
                        .wait_timeout(state, deadline.saturating_duration_since(Instant::now()))
                        .unwrap()
                        .0
                }
                None => self.control.0.changed.wait(state).unwrap(),
            };
        }
    }
}
impl Drop for Subscription {
    fn drop(&mut self) {
        let mut state = self.control.0.state.lock().unwrap();
        observe::disable(&mut state);
    }
}
#[cfg(test)]
pub(crate) fn spawn(
    driver: Box<dyn Driver>,
    audio: AudioSpec,
    options: SessionOptions,
    final_text_processor: Option<FinalTextProcessor>,
    permit: impl Send + 'static,
) -> Result<Session, AsrError> {
    spawn_with_deadline(driver, audio, options, final_text_processor, permit, None)
}

pub(crate) fn spawn_with_deadline(
    driver: Box<dyn Driver>,
    audio: AudioSpec,
    options: SessionOptions,
    final_text_processor: Option<FinalTextProcessor>,
    permit: impl Send + 'static,
    deadline: Option<Instant>,
) -> Result<Session, AsrError> {
    AudioSpec::mono(options.input.sample_rate)?;
    AudioSpec::mono(audio.sample_rate)?;

    if options.event_capacity == 0 {
        return Err(AsrError::invalid(
            "event_capacity must be greater than zero",
        ));
    }

    if options.max_transcript_bytes == 0 {
        return Err(AsrError::invalid(
            "max_transcript_bytes must be greater than zero",
        ));
    }

    if options.queue_duration.is_zero() {
        return Err(AsrError::invalid(
            "queue_duration must be greater than zero",
        ));
    }

    if options.max_chunk_duration.is_zero() {
        return Err(AsrError::invalid(
            "max_chunk_duration must be greater than zero",
        ));
    }

    if options.max_duration.is_zero() {
        return Err(AsrError::invalid("max_duration must be greater than zero"));
    }

    if options.max_duration > Duration::from_secs(24 * 3600) {
        return Err(AsrError::invalid("max_duration must not exceed 24 hours"));
    }

    if options.max_chunk_duration > options.queue_duration {
        return Err(AsrError::invalid(
            "max_chunk_duration must not exceed queue_duration",
        ));
    }

    let geometry = session_geometry(&options)?;
    let SessionGeometry {
        capacity,
        max_chunk_frames,
        max_session_frames,
    } = geometry;

    if capacity > MAX_QUEUE_FRAMES {
        return Err(AsrError::invalid(format!(
            "queue_duration requires {capacity} frames, exceeding the \
             configured safety limit of {MAX_QUEUE_FRAMES} frames"
        )));
    }

    let mut state = State::new(options.max_transcript_bytes);
    state.lifecycle.deadline = deadline;
    let control = Control(Arc::new(Shared {
        state: Mutex::new(state),
        changed: Condvar::new(),
        options,
        capacity,
        max_chunk_frames,
        max_session_frames,
        #[cfg(any(
            feature = "backend-dashscope",
            feature = "backend-openai-http",
            feature = "backend-openai-realtime"
        ))]
        wake: tokio::sync::Notify::new(),
    }));
    let worker = control.clone();
    std::thread::Builder::new()
        .name("asr-core-session".into())
        .spawn(move || {
            let _permit = permit;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::session::worker::run(driver, audio, &worker, final_text_processor)
            }));
            match result {
                Ok(Ok(())) => worker.complete(),
                Ok(Err(e)) => worker.fail(e),
                Err(_) => worker.fail(AsrError::backend("backend worker panicked")),
            }
        })
        .map_err(|e| AsrError::new(ErrorKind::Io, "start", e.to_string()))?;
    let input = AudioInput {
        control: control.clone(),
    };
    Ok(Session { control, input })
}

#[cfg(test)]
mod tests;
