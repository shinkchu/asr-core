#![doc = include_str!("../README.md")]
//! Bounded, runtime-independent speech recognition sessions.
//!
//! `Engine::prepare` creates reusable backend resources. `Engine::start` creates an isolated session;
//! audio enters through `AudioInput` and final results are obtained with `Session::finish`.
//! See the repository examples and docs/architecture.md for complete lifecycles.
pub mod audio;
mod backends;
mod config;
mod coordinator;
#[cfg(any(
    feature = "backend-dashscope",
    feature = "backend-openai-http",
    feature = "backend-openai-realtime"
))]
mod deadline;
mod engine;
mod error;
mod result;
mod session;
pub mod utils;
pub use audio::{AudioBuffer, AudioChunk, AudioSpec};
pub use config::{
    BiasPhrase, DashScopeConfig, EngineConfig, HttpMode, HttpResponse, OfflineConfig,
    OfflineFamily, OpenAiHttpConfig, OpenAiRealtimeConfig, PunctConfig, Secret, SessionOptions,
    SpeechHints, StreamingConfig, Timeouts, TransducerBiasConfig, VadConfig, DEFAULT_NUM_THREADS,
    MAX_NUM_THREADS,
};
pub use coordinator::{AudioInput, PushError, Session, Subscription};
pub use engine::{Engine, EngineOptions};
pub use error::{AsrError, ErrorKind};
pub use result::{
    BackendCapabilities, PartialView, Segment, SessionFailure, SessionOutcome, SessionPhase,
    SessionResult, SessionView, Transcript, Update,
};

#[cfg(feature = "capture-cpal")]
mod capture;
#[cfg(feature = "capture-cpal")]
pub use capture::CaptureSession;
