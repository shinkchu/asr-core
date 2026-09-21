//! Optional host conveniences around the engine core.
//!
//! The 0.4 core deliberately keeps recording, level metering, model asset
//! management and hot-swapping out of the `Engine → Session` path; hosts
//! implement those responsibilities themselves. This module provides
//! ready-made reference implementations for the common cases. Everything
//! here is additive and independent: a host may use any part of it, or the
//! equivalent hand-rolled logic, without affecting session behavior.
//!
//! - [`audio`]: RMS level metering, PCM16 WAV encoding, a raw-input
//!   [`audio::Recorder`] tee, and the [`audio::Resampler`].
//! - [`manager`]: background [`manager::EngineManager`] for hot-swapping a
//!   prepared [`crate::Engine`] with debounce and generation tracking.
//! - `models` (with `backend-sherpa` or `punct-sherpa`): local model
//!   directory family detection.
//! - [`precheck`]: cheap `crate::EngineConfig` prechecks sharing the exact
//!   validators `Engine::prepare` applies; always compiled, with per-backend
//!   branches following their features.
//! - `download` (with `model-download`): size- and checksum-verified
//!   model asset downloads.

pub mod audio;
#[cfg(feature = "model-download")]
pub mod download;
pub mod manager;
#[cfg(any(feature = "backend-sherpa", feature = "punct-sherpa"))]
pub mod models;
pub mod precheck;
