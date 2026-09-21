//! Filesystem- and parameter-level precheck of an [`EngineConfig`].
//!
//! `Engine::prepare` remains the authority: it loads the native models and
//! rejects anything they reject. [`validate`] runs the same cheap checks
//! `prepare` runs before any model load — local backends delegate to the
//! shared `backends::precheck` layer (family capability rules, VAD
//! parameters, hotword bias against the model's vocabulary, filesystem and
//! punctuation layout), and cloud variants go through their parameter
//! validation. It never loads native models: a passing precheck does not
//! replace `Engine::prepare`, and a failing one names the first problem
//! found.
//!
//! This module is always compiled: cloud branches follow their own backend
//! features, and disabled backends (including the local branches in builds
//! without `backend-sherpa`) mirror `Engine::prepare`'s rejection.

use crate::{AsrError, EngineConfig, ErrorKind};

/// Cheap precheck of an [`EngineConfig`]: local model, VAD and punctuation
/// paths are probed on disk, hotword bias and VAD parameters go through the
/// validators `Engine::prepare` applies, and cloud variants are run through
/// their parameter validation.
pub fn validate(config: &EngineConfig) -> Result<(), AsrError> {
    // Cloud variants share the exact parameter dispatch Engine::prepare
    // runs (single home: backends::validate_cloud_config, a no-op for local
    // variants); local branches delegate to the shared backends::precheck
    // layer — the exact checks Engine::prepare runs before native
    // initialization — so a passing precheck cannot fail prepare on a
    // cheap check.
    crate::backends::validate_cloud_config(config)?;
    match config {
        #[cfg(feature = "backend-sherpa")]
        EngineConfig::Streaming(config) => crate::backends::precheck::streaming(config),
        #[cfg(all(feature = "backend-sherpa", feature = "vad-silero"))]
        EngineConfig::Offline(config) => crate::backends::precheck::offline(config),
        // Backend compiled out: mirror Engine::prepare's rejection instead
        // of silently passing a configuration no engine can prepare.
        #[allow(unreachable_patterns)]
        _ => Err(AsrError::new(
            ErrorKind::UnsupportedCapability,
            "configuration",
            "requested backend is disabled in this build",
        )),
    }
}

#[cfg(test)]
mod tests;
