#[cfg(feature = "backend-dashscope")]
pub(crate) mod dashscope;
#[cfg(feature = "backend-sherpa")]
pub(crate) mod hotwords;
#[cfg(feature = "backend-openai-http")]
pub(crate) mod http;
#[cfg(feature = "backend-sherpa")]
pub(crate) mod local;
#[cfg(any(feature = "backend-sherpa", feature = "punct-sherpa"))]
pub(crate) mod model_layout;
#[cfg(any(
    feature = "backend-dashscope",
    feature = "backend-openai-http",
    feature = "backend-openai-realtime"
))]
pub(crate) mod network;

/// InvalidModel error at the shared "model" stage: layout and load
/// failures reported the same way by every local-backend module.
#[cfg(any(feature = "backend-sherpa", feature = "punct-sherpa"))]
pub(crate) fn model_error(message: impl Into<String>) -> crate::AsrError {
    crate::AsrError::new(crate::ErrorKind::InvalidModel, "model", message)
}

/// InvalidModel error at the "punctuation" stage shared by the punctuation
/// loader and its precheck.
#[cfg(feature = "punct-sherpa")]
pub(crate) fn punctuation_error(message: impl Into<String>) -> crate::AsrError {
    crate::AsrError::new(crate::ErrorKind::InvalidModel, "punctuation", message)
}
#[cfg(feature = "backend-sherpa")]
pub(crate) mod precheck;
#[cfg(feature = "punct-sherpa")]
pub(crate) mod punct;
#[cfg(feature = "backend-openai-realtime")]
pub(crate) mod realtime;
#[cfg(all(
    test,
    any(feature = "backend-dashscope", feature = "backend-openai-realtime")
))]
pub(crate) mod test_util;
#[cfg(all(
    feature = "vad-silero",
    any(feature = "backend-sherpa", feature = "backend-openai-http")
))]
pub(crate) mod vad;
#[cfg(any(feature = "backend-dashscope", feature = "backend-openai-realtime"))]
pub(crate) mod websocket;

/// Parameter-level validation of the cloud variants of an
/// [`crate::EngineConfig`]. This is the single dispatch home shared by
/// `Engine::prepare` and `crate::utils::precheck::validate`, so the two
/// entry points cannot drift apart; local variants are a no-op here and
/// are validated by [`precheck`] (and the loader) instead.
pub(crate) fn validate_cloud_config(config: &crate::EngineConfig) -> Result<(), crate::AsrError> {
    match config {
        #[cfg(feature = "backend-dashscope")]
        crate::EngineConfig::DashScope(cfg) => {
            websocket::validate_endpoint(&cfg.endpoint)?;
            cfg.validate_parameters()
        }
        #[cfg(feature = "backend-openai-http")]
        crate::EngineConfig::OpenAiHttp(cfg) => http::validate(cfg),
        #[cfg(feature = "backend-openai-realtime")]
        crate::EngineConfig::OpenAiRealtime(cfg) => {
            websocket::validate_endpoint(&cfg.endpoint)?;
            cfg.validate_parameters()
        }
        _ => Ok(()),
    }
}
