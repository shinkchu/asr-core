#[cfg(any(feature = "backend-dashscope", feature = "backend-openai-realtime"))]
use crate::Secret;
use crate::{session::driver::ResultSink, AsrError};

/// Parses a caller-supplied endpoint URL and enforces the cleanliness rules
/// shared by the HTTP and WebSocket transports: an allowed scheme, a host,
/// no embedded credentials and no fragment. Query strings are allowed only
/// when `allow_query` is set (WebSocket providers put routing parameters
/// there; the HTTP API root rejects them).
pub(crate) fn parse_clean_url(
    raw: &str,
    schemes: &[&str],
    allow_query: bool,
    invalid_message: &str,
    constraint_message: &str,
) -> Result<url::Url, AsrError> {
    let url = url::Url::parse(raw).map_err(|_| AsrError::invalid(invalid_message))?;
    if !schemes.contains(&url.scheme())
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || (!allow_query && url.query().is_some())
        || url.fragment().is_some()
    {
        return Err(AsrError::invalid(constraint_message));
    }
    Ok(url)
}

/// The Authorization header value for a usable API key; an empty (or
/// blank-only) key sends the request unauthenticated, matching the
/// `trim().is_empty()` semantics of config validation. Only the WebSocket
/// handshake builds the header by hand; the HTTP client uses its own
/// bearer_auth.
#[cfg(any(feature = "backend-dashscope", feature = "backend-openai-realtime"))]
pub(crate) fn bearer_value(api_key: &Secret) -> Option<String> {
    let key = api_key.0.trim();
    (!key.is_empty()).then(|| format!("Bearer {key}"))
}
use std::{future::Future, time::Duration};

pub(crate) struct Network {
    #[cfg(any(feature = "backend-dashscope", feature = "backend-openai-realtime"))]
    pub tls: std::sync::Arc<rustls::ClientConfig>,
    runtime: Option<tokio::runtime::Runtime>,
    #[cfg(feature = "backend-openai-http")]
    pub client: reqwest::Client,
}
impl Network {
    pub fn new() -> Result<Self, AsrError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("asr-core-network")
            .build()
            .map_err(|_| AsrError::backend("network runtime initialization failed"))?;
        #[cfg(any(feature = "backend-dashscope", feature = "backend-openai-realtime"))]
        let tls = rustls::ClientConfig::builder_with_provider(std::sync::Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .map_err(|_| AsrError::backend("TLS configuration failed"))?
        .with_root_certificates(rustls::RootCertStore::from_iter(
            webpki_roots::TLS_SERVER_ROOTS.iter().cloned(),
        ))
        .with_no_client_auth();
        Ok(Self {
            #[cfg(any(feature = "backend-dashscope", feature = "backend-openai-realtime"))]
            tls: std::sync::Arc::new(tls),
            runtime: Some(runtime),
            #[cfg(feature = "backend-openai-http")]
            client: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(|_| AsrError::backend("HTTP client initialization failed"))?,
        })
    }
    pub fn run<T>(
        &self,
        c: &ResultSink,
        stage: &str,
        timeout: Duration,
        future: impl Future<Output = Result<T, AsrError>>,
    ) -> Result<T, AsrError> {
        self.runtime.as_ref().unwrap().block_on(async {
            tokio::pin!(future);
            let deadline = crate::deadline::after(timeout, stage)?;
            loop {
                c.check()?;
                let end = c.deadline().map_or(deadline, |d| d.min(deadline));
                tokio::select! {
                    result = &mut future => return result,
                    _ = c.notified() => {},
                    _ = tokio::time::sleep_until(end.into()) => return Err(crate::deadline::exceeded(stage)),
                }
            }
        })
    }
}
impl Drop for Network {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}
