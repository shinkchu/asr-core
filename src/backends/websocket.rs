use super::network::Network;
use crate::{session::driver::ResultSink, AsrError, ErrorKind, Secret, Timeouts};
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_tungstenite::tungstenite::{
    self,
    client::IntoClientRequest,
    protocol::{frame::coding::CloseCode, CloseFrame},
    Error as WebSocketError, Message,
};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

type Socket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

pub(crate) fn validate_endpoint(endpoint: &str) -> Result<(), AsrError> {
    super::network::parse_clean_url(
        endpoint,
        &["ws", "wss"],
        true,
        "invalid WebSocket endpoint",
        "WebSocket endpoint must use ws/wss without credentials or fragment",
    )?;
    Ok(())
}

pub(crate) struct WsConnection {
    network: Arc<Network>,
    socket: Option<Socket>,
    timeouts: Timeouts,
}

impl WsConnection {
    pub(crate) fn new(network: Arc<Network>, timeouts: Timeouts) -> Self {
        Self {
            network,
            socket: None,
            timeouts,
        }
    }

    pub(crate) fn response_timeout(&self) -> Duration {
        self.timeouts.response
    }

    pub(crate) fn connect(
        &mut self,
        endpoint: &str,
        api_key: &Secret,
        extra_headers: &[(&'static str, &'static str)],
        sink: &ResultSink,
    ) -> Result<(), AsrError> {
        let mut request = endpoint
            .into_client_request()
            .map_err(|_| protocol("invalid WebSocket request"))?;
        if let Some(bearer) = super::network::bearer_value(api_key) {
            request.headers_mut().insert(
                "Authorization",
                bearer
                    .parse()
                    .map_err(|_| AsrError::invalid("invalid authorization value"))?,
            );
        }
        for (name, value) in extra_headers {
            request.headers_mut().insert(
                *name,
                (*value)
                    .parse()
                    .map_err(|_| AsrError::invalid("invalid protocol header value"))?,
            );
        }
        let mut config = tungstenite::protocol::WebSocketConfig::default();
        config.max_message_size = Some(1024 * 1024);
        config.max_frame_size = Some(1024 * 1024);
        let socket = self
            .network
            .run(sink, "websocket connect", self.timeouts.connect, async {
                tokio_tungstenite::connect_async_tls_with_config(
                    request,
                    Some(config),
                    false,
                    Some(tokio_tungstenite::Connector::Rustls(
                        self.network.tls.clone(),
                    )),
                )
                .await
                .map(|(socket, _)| socket)
                .map_err(|error| websocket_error("websocket handshake", error))
            })?;
        self.socket = Some(socket);
        Ok(())
    }

    pub(crate) fn send(&mut self, message: Message, sink: &ResultSink) -> Result<(), AsrError> {
        let socket = self
            .socket
            .as_mut()
            .ok_or_else(|| protocol("websocket is not connected"))?;
        self.network
            .run(sink, "websocket send", self.timeouts.send, async {
                socket
                    .send(message)
                    .await
                    .map_err(|error| websocket_error("websocket send", error))
            })
    }

    pub(crate) fn send_json(&mut self, value: Value, sink: &ResultSink) -> Result<(), AsrError> {
        self.send(Message::Text(value.to_string().into()), sink)
    }

    pub(crate) fn receive_json(
        &mut self,
        sink: &ResultSink,
        timeout: Duration,
    ) -> Result<Option<Value>, AsrError> {
        let socket = self
            .socket
            .as_mut()
            .ok_or_else(|| protocol("websocket is not connected"))?;
        let message = self
            .network
            .run(sink, "websocket receive", timeout, async {
                socket
                    .next()
                    .await
                    .ok_or_else(|| protocol("connection closed before completion"))?
                    .map_err(|error| websocket_error("websocket receive", error))
            })?;
        match message {
            Message::Text(text) => serde_json::from_str(&text)
                .map(Some)
                .map_err(|_| protocol("invalid WebSocket JSON")),
            Message::Ping(payload) => {
                self.send(Message::Pong(payload), sink)?;
                Ok(None)
            }
            Message::Close(_) => Err(protocol("websocket closed before completion")),
            _ => Ok(None),
        }
    }

    /// Graceful shutdown for the success path: send a Close frame so the peer
    /// observes a proper close handshake instead of a bare TCP FIN, then drop
    /// the socket without waiting for the peer's close reply.
    ///
    /// Best-effort only: send failures, cancellation, or the send timeout are
    /// ignored so closing never turns a successful finish into an error; the
    /// socket is always dropped. Error and cancellation paths never reach this
    /// method and keep their previous drop-without-handshake behavior.
    pub(crate) fn close(&mut self, sink: &ResultSink) {
        let Some(socket) = self.socket.as_mut() else {
            return;
        };
        let _ = self
            .network
            .run(sink, "websocket close", self.timeouts.send, async {
                socket
                    .send(Message::Close(Some(CloseFrame {
                        code: CloseCode::Normal,
                        reason: "finished".into(),
                    })))
                    .await
                    .map_err(|error| websocket_error("websocket close", error))
            });
        self.socket.take();
    }
}

pub(crate) fn protocol(message: impl Into<String>) -> AsrError {
    AsrError::new(ErrorKind::Protocol, "websocket", message)
}

/// Upper bound for one server-provided error field so an abnormal server
/// cannot inflate the error message surfaced to callers. Shared by every
/// WebSocket driver's server-error rendering.
pub(crate) const SERVER_ERROR_FIELD_LIMIT: usize = 300;

/// Reads one server-provided error field: non-string and blank values
/// count as absent, oversized values are truncated.
pub(crate) fn server_error_field(value: &Value) -> Option<String> {
    let text = value.as_str()?.trim();
    if text.is_empty() {
        return None;
    }
    let mut field: String = text.chars().take(SERVER_ERROR_FIELD_LIMIT).collect();
    if field.len() < text.len() {
        field.push('…');
    }
    Some(field)
}

/// Poll cadence the WebSocket drivers report to the session worker.
pub(crate) const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Receive window for one poll slot's event pump. Deliberately its own
/// constant: it bounds a single WebSocket receive inside a poll, not the
/// worker cadence ([`POLL_INTERVAL`]), so tuning one must not silently
/// move the other.
pub(crate) const POLL_RECEIVE_TIMEOUT: Duration = Duration::from_millis(1);

/// Runs one driver wait loop (handshake confirm, close barrier, finish
/// predicate) under a single response-timeout window. `step(None)` reports
/// whether the wait is already over without touching the wire (the original
/// loops' check-first shape); `step(Some(remaining))` receives one event —
/// or a timeout — and reports the same. The deadline stage string and the
/// remainder arithmetic live only here so the two drivers cannot drift.
pub(crate) fn wait_for_event<F>(response_timeout: Duration, mut step: F) -> Result<(), AsrError>
where
    F: FnMut(Option<Duration>) -> Result<bool, AsrError>,
{
    let deadline = crate::deadline::after(response_timeout, "websocket response")?;
    while !step(None)? {
        step(Some(deadline.saturating_duration_since(Instant::now())))?;
    }
    Ok(())
}

/// One non-blocking event pump for the session worker's poll slot: a 1 ms
/// receive whose timeout degrades to a cancellation check, so an idle
/// connection still observes cancel and deadline state.
pub(crate) fn poll_event<R>(sink: &ResultSink, mut receive: R) -> Result<(), AsrError>
where
    R: FnMut(Duration) -> Result<(), AsrError>,
{
    match receive(POLL_RECEIVE_TIMEOUT) {
        Err(error) if error.kind == ErrorKind::Timeout => sink.check(),
        result => result,
    }
}

/// Composes a protocol error from a static prefix plus the server-provided
/// code and message fields: `"{prefix} ({code}: {message})"` degrading to
/// `"{prefix} ({message})"` and then to the bare prefix as fields stop
/// being renderable. Fields are capped by [`SERVER_ERROR_FIELD_LIMIT`].
pub(crate) fn server_error(prefix: &str, code: &Value, message: &Value) -> AsrError {
    let Some(message) = server_error_field(message) else {
        return protocol(prefix);
    };
    match server_error_field(code) {
        Some(code) => protocol(format!("{prefix} ({code}: {message})")),
        None => protocol(format!("{prefix} ({message})")),
    }
}

pub(crate) fn websocket_error(stage: &str, error: WebSocketError) -> AsrError {
    let (kind, message) = match error {
        WebSocketError::ConnectionClosed => (
            ErrorKind::Protocol,
            "WebSocket connection closed before completion",
        ),
        WebSocketError::AlreadyClosed => (ErrorKind::Backend, "WebSocket was already closed"),
        WebSocketError::Io(error) => {
            let message = match error.kind() {
                std::io::ErrorKind::ConnectionRefused => "WebSocket connection was refused",
                std::io::ErrorKind::ConnectionReset => "WebSocket connection was reset",
                std::io::ErrorKind::ConnectionAborted => "WebSocket connection was aborted",
                std::io::ErrorKind::NotConnected => "WebSocket connection was lost",
                std::io::ErrorKind::BrokenPipe => "WebSocket connection pipe was broken",
                std::io::ErrorKind::TimedOut => "WebSocket I/O timed out",
                std::io::ErrorKind::UnexpectedEof => "WebSocket connection ended unexpectedly",
                _ => "WebSocket I/O failed",
            };
            let kind = if error.kind() == std::io::ErrorKind::TimedOut {
                ErrorKind::Timeout
            } else {
                ErrorKind::Io
            };
            (kind, message)
        }
        WebSocketError::Tls(_) => (ErrorKind::Protocol, "WebSocket TLS handshake failed"),
        WebSocketError::Capacity(_) => (
            ErrorKind::ResourceLimit,
            "WebSocket message exceeded the configured limit",
        ),
        WebSocketError::Protocol(_) => (ErrorKind::Protocol, "WebSocket protocol error"),
        WebSocketError::WriteBufferFull(_) => {
            (ErrorKind::WouldBlock, "WebSocket write buffer is full")
        }
        WebSocketError::Utf8(_) => (ErrorKind::Protocol, "WebSocket text was not valid UTF-8"),
        WebSocketError::AttackAttempt => (
            ErrorKind::Protocol,
            "WebSocket peer sent a suspicious message",
        ),
        WebSocketError::Url(_) | WebSocketError::HttpFormat(_) => {
            (ErrorKind::InvalidInput, "invalid WebSocket request")
        }
        WebSocketError::Http(response) => {
            let mut error = AsrError::new(
                ErrorKind::Http,
                stage,
                format!(
                    "WebSocket handshake returned HTTP {}",
                    response.status().as_u16()
                ),
            );
            error.http_status = Some(response.status().as_u16());
            return error;
        }
    };
    AsrError::new(kind, stage, message)
}

#[cfg(test)]
mod tests;
