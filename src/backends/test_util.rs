//! Shared fixtures for the WebSocket-backed cloud driver tests: a minimal
//! synchronous in-process server plus an engine constructor.

use crate::{AsrError, Engine, EngineConfig, EngineOptions};
use serde_json::Value;
use std::{
    net::{TcpListener, TcpStream},
    time::Duration,
};
use tokio_tungstenite::tungstenite::{self, accept_hdr, Message, WebSocket};

pub(crate) fn prepare(config: EngineConfig) -> Result<Engine, AsrError> {
    Engine::prepare(config, EngineOptions::default())
}

#[allow(clippy::result_large_err)]
pub(crate) fn server(
    run: impl FnOnce(WebSocket<TcpStream>) + Send + 'static,
) -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let worker = std::thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(20)))
            .unwrap();
        let socket = accept_hdr(
            socket,
            |request: &tungstenite::handshake::server::Request, response| {
                assert!(request.headers().contains_key("sec-websocket-key"));
                assert_eq!(request.headers()["authorization"], "Bearer fixture-key");
                Ok(response)
            },
        )
        .unwrap();
        run(socket);
    });
    (format!("ws://{address}/fixture"), worker)
}

pub(crate) fn read(socket: &mut WebSocket<TcpStream>) -> Value {
    serde_json::from_str(socket.read().unwrap().to_text().unwrap()).unwrap()
}

pub(crate) fn send(socket: &mut WebSocket<TcpStream>, value: Value) {
    socket
        .send(Message::Text(value.to_string().into()))
        .unwrap();
}
