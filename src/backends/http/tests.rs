use super::*;
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::mpsc,
    time::{Duration, Instant},
};
fn prepare(config: EngineConfig) -> Result<Engine, AsrError> {
    Engine::prepare(config, EngineOptions::default())
}
fn server(status: u16, body: &'static str) -> (String, std::thread::JoinHandle<Vec<u8>>) {
    server_with_response(move |socket| {
        write!(socket,"HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nConnection: close\r\nx-request-id: fixture-123\r\n\r\n{body}",body.len()).unwrap();
    })
}
fn server_with_response(
    respond: impl FnOnce(&mut TcpStream) + Send + 'static,
) -> (String, std::thread::JoinHandle<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let worker = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut request = Vec::new();
        let mut buffer = [0; 4096];
        loop {
            let n = socket.read(&mut buffer).unwrap();
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..n]);
            if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&request[..end]).to_lowercase();
                let length: usize = headers
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .unwrap()
                    .trim()
                    .parse()
                    .unwrap();
                if request.len() >= end + 4 + length {
                    break;
                }
            }
        }
        respond(&mut socket);
        request
    });
    (format!("http://{address}/proxy/v1"), worker)
}
fn run(config: OpenAiHttpConfig) -> SessionResult {
    run_samples(config, 1600)
}
fn run_samples(config: OpenAiHttpConfig, frames: usize) -> SessionResult {
    let engine = prepare(EngineConfig::OpenAiHttp(config)).unwrap();
    let session = engine.start(Default::default()).unwrap();
    let input = session.input();
    for start in (0..frames).step_by(16000) {
        input
            .push_wait(
                AudioChunk::mono(vec![0.1; (frames - start).min(16000)], 16000).unwrap(),
                Instant::now() + Duration::from_secs(3),
            )
            .unwrap();
    }
    session.finish(Instant::now() + Duration::from_secs(3))
}

#[test]
fn transcribe_chunks_full_audio_and_honors_the_text_budget() {
    for max_transcript_bytes in [1024, 3] {
        let (root, worker) = server(200, r#"{"text":"hello"}"#);
        let engine = prepare(EngineConfig::OpenAiHttp(OpenAiHttpConfig::new(
            root,
            "fixture",
            Secret::new("key"),
        )))
        .unwrap();
        let audio = AudioBuffer::mono(vec![0.1; 1600], 16000).unwrap();
        let result = engine.transcribe(
            &audio,
            SessionOptions {
                max_chunk_duration: Duration::from_millis(10),
                max_transcript_bytes,
                ..SessionOptions::new(audio.spec)
            },
            Instant::now() + Duration::from_secs(3),
        );
        if max_transcript_bytes == 1024 {
            let outcome = result.unwrap();
            assert_eq!(outcome.received_frames, 1600);
            assert_eq!(outcome.processed_frames, 1600);
            assert_eq!(outcome.transcript.text(), "hello");
        } else {
            let failure = result.unwrap_err();
            assert_eq!(failure.error.kind, ErrorKind::ResourceLimit);
            assert_eq!(failure.outcome.received_frames, 1600);
        }
        worker.join().unwrap();
    }
}

#[test]
fn transcribe_rejects_a_known_oversize_whole_recording_before_start() {
    let mut config = OpenAiHttpConfig::new("https://example.com/v1", "fixture", Secret::new("key"));
    config.max_upload_bytes = audio::PCM16_WAV_HEADER_BYTES + 200;
    let engine = prepare(EngineConfig::OpenAiHttp(config)).unwrap();
    let audio = AudioBuffer::mono(vec![0.0; 101], 16000).unwrap();
    let failure = engine
        .transcribe(
            &audio,
            SessionOptions::new(audio.spec),
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap_err();
    assert_eq!(failure.error.kind, ErrorKind::ResourceLimit);
    assert_eq!(failure.error.stage, "upload");
    assert_eq!(failure.outcome.received_frames, 0);
    assert_eq!(engine.active_sessions(), 0);
}
#[test]
fn recognition_after_upload_uses_response_budget() {
    let (root, worker) = server_with_response(|socket| {
        // This delay starts only after the server has read the entire upload.
        std::thread::sleep(Duration::from_millis(500));
        write!(
            socket,
            "HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{{\"text\":\"okay\"}}"
        )
        .unwrap();
    });
    let mut config = OpenAiHttpConfig::new(root, "fixture", Secret::new("key"));
    config.timeouts = Timeouts {
        connect: Duration::from_millis(150),
        send: Duration::from_millis(150),
        response: Duration::from_secs(2),
    };
    let result = run(config);
    worker.join().unwrap();
    assert_eq!(result.unwrap().transcript.text(), "okay");
}
#[test]
fn response_deadline_still_bounds_waiting_for_headers() {
    let (release_tx, release_rx) = mpsc::channel();
    let (root, worker) = server_with_response(move |_| {
        release_rx.recv_timeout(Duration::from_secs(3)).unwrap();
    });
    let mut config = OpenAiHttpConfig::new(root, "fixture", Secret::new("key"));
    config.timeouts.response = Duration::from_millis(200);
    let failure = run(config).unwrap_err();
    release_tx.send(()).unwrap();
    worker.join().unwrap();
    assert_eq!(failure.error.kind, ErrorKind::Timeout);
    assert_eq!(failure.error.stage, "transcription");
}
#[test]
fn stalled_tls_handshake_obeys_connect_deadline() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let (_socket, _) = listener.accept().unwrap();
        release_rx.recv_timeout(Duration::from_secs(3)).unwrap();
    });
    let mut config = OpenAiHttpConfig::new(
        format!("https://{address}/v1"),
        "fixture",
        Secret::new("key"),
    );
    config.timeouts.connect = Duration::from_millis(200);
    let failure = run(config).unwrap_err();
    release_tx.send(()).unwrap();
    worker.join().unwrap();
    assert_eq!(failure.error.kind, ErrorKind::Timeout);
    assert_eq!(failure.error.stage, "HTTP connect");
}
#[test]
fn blocked_upload_obeys_send_deadline() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        // Read only headers, leaving an upload larger than the socket buffers blocked.
        let mut headers = Vec::new();
        while !headers.ends_with(b"\r\n\r\n") {
            let mut byte = [0];
            socket.read_exact(&mut byte).unwrap();
            headers.push(byte[0]);
        }
        release_rx.recv_timeout(Duration::from_secs(3)).unwrap();
    });
    let mut config = OpenAiHttpConfig::new(
        format!("http://{address}/v1"),
        "fixture",
        Secret::new("key"),
    );
    config.timeouts.send = Duration::from_millis(200);
    let failure = run_samples(config, 4 * 1024 * 1024).unwrap_err();
    release_tx.send(()).unwrap();
    worker.join().unwrap();
    assert_eq!(failure.error.kind, ErrorKind::Timeout);
    assert_eq!(failure.error.stage, "HTTP upload");
}
#[test]
fn waiting_for_http_response_obeys_finish_deadline_and_cancel() {
    for cancel in [false, true] {
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (root, worker) = server_with_response(move |_| {
            ready_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        });
        let engine = prepare(EngineConfig::OpenAiHttp(OpenAiHttpConfig::new(
            root,
            "fixture",
            Secret::new("key"),
        )))
        .unwrap();
        let session = engine.start(Default::default()).unwrap();
        let input = session.input();
        input
            .try_push(AudioChunk::mono(vec![0.1; 1600], 16000).unwrap())
            .unwrap();
        let result = std::thread::scope(|scope| {
            let finishing = scope.spawn(|| session.finish(Instant::now() + Duration::from_secs(2)));
            ready_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            if cancel {
                session.cancel();
            } else {
                // A second finish must shorten the in-flight HTTP wait too.
                let _ = session.finish(Instant::now() + Duration::from_millis(40));
            }
            finishing.join().unwrap()
        });
        release_tx.send(()).unwrap();
        worker.join().unwrap();
        let failure = result.unwrap_err();
        assert_eq!(
            failure.error.kind,
            if cancel {
                ErrorKind::Cancelled
            } else {
                ErrorKind::Timeout
            }
        );
    }
}
#[test]
fn multipart_preserves_root_and_serializes_valid_wav() {
    let (root, worker) = server(200, "{\"text\":\"hello\"}");
    assert_eq!(
        run(OpenAiHttpConfig::new(
            root,
            "fixture",
            Secret::new("test-secret")
        ))
        .unwrap()
        .transcript
        .text(),
        "hello"
    );
    let request = worker.join().unwrap();
    let text = String::from_utf8_lossy(&request);
    assert!(text.starts_with("POST /proxy/v1/audio/transcriptions HTTP/1.1"));
    assert!(text
        .to_lowercase()
        .contains("authorization: bearer test-secret"));
    let begin = request.windows(4).position(|w| w == b"RIFF").unwrap();
    let length = u32::from_le_bytes(request[begin + 4..begin + 8].try_into().unwrap()) as usize + 8;
    assert_eq!(
        audio::decode_wav_pcm16(&request[begin..begin + length])
            .unwrap()
            .samples
            .len(),
        1600
    );
}
#[test]
fn provider_error_retains_status_without_echoing_body() {
    let (root, worker) = server(401, "test-secret unauthorized");
    let error = run(OpenAiHttpConfig::new(
        root,
        "fixture",
        Secret::new("test-secret"),
    ))
    .unwrap_err()
    .error;
    assert_eq!(error.http_status, Some(401));
    assert_eq!(error.request_id.as_deref(), Some("fixture-123"));
    assert!(!error.message.contains("test-secret"));
    worker.join().unwrap();
}
#[test]
fn connection_failure_retains_a_sanitized_cause() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let mut config = OpenAiHttpConfig::new(
        format!("http://{address}/v1"),
        "fixture",
        Secret::new("key"),
    );
    config.timeouts.connect = Duration::from_millis(250);
    config.timeouts.response = Duration::from_secs(1);
    let error = run(config).unwrap_err().error;
    assert_eq!(error.kind, ErrorKind::Http);
    assert_eq!(error.stage, "request");
    assert_eq!(error.message, "HTTP connection failed");
    assert!(!error.message.contains(&address.to_string()));
}
#[test]
fn sse_replaces_deltas_with_final_text_and_requires_done() {
    let body = "data: {\"type\":\"transcript.text.delta\",\"delta\":\"helo\"}\r\n\r\ndata: {\"type\":\"transcript.text.done\",\"text\":\"hello\"}\r\n\r\n";
    let (root, worker) = server(200, body);
    let mut config = OpenAiHttpConfig::new(root, "fixture", Secret::new("key"));
    config.response = HttpResponse::Sse;
    assert_eq!(run(config).unwrap().transcript.text(), "hello");
    worker.join().unwrap();
    assert_eq!(
        parse_sse("data: {\"type\":\"transcript.text.delta\",\"delta\":\"he\"}\n\n").unwrap(),
        ("he".into(), None)
    );
    assert_eq!(
        parse_sse("data: {\"type\":").unwrap(),
        (String::new(), None)
    );
    assert_eq!(
        parse_sse(": keepalive\n\ndata: {\"type\":\"transcript.text.delta\",\"delta\":\"ok\"}\n\n")
            .unwrap(),
        ("ok".into(), None)
    );
    assert_eq!(
        parse_sse("data: ping\n\n").unwrap_err().kind,
        ErrorKind::Protocol
    );
}
#[test]
fn sse_done_finishes_fragmented_text_without_http_eof() {
    let (release_tx, release_rx) = mpsc::channel();
    let (root, worker) = server_with_response(move |socket| {
        write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n").unwrap();
        let events = "data: {\"type\":\"transcript.text.delta\",\"delta\":\"临时\"}\r\n\r\ndata: {\"type\":\"transcript.text.done\",\"text\":\"最终文本\"}\r\n\r\n";
        // Split inside UTF-8 characters, JSON tokens and CRLF delimiters.
        for chunk in events.as_bytes().chunks(7) {
            write!(socket, "{:x}\r\n", chunk.len()).unwrap();
            socket.write_all(chunk).unwrap();
            socket.write_all(b"\r\n").unwrap();
        }
        socket.flush().unwrap();
        // Do not send the terminating HTTP chunk or close the socket until
        // the client returns: success must come from the done event alone.
        release_rx.recv_timeout(Duration::from_secs(3)).unwrap();
    });
    let mut config = OpenAiHttpConfig::new(root, "fixture", Secret::new("key"));
    config.response = HttpResponse::Sse;
    config.timeouts.response = Duration::from_millis(500);
    let result = run(config);
    release_tx.send(()).unwrap();
    worker.join().unwrap();
    let outcome = result.unwrap();
    assert_eq!(outcome.transcript.text(), "最终文本");
    assert_eq!(outcome.transcript.segments.len(), 1);
}
#[test]
fn sse_completion_is_independent_of_trailing_events() {
    let done = "data: {\"type\":\"transcript.text.done\",\"text\":\"final\"}\n\n";
    for tail in [
        "data: {\"type\":\"error\"}\n\n",
        "data: {invalid json}\n\n",
        "data: {\"type\":\"transcript.text.done\",\"text\":\"duplicate\"}\n\n",
    ] {
        assert_eq!(
            parse_sse(&format!("{done}{tail}")).unwrap(),
            ("final".into(), Some("final".into()))
        );
    }
}
#[test]
fn sse_eof_without_done_is_failure_and_does_not_confirm_partial() {
    let (root, worker) = server(
        200,
        "data: {\"type\":\"transcript.text.delta\",\"delta\":\"temporary\"}\n\n",
    );
    let mut config = OpenAiHttpConfig::new(root, "fixture", Secret::new("key"));
    config.response = HttpResponse::Sse;
    let failure = run(config).unwrap_err();
    worker.join().unwrap();
    assert_eq!(failure.error.kind, ErrorKind::Protocol);
    assert!(failure.outcome.transcript.segments.is_empty());
}
#[test]
fn response_cap_and_missing_text_are_errors() {
    let (root, worker) = server(200, "{\"unexpected\":true}");
    assert_eq!(
        run(OpenAiHttpConfig::new(root, "fixture", Secret::new("key")))
            .unwrap_err()
            .error
            .kind,
        ErrorKind::Protocol
    );
    worker.join().unwrap();
    let (root, worker) = server(200, "a long response");
    let mut config = OpenAiHttpConfig::new(root, "fixture", Secret::new("key"));
    config.max_response_bytes = 2;
    assert_eq!(
        run(config).unwrap_err().error.kind,
        ErrorKind::ResourceLimit
    );
    worker.join().unwrap();
}
