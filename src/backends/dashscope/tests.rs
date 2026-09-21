use super::*;
use crate::ErrorKind;
use crate::{
    backends::test_util::{prepare, read, send, server},
    AudioBuffer, AudioChunk, Engine, EngineConfig, EngineOptions, Secret, SessionOptions, Timeouts,
};
use std::{
    net::TcpListener,
    time::{Duration, Instant},
};

fn driver() -> DashScopeDriver {
    DashScopeDriver::new(
        DashScopeConfig {
            endpoint: "ws://127.0.0.1:1".into(),
            model: "fixture".into(),
            api_key: Secret::new("fixture-key"),
            timeouts: Default::default(),
        },
        Arc::new(Network::new().unwrap()),
    )
}

fn ws_engine(endpoint: String, response: Duration) -> Engine {
    Engine::prepare(
        EngineConfig::DashScope(DashScopeConfig {
            endpoint,
            model: "fixture".into(),
            api_key: Secret::new("fixture-key"),
            timeouts: Timeouts {
                response,
                ..Default::default()
            },
        }),
        EngineOptions::default(),
    )
    .unwrap()
}

#[test]
fn interval_finals_keep_the_historical_dedup_behavior() {
    let sentence = json!({"text": "confirmed", "begin_time": 0, "end_time": 100});
    let (begin, end) = DashScopeDriver::sentence_times(&sentence);
    let identity = DashScopeDriver::completed_identity(&sentence, begin, end);
    assert_eq!(identity.as_deref(), Some("0:100"));
    let mut driver = driver();
    assert!(driver.mark_completed(identity.clone()));
    // The interval key still drops a re-sent final for the same
    // interval, and distinct intervals never collide.
    assert!(!driver.mark_completed(identity));
    assert!(driver.mark_completed(DashScopeDriver::completed_identity(
        &json!({}),
        Some(100.0),
        Some(200.0)
    )));
}

#[test]
fn finals_without_identity_always_commit() {
    // Without `sentence_id` or timestamps a re-delivered final cannot be
    // told apart from a legitimate repeat, so nothing is deduplicated.
    assert_eq!(
        DashScopeDriver::completed_identity(&json!({}), None, None),
        None
    );
    // A half-present interval is not an identity either.
    assert_eq!(
        DashScopeDriver::completed_identity(&json!({}), Some(0.0), None),
        None
    );
    let mut driver = driver();
    assert!(driver.mark_completed(None));
    // Two identical timestamp-less finals both reach the transcript.
    assert!(driver.mark_completed(None));
}

#[test]
fn sentence_id_joins_the_interval_in_the_dedup_identity() {
    let sentence = json!({"text": "confirmed", "sentence_id": 3, "begin_time": 0, "end_time": 100});
    let (begin, end) = DashScopeDriver::sentence_times(&sentence);
    let identity = DashScopeDriver::completed_identity(&sentence, begin, end);
    assert_eq!(identity.as_deref(), Some("id:3:0:100"));
    let mut driver = driver();
    assert!(driver.mark_completed(identity.clone()));
    assert!(!driver.mark_completed(identity));
    // A distinct sentence id breaks the interval collision even when the
    // server repeats the same interval for every sentence.
    assert!(driver.mark_completed(DashScopeDriver::completed_identity(
        &json!({"sentence_id": 4}),
        Some(0.0),
        Some(100.0)
    )));
}

#[test]
fn sentence_id_alone_deduplicates_redelivered_finals() {
    // `sentence_id` is the protocol's per-task sentence sequence (0 on
    // heartbeats) and parses uniformly whether the server sends it as an
    // integer or a floating point number.
    let identity = DashScopeDriver::completed_identity(&json!({"sentence_id": 7}), None, None);
    assert_eq!(identity.as_deref(), Some("id:7"));
    assert_eq!(
        DashScopeDriver::completed_identity(&json!({"sentence_id": 7.0}), None, None),
        identity
    );
    let mut driver = driver();
    assert!(driver.mark_completed(identity.clone()));
    assert!(!driver.mark_completed(identity));
    assert!(driver.mark_completed(DashScopeDriver::completed_identity(
        &json!({"sentence_id": 8}),
        None,
        None
    )));
}

#[test]
fn float_timestamps_deduplicate_and_match_the_commit_path() {
    let sentence = json!({
        "text": "confirmed",
        "sentence_end": true,
        "begin_time": 100.5,
        "end_time": 250.75
    });
    let (begin, end) = DashScopeDriver::sentence_times(&sentence);
    assert_eq!((begin, end), (Some(100.5), Some(250.75)));
    let identity = DashScopeDriver::completed_identity(&sentence, begin, end);
    let mut driver = driver();
    assert!(driver.mark_completed(identity.clone()));
    assert!(!driver.mark_completed(identity));
    // The commit path converts the very same parse into seconds.
    assert!((begin.unwrap() / 1000.0 - 0.1005).abs() < f64::EPSILON * 100.0);
    assert!((end.unwrap() / 1000.0 - 0.25075).abs() < f64::EPSILON * 100.0);
}

#[test]
fn sentence_times_parses_integers_floats_and_missing_uniformly() {
    let (begin, end) = DashScopeDriver::sentence_times(&json!({"begin_time": 0, "end_time": 100}));
    assert_eq!((begin, end), (Some(0.0), Some(100.0)));
    let (begin, end) =
        DashScopeDriver::sentence_times(&json!({"begin_time": 100.5, "end_time": 250.75}));
    assert_eq!((begin, end), (Some(100.5), Some(250.75)));
    let (begin, end) = DashScopeDriver::sentence_times(&json!({"text": "x"}));
    assert_eq!((begin, end), (None, None));
    let (begin, end) =
        DashScopeDriver::sentence_times(&json!({"begin_time": "0", "end_time": null}));
    assert_eq!((begin, end), (None, None));
}

#[test]
fn task_failed_composes_the_server_code_and_message() {
    let error = DashScopeDriver::task_failed_error(&json!({
        "header": {
            "event": "task-failed",
            "error_code": "MODEL_NOT_EXIST",
            "error_message": "Model not exist"
        }
    }));
    assert_eq!(error.kind, ErrorKind::Protocol);
    assert_eq!(
        error.message,
        "DashScope task failed (MODEL_NOT_EXIST: Model not exist)"
    );
}

#[test]
fn task_failed_message_without_code_only_keeps_the_message() {
    let error = DashScopeDriver::task_failed_error(&json!({
        "header": {"error_message": "quota exhausted"}
    }));
    assert_eq!(error.message, "DashScope task failed (quota exhausted)");
}

#[test]
fn task_failed_without_renderable_details_falls_back_to_static_message() {
    for header in [
        json!({"event": "task-failed"}),
        json!({"error_code": "QUOTA_EXCEEDED"}),
        json!({"error_code": 7, "error_message": "   "}),
    ] {
        let error = DashScopeDriver::task_failed_error(&json!({"header": header}));
        assert_eq!(error.kind, ErrorKind::Protocol);
        assert_eq!(error.message, "DashScope task failed");
    }
}

#[test]
fn task_failed_oversized_server_fields_are_truncated() {
    let error = DashScopeDriver::task_failed_error(&json!({
        "header": {"error_code": "C", "error_message": "x".repeat(1000)}
    }));
    assert_eq!(
        error.message,
        format!(
            "DashScope task failed (C: {}…)",
            "x".repeat(crate::backends::websocket::SERVER_ERROR_FIELD_LIMIT)
        )
    );
}

#[test]
fn ownership_rejects_foreign_and_accepts_matching_or_absent_task_ids() {
    let driver = driver();
    assert!(!driver.owns_event(&json!({
        "header": {"event": "result-generated", "task_id": "00000000-0000-0000-0000-000000000000"}
    })));
    // The server echoes the locally generated id back, including on the
    // first `task-started` that confirms the handshake.
    assert!(driver.owns_event(&json!({
        "header": {"event": "task-started", "task_id": driver.task_id}
    })));
    // An event without a task id cannot be attributed to another task.
    assert!(driver.owns_event(&json!({"header": {"event": "task-started"}})));
}

#[test]
fn foreign_task_started_does_not_confirm_the_handshake() {
    let (endpoint, worker) = server(|mut socket| {
        read(&mut socket);
        // A `task-started` for another task must not mark this session
        // ready; the driver keeps waiting for its own id.
        send(
            &mut socket,
            json!({"header": {"event": "task-started", "task_id": "not-ours"}}),
        );
        std::thread::sleep(Duration::from_millis(300));
    });
    let engine = ws_engine(endpoint, Duration::from_millis(100));
    let session = engine.start(Default::default()).unwrap();
    let began = Instant::now();
    assert_eq!(
        session
            .finish(began + Duration::from_secs(2))
            .unwrap_err()
            .error
            .kind,
        ErrorKind::Timeout
    );
    assert!(began.elapsed() < Duration::from_secs(1));
    worker.join().unwrap();
}

#[test]
fn foreign_task_events_never_reach_the_transcript() {
    let (endpoint, worker) = server(|mut socket| {
        let task = read(&mut socket);
        let task_id = task["header"]["task_id"].as_str().unwrap().to_string();
        send(
            &mut socket,
            json!({"header": {"event": "task-started", "task_id": task_id}}),
        );
        assert!(matches!(socket.read().unwrap(), Message::Binary(_)));
        // An event misrouted from another task must neither enter the
        // transcript nor advance the sentence index.
        send(
            &mut socket,
            json!({
                "header": {"event": "result-generated", "task_id": "not-ours"},
                "payload": {"output": {"sentence": {
                    "text": "misrouted", "sentence_end": true,
                    "begin_time": 0, "end_time": 100
                }}}
            }),
        );
        send(
            &mut socket,
            json!({
                "header": {"event": "result-generated", "task_id": task_id},
                "payload": {"output": {"sentence": {
                    "text": "confirmed", "sentence_end": true,
                    "begin_time": 0, "end_time": 100
                }}}
            }),
        );
        assert_eq!(read(&mut socket)["header"]["action"], "finish-task");
        send(
            &mut socket,
            json!({"header": {"event": "task-finished", "task_id": task_id}}),
        );
        assert!(matches!(socket.read().unwrap(), Message::Close(_)));
    });
    let engine = ws_engine(endpoint, Duration::from_secs(3));
    let session = engine.start(Default::default()).unwrap();
    session
        .input()
        .try_push(AudioChunk::mono(vec![0.1; 1600], 16000).unwrap())
        .unwrap();
    assert_eq!(
        session
            .finish(Instant::now() + Duration::from_secs(3))
            .unwrap()
            .transcript
            .text(),
        "confirmed"
    );
    worker.join().unwrap();
}

#[test]
fn identical_timestampless_finals_commit_and_id_redelivery_deduplicates() {
    let (endpoint, worker) = server(|mut socket| {
        let task = read(&mut socket);
        let task_id = task["header"]["task_id"].as_str().unwrap().to_string();
        send(
            &mut socket,
            json!({"header": {"event": "task-started", "task_id": task_id}}),
        );
        assert!(matches!(socket.read().unwrap(), Message::Binary(_)));
        // Two legitimate finals with identical text and no identity
        // fields: both must reach the transcript.
        let repeated = json!({
            "header": {"event": "result-generated", "task_id": task_id},
            "payload": {"output": {"sentence": {
                "text": "confirmed", "sentence_end": true
            }}}
        });
        send(&mut socket, repeated.clone());
        send(&mut socket, repeated);
        // A re-delivered final carrying `sentence_id` still dedupes.
        let identified = json!({
            "header": {"event": "result-generated", "task_id": task_id},
            "payload": {"output": {"sentence": {
                "text": "tail", "sentence_end": true,
                "sentence_id": 5, "begin_time": 100, "end_time": 200
            }}}
        });
        send(&mut socket, identified.clone());
        send(&mut socket, identified);
        assert_eq!(read(&mut socket)["header"]["action"], "finish-task");
        send(
            &mut socket,
            json!({"header": {"event": "task-finished", "task_id": task_id}}),
        );
        assert!(matches!(socket.read().unwrap(), Message::Close(_)));
    });
    let engine = ws_engine(endpoint, Duration::from_secs(3));
    let session = engine.start(Default::default()).unwrap();
    session
        .input()
        .try_push(AudioChunk::mono(vec![0.1; 1600], 16000).unwrap())
        .unwrap();
    let transcript = session
        .finish(Instant::now() + Duration::from_secs(3))
        .unwrap()
        .transcript;
    assert_eq!(transcript.segments.len(), 3);
    assert_eq!(transcript.text(), "confirmed confirmed tail");
    worker.join().unwrap();
}

#[test]
fn dashscope_handshake_uuid_and_finish_only_confirms_final_sentences() {
    let (endpoint, worker) = server(|mut socket| {
        let task = read(&mut socket);
        uuid::Uuid::parse_str(task["header"]["task_id"].as_str().unwrap()).unwrap();
        assert_eq!(task["payload"]["parameters"]["heartbeat"], true);
        send(&mut socket, json!({"header":{"event":"task-started"}}));
        assert!(matches!(socket.read().unwrap(), Message::Binary(_)));
        send(
            &mut socket,
            json!({"header":{"event":"result-generated"},"payload":{"output":{"sentence":{"text":"temporary","sentence_end":false}}}}),
        );
        send(
            &mut socket,
            json!({"header":{"event":"result-generated"},"payload":{"output":{"sentence":{"text":"ignored","sentence_end":true,"heartbeat":true}}}}),
        );
        assert_eq!(read(&mut socket)["header"]["action"], "finish-task");
        for _ in 0..2 {
            send(
                &mut socket,
                json!({"header":{"event":"result-generated"},"payload":{"output":{"sentence":{"text":"confirmed","sentence_end":true,"begin_time":0,"end_time":100}}}}),
            );
        }
        send(&mut socket, json!({"header":{"event":"task-finished"}}));
        // The success path must initiate the WebSocket close handshake.
        assert!(matches!(socket.read().unwrap(), Message::Close(_)));
    });
    let engine = prepare(EngineConfig::DashScope(DashScopeConfig {
        endpoint,
        model: "fixture".into(),
        api_key: Secret::new("fixture-key"),
        timeouts: Default::default(),
    }))
    .unwrap();
    let session = engine.start(Default::default()).unwrap();
    session
        .input()
        .try_push(AudioChunk::mono(vec![0.1; 1600], 16000).unwrap())
        .unwrap();
    assert_eq!(
        session
            .finish(Instant::now() + Duration::from_secs(3))
            .unwrap()
            .transcript
            .text(),
        "confirmed"
    );
    worker.join().unwrap();
}

#[test]
fn dashscope_heartbeats_do_not_extend_finish_deadline() {
    let (endpoint, worker) = server(|mut socket| {
        read(&mut socket);
        send(&mut socket, json!({"header":{"event":"task-started"}}));
        read(&mut socket);
        for _ in 0..100 {
            if socket
                .send(Message::Text(
                    json!({"header":{"event":"result-generated"},"payload":{"output":{"sentence":{"text":"ignored","sentence_end":true,"heartbeat":true}}}})
                        .to_string()
                        .into(),
                ))
                .is_err()
            {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    });
    let engine = prepare(EngineConfig::DashScope(DashScopeConfig {
        endpoint,
        model: "fixture".into(),
        api_key: Secret::new("fixture-key"),
        timeouts: Default::default(),
    }))
    .unwrap();
    let session = engine.start(Default::default()).unwrap();
    let began = Instant::now();
    assert_eq!(
        session
            .finish(began + Duration::from_millis(200))
            .unwrap_err()
            .error
            .kind,
        ErrorKind::Timeout
    );
    assert!(began.elapsed() < Duration::from_secs(1));
    worker.join().unwrap();
}

#[test]
fn dashscope_task_failure_and_early_disconnect_are_not_success() {
    for task_failure in [true, false] {
        let (endpoint, worker) = server(move |mut socket| {
            read(&mut socket);
            if task_failure {
                send(&mut socket, json!({"header":{"event":"task-failed"}}));
            }
        });
        let engine = prepare(EngineConfig::DashScope(DashScopeConfig {
            endpoint,
            model: "fixture".into(),
            api_key: Secret::new("fixture-key"),
            timeouts: Default::default(),
        }))
        .unwrap();
        let session = engine.start(Default::default()).unwrap();
        assert_eq!(
            session
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap_err()
                .error
                .kind,
            ErrorKind::Protocol
        );
        worker.join().unwrap();
    }
}

#[test]
fn dashscope_stalled_handshake_obeys_connect_timeout() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let worker = std::thread::spawn(move || {
        let (_socket, _) = listener.accept().unwrap();
        std::thread::sleep(Duration::from_millis(250));
    });
    let timeouts = Timeouts {
        connect: Duration::from_millis(50),
        ..Default::default()
    };
    let engine = prepare(EngineConfig::DashScope(DashScopeConfig {
        endpoint: format!("ws://{address}"),
        model: "fixture".into(),
        api_key: Secret::new("fixture-key"),
        timeouts,
    }))
    .unwrap();
    let session = engine.start(Default::default()).unwrap();
    let began = Instant::now();
    assert_eq!(
        session
            .finish(began + Duration::from_secs(2))
            .unwrap_err()
            .error
            .kind,
        ErrorKind::Timeout
    );
    assert!(began.elapsed() < Duration::from_secs(1));
    worker.join().unwrap();
}

#[test]
fn transcribe_deadline_covers_a_stalled_websocket_start() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let worker = std::thread::spawn(move || {
        listener.set_nonblocking(true).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while Instant::now() < deadline {
            match listener.accept() {
                Ok((_socket, _)) => {
                    std::thread::sleep(Duration::from_millis(300));
                    return;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("fixture accept failed: {error}"),
            }
        }
    });
    let engine = prepare(EngineConfig::DashScope(DashScopeConfig {
        endpoint: format!("ws://{address}"),
        model: "fixture".into(),
        api_key: Secret::new("fixture-key"),
        timeouts: Default::default(),
    }))
    .unwrap();
    let audio = AudioBuffer::mono(vec![0.1; 1600], 16000).unwrap();
    let began = Instant::now();
    let failure = engine
        .transcribe(
            &audio,
            SessionOptions::new(audio.spec),
            began + Duration::from_millis(100),
        )
        .unwrap_err();
    assert_eq!(failure.error.kind, ErrorKind::Timeout);
    assert!(began.elapsed() < Duration::from_secs(1));
    worker.join().unwrap();
}
