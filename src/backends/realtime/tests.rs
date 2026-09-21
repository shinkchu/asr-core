use super::*;
use crate::ErrorKind;
use crate::{
    backends::test_util::{prepare, read, send, server},
    AudioBuffer, AudioChunk, Engine, EngineConfig, Secret, Session, SessionOptions,
};
use std::{
    net::{TcpListener, TcpStream},
    time::{Duration, Instant},
};
use tokio_tungstenite::tungstenite::{self, accept_hdr, Message, WebSocket};

#[test]
fn server_error_payload_composes_the_server_code_and_message() {
    let error = RealtimeDriver::server_event_error(
        "realtime provider reported an error",
        &json!({"code": "invalid_api_key", "message": "Incorrect API key provided"}),
    );
    assert_eq!(error.kind, ErrorKind::Protocol);
    assert_eq!(
        error.message,
        "realtime provider reported an error (invalid_api_key: Incorrect API key provided)"
    );
}

#[test]
fn server_error_payload_message_without_code_only_keeps_the_message() {
    let error = RealtimeDriver::server_event_error(
        "realtime transcription failed",
        &json!({"message": "transcription model failed"}),
    );
    assert_eq!(
        error.message,
        "realtime transcription failed (transcription model failed)"
    );
}

#[test]
fn server_error_payload_without_renderable_message_falls_back_to_static() {
    for payload in [
        json!({}),
        json!({"code": "quota_exceeded"}),
        json!({"code": 7, "message": "   "}),
    ] {
        let error = RealtimeDriver::server_event_error("realtime transcription failed", &payload);
        assert_eq!(error.kind, ErrorKind::Protocol);
        assert_eq!(error.message, "realtime transcription failed");
    }
}

#[test]
fn server_error_payload_oversized_fields_are_truncated() {
    let error = RealtimeDriver::server_event_error(
        "realtime transcription failed",
        &json!({"code": "C", "message": "y".repeat(1000)}),
    );
    assert_eq!(
        error.message,
        format!(
            "realtime transcription failed (C: {}…)",
            "y".repeat(crate::backends::websocket::SERVER_ERROR_FIELD_LIMIT)
        )
    );
}

fn read_until_realtime_commit(socket: &mut WebSocket<TcpStream>) {
    loop {
        let value = read(socket);
        if value["type"] == "input_audio_buffer.commit" {
            return;
        }
        assert_eq!(value["type"], "input_audio_buffer.append");
    }
}

fn realtime_engine(endpoint: String, response: Duration) -> Engine {
    let mut config = OpenAiRealtimeConfig::new(endpoint, "fixture", Secret::new("fixture-key"));
    config.server_vad = false;
    config.timeouts.response = response;
    prepare(EngineConfig::OpenAiRealtime(config)).unwrap()
}

fn push_realtime_audio(session: &Session) {
    session
        .input()
        .try_push(AudioChunk::mono(vec![0.1; 1600], 16000).unwrap())
        .unwrap();
}

#[test]
fn realtime_indexes_early_final_and_ignores_duplicates_and_late_delta() {
    let (endpoint, worker) = server(|mut socket| {
        let update = read(&mut socket);
        assert_eq!(update["session"]["audio"]["input"]["format"]["rate"], 24000);
        send(&mut socket, json!({"type":"session.updated"}));
        read_until_realtime_commit(&mut socket);
        send(
            &mut socket,
            json!({"type":"conversation.item.input_audio_transcription.completed","item_id":"b","transcript":"second"}),
        );
        send(
            &mut socket,
            json!({"type":"conversation.item.input_audio_transcription.delta","item_id":"b","delta":"late"}),
        );
        send(
            &mut socket,
            json!({"type":"conversation.item.input_audio_transcription.completed","item_id":"b","transcript":"different"}),
        );
        send(
            &mut socket,
            json!({"type":"input_audio_buffer.committed","item_id":"a","previous_item_id":null}),
        );
        send(
            &mut socket,
            json!({"type":"input_audio_buffer.committed","item_id":"a","previous_item_id":null}),
        );
        send(
            &mut socket,
            json!({"type":"input_audio_buffer.committed","item_id":"b","previous_item_id":"a"}),
        );
        send(
            &mut socket,
            json!({"type":"conversation.item.input_audio_transcription.delta","item_id":"a","delta":"fir"}),
        );
        send(
            &mut socket,
            json!({"type":"conversation.item.input_audio_transcription.delta","item_id":"a","delta":"st"}),
        );
        send(
            &mut socket,
            json!({"type":"conversation.item.input_audio_transcription.completed","item_id":"a","transcript":"first"}),
        );
    });
    let session = realtime_engine(endpoint, Duration::from_secs(3))
        .start(Default::default())
        .unwrap();
    push_realtime_audio(&session);
    assert_eq!(
        session
            .finish(Instant::now() + Duration::from_secs(3))
            .unwrap()
            .transcript
            .text(),
        "first second"
    );
    worker.join().unwrap();
}

#[test]
fn realtime_rejects_unknown_committed_predecessor() {
    let (endpoint, worker) = server(|mut socket| {
        read(&mut socket);
        send(&mut socket, json!({"type":"session.updated"}));
        read_until_realtime_commit(&mut socket);
        send(
            &mut socket,
            json!({"type":"input_audio_buffer.committed","item_id":"b","previous_item_id":"unknown"}),
        );
    });
    let session = realtime_engine(endpoint, Duration::from_secs(3))
        .start(Default::default())
        .unwrap();
    push_realtime_audio(&session);
    let failure = session
        .finish(Instant::now() + Duration::from_secs(3))
        .unwrap_err();
    assert_eq!(failure.error.kind, ErrorKind::Protocol);
    assert_eq!(
        failure.error.message,
        "committed item predecessor is unknown"
    );
    worker.join().unwrap();
}

#[test]
fn realtime_server_vad_finish_waits_for_update_barrier_and_final_item() {
    let (endpoint, worker) = server(|mut socket| {
        read(&mut socket);
        send(&mut socket, json!({"type":"session.updated"}));
        loop {
            if read(&mut socket)["type"] == "session.update" {
                break;
            }
        }
        send(
            &mut socket,
            json!({"type":"input_audio_buffer.committed","item_id":"auto","previous_item_id":null}),
        );
        send(
            &mut socket,
            json!({"type":"conversation.item.input_audio_transcription.completed","item_id":"auto","transcript":"first"}),
        );
        send(&mut socket, json!({"type":"session.updated"}));
        assert_eq!(read(&mut socket)["type"], "input_audio_buffer.append");
        assert_eq!(read(&mut socket)["type"], "input_audio_buffer.commit");
        send(
            &mut socket,
            json!({"type":"input_audio_buffer.committed","item_id":"tail","previous_item_id":"auto"}),
        );
        send(
            &mut socket,
            json!({"type":"conversation.item.input_audio_transcription.completed","item_id":"tail","transcript":"last"}),
        );
    });
    let config = OpenAiRealtimeConfig::new(endpoint, "fixture", Secret::new("fixture-key"));
    let engine = prepare(EngineConfig::OpenAiRealtime(config)).unwrap();
    let session = engine.start(Default::default()).unwrap();
    push_realtime_audio(&session);
    assert_eq!(
        session
            .finish(Instant::now() + Duration::from_secs(3))
            .unwrap()
            .transcript
            .text(),
        "first last"
    );
    worker.join().unwrap();
}

#[test]
fn realtime_finish_does_not_accept_equal_but_mismatched_item_sets() {
    let (endpoint, worker) = server(|mut socket| {
        read(&mut socket);
        send(&mut socket, json!({"type":"session.updated"}));
        read_until_realtime_commit(&mut socket);
        send(
            &mut socket,
            json!({"type":"conversation.item.input_audio_transcription.completed","item_id":"orphan","transcript":"wrong"}),
        );
        send(
            &mut socket,
            json!({"type":"input_audio_buffer.committed","item_id":"tail","previous_item_id":null}),
        );
        std::thread::sleep(Duration::from_millis(400));
    });
    let session = realtime_engine(endpoint, Duration::from_secs(3))
        .start(Default::default())
        .unwrap();
    push_realtime_audio(&session);
    assert_eq!(
        session
            .finish(Instant::now() + Duration::from_millis(200))
            .unwrap_err()
            .error
            .kind,
        ErrorKind::Timeout
    );
    worker.join().unwrap();
}

#[test]
#[allow(clippy::result_large_err)]
fn realtime_handshake_sends_openai_beta_header_and_closes_on_finish() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let worker = std::thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(20)))
            .unwrap();
        let mut socket = accept_hdr(
            socket,
            |request: &tungstenite::handshake::server::Request, response| {
                assert_eq!(request.headers()["authorization"], "Bearer fixture-key");
                // The official Realtime endpoint rejects the upgrade
                // without this header, so it must be on the request.
                assert_eq!(request.headers()["openai-beta"], "realtime:v1");
                Ok(response)
            },
        )
        .unwrap();
        assert_eq!(read(&mut socket)["type"], "session.update");
        send(&mut socket, json!({"type":"session.updated"}));
        // Zero frames make finish resolve locally; the server must see
        // the graceful Close handshake.
        assert!(matches!(socket.read().unwrap(), Message::Close(_)));
    });
    let session = realtime_engine(format!("ws://{address}/fixture"), Duration::from_secs(3))
        .start(Default::default())
        .unwrap();
    session
        .finish(Instant::now() + Duration::from_secs(3))
        .unwrap();
    worker.join().unwrap();
}

#[test]
fn realtime_finish_waits_for_unfinished_delta_items() {
    // A delta for an item whose committed event never arrives exists only
    // in the result store. The finish predicate must keep waiting for it
    // instead of succeeding and silently dropping the partial text.
    let (endpoint, worker) = server(|mut socket| {
        read(&mut socket);
        send(&mut socket, json!({"type":"session.updated"}));
        read_until_realtime_commit(&mut socket);
        send(
            &mut socket,
            json!({"type":"conversation.item.input_audio_transcription.delta","item_id":"stray","delta":"half"}),
        );
        send(
            &mut socket,
            json!({"type":"input_audio_buffer.committed","item_id":"tail","previous_item_id":null}),
        );
        send(
            &mut socket,
            json!({"type":"conversation.item.input_audio_transcription.completed","item_id":"tail","transcript":"last"}),
        );
        std::thread::sleep(Duration::from_millis(400));
    });
    let session = realtime_engine(endpoint, Duration::from_secs(3))
        .start(Default::default())
        .unwrap();
    push_realtime_audio(&session);
    let failure = session
        .finish(Instant::now() + Duration::from_millis(200))
        .unwrap_err();
    assert_eq!(failure.error.kind, ErrorKind::Timeout);
    // The stranded partial must not surface as text; the legitimately
    // completed item is still retained in the failure outcome.
    assert_eq!(failure.outcome.transcript.text(), "last");
    worker.join().unwrap();
}

#[test]
fn realtime_protocol_id_and_unindexed_limits_are_enforced() {
    #[derive(Clone, Copy, Debug)]
    enum Case {
        PerItem,
        TotalBytes,
        Outstanding,
        Unindexed,
    }

    for case in [
        Case::PerItem,
        Case::TotalBytes,
        Case::Outstanding,
        Case::Unindexed,
    ] {
        let (endpoint, worker) = server(move |mut socket| {
            read(&mut socket);
            send(&mut socket, json!({"type":"session.updated"}));
            read_until_realtime_commit(&mut socket);
            // Keep the finish barrier open while the fixture sends more than
            // one item after the explicit commit acknowledgement.
            send(
                &mut socket,
                json!({"type":"conversation.item.input_audio_transcription.completed","item_id":"guard","transcript":""}),
            );
            match case {
                Case::PerItem => send(
                    &mut socket,
                    json!({"type":"input_audio_buffer.committed","item_id":"x".repeat(1025),"previous_item_id":null}),
                ),
                Case::TotalBytes => {
                    let mut previous: Option<String> = None;
                    for index in 0..=1023 {
                        let prefix = format!("{index:04}");
                        let id = format!("{prefix}{}", "x".repeat(1024 - prefix.len()));
                        send(
                            &mut socket,
                            json!({"type":"input_audio_buffer.committed","item_id":id,"previous_item_id":previous}),
                        );
                        if index == 1023 {
                            break;
                        }
                        send(
                            &mut socket,
                            json!({"type":"conversation.item.input_audio_transcription.completed","item_id":id,"transcript":""}),
                        );
                        previous = Some(id);
                    }
                }
                Case::Outstanding => {
                    let mut previous: Option<String> = None;
                    for index in 0..129 {
                        let id = format!("item-{index}");
                        send(
                            &mut socket,
                            json!({"type":"input_audio_buffer.committed","item_id":id,"previous_item_id":previous}),
                        );
                        previous = Some(id);
                    }
                }
                Case::Unindexed => {
                    for index in 0..128 {
                        send(
                            &mut socket,
                            json!({"type":"conversation.item.input_audio_transcription.completed","item_id":format!("item-{index}"),"transcript":""}),
                        );
                    }
                }
            }
        });
        let session = realtime_engine(endpoint, Duration::from_secs(15))
            .start(Default::default())
            .unwrap();
        push_realtime_audio(&session);
        let result = session.finish(Instant::now() + Duration::from_secs(20));
        assert!(result.is_err(), "{case:?} unexpectedly completed");
        let error = result.unwrap_err().error;
        assert_eq!(
            error.kind,
            ErrorKind::ResourceLimit,
            "{case:?} failed with {error:?}"
        );
        worker.join().unwrap();
    }
}

#[test]
fn realtime_late_events_cannot_change_cancelled_outcome() {
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let (endpoint, worker) = server(move |mut socket| {
        read(&mut socket);
        send(&mut socket, json!({"type":"session.updated"}));
        ready_tx.send(()).unwrap();
        std::thread::sleep(Duration::from_millis(50));
        let _ = socket.send(Message::Text(
            json!({"type":"conversation.item.input_audio_transcription.completed","item_id":"late","transcript":"late"})
                .to_string()
                .into(),
        ));
    });
    let session = realtime_engine(endpoint, Duration::from_secs(3))
        .start(Default::default())
        .unwrap();
    ready_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    push_realtime_audio(&session);
    session.cancel();
    let failure = session
        .finish(Instant::now() + Duration::from_secs(2))
        .unwrap_err();
    assert_eq!(failure.error.kind, ErrorKind::Cancelled);
    assert!(failure.outcome.transcript.segments.is_empty());
    worker.join().unwrap();
}

#[test]
fn transcribe_preserves_a_backend_error_that_closes_input_mid_submission() {
    let (endpoint, worker) = server(|mut socket| {
        read(&mut socket);
        send(&mut socket, json!({"type":"session.updated"}));
        assert_eq!(read(&mut socket)["type"], "input_audio_buffer.append");
        send(
            &mut socket,
            json!({"type":"error","error":{"code":"fixture"}}),
        );
    });
    let engine = realtime_engine(endpoint, Duration::from_secs(3));
    let audio = AudioBuffer::mono(vec![0.1; 24000], 24000).unwrap();
    let failure = engine
        .transcribe(
            &audio,
            SessionOptions {
                queue_duration: Duration::from_millis(10),
                max_chunk_duration: Duration::from_millis(10),
                ..SessionOptions::new(audio.spec)
            },
            Instant::now() + Duration::from_secs(3),
        )
        .unwrap_err();
    assert_eq!(failure.error.kind, ErrorKind::Protocol);
    assert_eq!(failure.error.message, "realtime provider reported an error");
    assert!(failure.outcome.received_frames < audio.samples.len() as u64);
    worker.join().unwrap();
}
