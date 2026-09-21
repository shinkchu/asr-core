use super::{
    network::Network,
    websocket::{poll_event, protocol, server_error, wait_for_event, WsConnection, POLL_INTERVAL},
};
use crate::{
    audio,
    session::driver::{Driver, ResultSink},
    AsrError, OpenAiRealtimeConfig,
};
use base64::Engine as _;
use serde_json::{json, Value};
use std::{collections::HashMap, collections::HashSet, sync::Arc, time::Duration};

pub(crate) struct RealtimeDriver {
    config: OpenAiRealtimeConfig,
    connection: WsConnection,
    ready: bool,
    finishing: bool,
    index: u64,
    frames: usize,
    items: HashMap<String, u64>,
    completed: HashSet<String>,
    /// Items that delivered at least one delta but no `completed` event yet.
    /// Deltas land only in the shared result store, so without tracking them
    /// here the finish predicate below cannot see an unfinished utterance.
    open_deltas: HashSet<String>,
    finish_ack: bool,
}

impl RealtimeDriver {
    /// The finish wait ends only once the explicit commit is acknowledged,
    /// every committed item has a completed transcript, and no delta item is
    /// left open in the result store (its text exists only there, so a
    /// truncated server stream must surface as a timeout instead of a
    /// success that silently drops the partial text).
    fn finish_settled(&self) -> bool {
        self.finish_ack && self.all_committed_items_completed() && self.open_deltas.is_empty()
    }

    /// Builds an error for a realtime server error payload (`error` object
    /// with optional `code`/`message`) so quota and model failures stay
    /// diagnosable; the composition and field capping live in
    /// `websocket::server_error`.
    fn server_event_error(prefix: &str, error: &Value) -> AsrError {
        server_error(prefix, &error["code"], &error["message"])
    }

    pub(crate) fn new(config: OpenAiRealtimeConfig, network: Arc<Network>) -> Self {
        let connection = WsConnection::new(network, config.timeouts.clone());
        Self {
            config,
            connection,
            ready: false,
            finishing: false,
            index: 0,
            frames: 0,
            items: HashMap::new(),
            completed: HashSet::new(),
            open_deltas: HashSet::new(),
            finish_ack: false,
        }
    }

    fn receive_event(&mut self, sink: &ResultSink, timeout: Duration) -> Result<(), AsrError> {
        if let Some(value) = self.connection.receive_json(sink, timeout)? {
            self.event(value, sink)?;
        }
        Ok(())
    }

    fn event(&mut self, value: Value, sink: &ResultSink) -> Result<(), AsrError> {
        match value["type"].as_str() {
            Some("session.updated") => self.ready = true,
            Some("input_audio_buffer.committed") => {
                let id = value["item_id"]
                    .as_str()
                    .ok_or_else(|| protocol("commit missing item_id"))?;
                if !self.items.contains_key(id) {
                    if let Some(previous) = value["previous_item_id"].as_str() {
                        if !previous.is_empty() && !self.items.contains_key(previous) {
                            return Err(protocol("committed item predecessor is unknown"));
                        }
                        if !previous.is_empty() {
                            sink.register_id(previous)?;
                        }
                    }
                    sink.register_id(id)?;
                    let index = self.index;
                    if self.completed.contains(id) {
                        sink.index_unindexed(index, id, None, None)?;
                    }
                    self.items.insert(id.to_owned(), index);
                    self.index += 1;
                }
                if self.finishing {
                    self.finish_ack = true;
                }
            }
            Some("conversation.item.input_audio_transcription.delta") => {
                let id = value["item_id"]
                    .as_str()
                    .ok_or_else(|| protocol("delta missing item_id"))?;
                if self.completed.contains(id) {
                    return Ok(());
                }
                let delta = value["delta"]
                    .as_str()
                    .ok_or_else(|| protocol("delta missing text"))?;
                sink.append_delta(id.to_owned(), delta)?;
                self.open_deltas.insert(id.to_owned());
            }
            Some("conversation.item.input_audio_transcription.completed") => {
                let id = value["item_id"]
                    .as_str()
                    .ok_or_else(|| protocol("completion missing item_id"))?;
                if self.completed.contains(id) {
                    return Ok(());
                }
                let text = value["transcript"]
                    .as_str()
                    .ok_or_else(|| protocol("completion missing transcript"))?;
                if let Some(index) = self.items.get(id).copied() {
                    sink.commit(index, id, text, None, None)?;
                } else {
                    sink.complete_unindexed(id, text)?;
                }
                self.completed.insert(id.to_owned());
                self.open_deltas.remove(id);
            }
            Some("conversation.item.input_audio_transcription.failed") => {
                return Err(Self::server_event_error(
                    "realtime transcription failed",
                    &value["error"],
                ));
            }
            Some("error") => {
                let error = &value["error"];
                if self.finishing && error["code"] == "input_audio_buffer_commit_empty" {
                    self.finish_ack = true;
                } else {
                    return Err(Self::server_event_error(
                        "realtime provider reported an error",
                        error,
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn all_committed_items_completed(&self) -> bool {
        self.items.len() == self.completed.len()
            && self.items.keys().all(|id| self.completed.contains(id))
    }
}

impl Driver for RealtimeDriver {
    fn start(&mut self, sink: &ResultSink) -> Result<(), AsrError> {
        self.connection.connect(
            &self.config.endpoint,
            &self.config.api_key,
            &[("OpenAI-Beta", "realtime:v1")],
            sink,
        )?;
        self.connection.send_json(
            json!({
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {"model": self.config.model},
                            "turn_detection": if self.config.server_vad {
                                json!({"type": "server_vad"})
                            } else {
                                Value::Null
                            }
                        }
                    }
                }
            }),
            sink,
        )?;
        let timeout = self.connection.response_timeout();
        wait_for_event(timeout, |remaining| match remaining {
            None => Ok(self.ready),
            Some(remaining) => self.receive_event(sink, remaining).map(|()| self.ready),
        })
    }

    fn push(&mut self, samples: &[f32], sink: &ResultSink) -> Result<(), AsrError> {
        if samples.is_empty() {
            return Ok(());
        }
        self.frames += samples.len();
        self.connection.send_json(
            json!({
                "type": "input_audio_buffer.append",
                "audio": base64::engine::general_purpose::STANDARD.encode(audio::pcm_bytes(samples))
            }),
            sink,
        )
    }

    fn poll(&mut self, sink: &ResultSink) -> Result<(), AsrError> {
        poll_event(sink, |timeout| self.receive_event(sink, timeout))
    }

    fn poll_interval(&self) -> Option<Duration> {
        Some(POLL_INTERVAL)
    }

    fn finish(&mut self, sink: &ResultSink) -> Result<(), AsrError> {
        if self.config.server_vad && self.frames > 0 {
            self.ready = false;
            self.connection.send_json(
                json!({
                    "type": "session.update",
                    "session": {
                        "type": "transcription",
                        "audio": {"input": {"turn_detection": null}}
                    }
                }),
                sink,
            )?;
            let timeout = self.connection.response_timeout();
            wait_for_event(timeout, |remaining| match remaining {
                None => Ok(self.ready),
                Some(remaining) => self.receive_event(sink, remaining).map(|()| self.ready),
            })?;
        }
        if self.frames > 0 {
            self.push(&[0.0; 2400], sink)?;
        }

        self.finishing = true;
        if self.frames == 0 {
            self.finish_ack = true;
        } else {
            self.connection
                .send_json(json!({"type": "input_audio_buffer.commit"}), sink)?;
        }

        let timeout = self.connection.response_timeout();
        // An open delta only exists in the result store: the item may never
        // have been observed as committed, so `all_committed_items_completed`
        // cannot see it. Waiting here turns a truncated server stream into a
        // visible timeout instead of a success whose settlement silently
        // drops the partial text.
        wait_for_event(timeout, |remaining| match remaining {
            None => Ok(self.finish_settled()),
            Some(remaining) => self
                .receive_event(sink, remaining)
                .map(|()| self.finish_settled()),
        })?;
        self.connection.close(sink);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
