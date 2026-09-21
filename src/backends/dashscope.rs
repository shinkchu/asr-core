use super::{
    network::Network,
    websocket::{poll_event, protocol, server_error, wait_for_event, WsConnection, POLL_INTERVAL},
};
use crate::{
    audio,
    session::driver::{Driver, ResultSink},
    AsrError, DashScopeConfig,
};
use serde_json::{json, Value};
use std::{collections::HashSet, sync::Arc, time::Duration};
use tokio_tungstenite::tungstenite::Message;

pub(crate) struct DashScopeDriver {
    config: DashScopeConfig,
    connection: WsConnection,
    task_id: String,
    ready: bool,
    done: bool,
    finishing: bool,
    index: u64,
    completed_sentences: HashSet<String>,
}

impl DashScopeDriver {
    pub(crate) fn new(config: DashScopeConfig, network: Arc<Network>) -> Self {
        let connection = WsConnection::new(network, config.timeouts.clone());
        Self {
            config,
            connection,
            task_id: uuid::Uuid::new_v4().to_string(),
            ready: false,
            done: false,
            finishing: false,
            index: 0,
            completed_sentences: HashSet::new(),
        }
    }

    fn receive_event(&mut self, sink: &ResultSink, timeout: Duration) -> Result<(), AsrError> {
        if let Some(value) = self.connection.receive_json(sink, timeout)? {
            self.event(value, sink)?;
        }
        Ok(())
    }

    /// Parses the sentence timestamps once so the dedup key and the committed
    /// seconds can never disagree about integer versus floating point values.
    fn sentence_times(sentence: &Value) -> (Option<f64>, Option<f64>) {
        (
            sentence["begin_time"].as_f64(),
            sentence["end_time"].as_f64(),
        )
    }

    /// Identity that a final sentence must not repeat within one task, built
    /// only from fields a re-delivered event necessarily reproduces and that
    /// two distinct sentences cannot both carry: the protocol's `sentence_id`
    /// sequence and a fully parseable millisecond interval. The key combines
    /// every identity field present, so a re-sent final is dropped while
    /// distinct sentences survive as long as any identity field differs.
    /// Without any identity field a re-delivery is indistinguishable from a
    /// legitimate repeat, so no key is produced and the final commits
    /// unconditionally — fingerprinting the text instead silently dropped
    /// consecutive identical sentences.
    fn completed_identity(
        sentence: &Value,
        begin: Option<f64>,
        end: Option<f64>,
    ) -> Option<String> {
        let id = sentence["sentence_id"].as_f64();
        match (id, begin, end) {
            (Some(id), Some(begin), Some(end)) => Some(format!("id:{id}:{begin}:{end}")),
            (Some(id), ..) => Some(format!("id:{id}")),
            (None, Some(begin), Some(end)) => Some(format!("{begin}:{end}")),
            _ => None,
        }
    }

    /// Returns whether this final sentence was seen for the first time.
    /// Sentences without any identity always count as first-seen.
    fn mark_completed(&mut self, identity: Option<String>) -> bool {
        match identity {
            Some(key) => self.completed_sentences.insert(key),
            None => true,
        }
    }

    /// Builds the `task-failed` error from the event header so quota and
    /// model failures stay diagnosable; the composition and field capping
    /// live in `websocket::server_error`.
    fn task_failed_error(value: &Value) -> AsrError {
        let header = &value["header"];
        server_error(
            "DashScope task failed",
            &header["error_code"],
            &header["error_message"],
        )
    }

    /// Returns whether an event belongs to this session. The server echoes the
    /// client-generated task id back on every event, so a present `task_id`
    /// that differs identifies an event misrouted onto this connection: it
    /// must neither confirm the handshake, mutate session state, nor surface
    /// results. Events without a task id cannot be attributed to another task
    /// and stay accepted.
    fn owns_event(&self, value: &Value) -> bool {
        value["header"]["task_id"]
            .as_str()
            .is_none_or(|event_task_id| event_task_id == self.task_id)
    }

    fn event(&mut self, value: Value, sink: &ResultSink) -> Result<(), AsrError> {
        if !self.owns_event(&value) {
            return Ok(());
        }
        match value["header"]["event"].as_str() {
            Some("task-started") => self.ready = true,
            Some("task-finished") => {
                if !self.finishing {
                    return Err(protocol("task finished before input closed"));
                }
                self.done = true;
            }
            Some("task-failed") => return Err(Self::task_failed_error(&value)),
            Some("result-generated") => {
                let sentence = &value["payload"]["output"]["sentence"];
                if sentence["heartbeat"].as_bool() == Some(true) {
                    return Ok(());
                }
                let Some(text) = sentence["text"].as_str() else {
                    return Ok(());
                };
                if sentence["sentence_end"].as_bool() == Some(true) {
                    let (begin, end) = Self::sentence_times(sentence);
                    if !self.mark_completed(Self::completed_identity(sentence, begin, end)) {
                        return Ok(());
                    }
                    sink.commit(
                        self.index,
                        self.index.to_string(),
                        text,
                        begin.map(|value| value / 1000.0),
                        end.map(|value| value / 1000.0),
                    )?;
                    self.index += 1;
                } else {
                    sink.partial(self.index.to_string(), text)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

impl Driver for DashScopeDriver {
    fn start(&mut self, sink: &ResultSink) -> Result<(), AsrError> {
        self.connection
            .connect(&self.config.endpoint, &self.config.api_key, &[], sink)?;
        self.connection.send_json(
            json!({
                "header": {
                    "action": "run-task",
                    "task_id": self.task_id,
                    "streaming": "duplex"
                },
                "payload": {
                    "task_group": "audio",
                    "task": "asr",
                    "function": "recognition",
                    "model": self.config.model,
                    "parameters": {"format": "pcm", "sample_rate": 16000, "heartbeat": true},
                    "input": {}
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
        self.connection
            .send(Message::Binary(audio::pcm_bytes(samples).into()), sink)
    }

    fn poll(&mut self, sink: &ResultSink) -> Result<(), AsrError> {
        poll_event(sink, |timeout| self.receive_event(sink, timeout))
    }

    fn poll_interval(&self) -> Option<Duration> {
        Some(POLL_INTERVAL)
    }

    fn finish(&mut self, sink: &ResultSink) -> Result<(), AsrError> {
        self.finishing = true;
        self.connection.send_json(
            json!({
                "header": {
                    "action": "finish-task",
                    "task_id": self.task_id,
                    "streaming": "duplex"
                },
                "payload": {"input": {}}
            }),
            sink,
        )?;
        let timeout = self.connection.response_timeout();
        wait_for_event(timeout, |remaining| match remaining {
            None => Ok(self.done),
            Some(remaining) => self.receive_event(sink, remaining).map(|()| self.done),
        })?;
        self.connection.close(sink);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
