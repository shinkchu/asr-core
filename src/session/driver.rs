#![cfg_attr(
    not(any(
        feature = "backend-sherpa",
        feature = "backend-dashscope",
        feature = "backend-openai-http",
        feature = "backend-openai-realtime"
    )),
    allow(dead_code)
)]

use super::results::CommitPreflight;
use crate::{coordinator::Control, AsrError};
use std::{sync::Arc, time::Duration};

/// Best-effort final-text rewrite. The backend implementation returns the
/// original text when inference fails.
pub(crate) type FinalTextProcessor = Arc<dyn Fn(&str) -> String + Send + Sync>;

#[derive(Clone)]
pub(crate) struct ResultSink {
    control: Control,
    final_text_processor: Option<FinalTextProcessor>,
}

impl ResultSink {
    pub(crate) fn new(control: Control, final_text_processor: Option<FinalTextProcessor>) -> Self {
        Self {
            control,
            final_text_processor,
        }
    }

    fn process_final(&self, text: String) -> String {
        if text.trim().is_empty() {
            return text;
        }
        match &self.final_text_processor {
            Some(processor) => processor(&text),
            None => text,
        }
    }

    pub(crate) fn check(&self) -> Result<(), AsrError> {
        self.control.check()
    }

    #[cfg_attr(
        not(any(
            feature = "backend-sherpa",
            feature = "backend-dashscope",
            feature = "backend-openai-http"
        )),
        allow(dead_code)
    )]
    pub(crate) fn partial(
        &self,
        id: impl Into<String>,
        text: impl Into<String>,
    ) -> Result<(), AsrError> {
        self.control.partial(id, text)
    }

    /// Appends a protocol delta directly to the authoritative result store.
    #[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
    pub(crate) fn append_delta(&self, id: impl Into<String>, delta: &str) -> Result<(), AsrError> {
        self.control.append_delta(id, delta)
    }

    /// Registers protocol metadata before a backend retains the ID.
    #[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
    pub(crate) fn register_id(&self, id: &str) -> Result<(), AsrError> {
        self.control.register_id(id)
    }

    #[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
    pub(crate) fn complete_unindexed(
        &self,
        id: impl Into<String>,
        text: impl Into<String>,
    ) -> Result<(), AsrError> {
        let id = id.into();
        if !self.control.preflight_unindexed(&id)? {
            return Ok(());
        }
        self.control
            .complete_unindexed(id, self.process_final(text.into()))
    }

    #[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
    pub(crate) fn index_unindexed(
        &self,
        index: u64,
        id: &str,
        start: Option<f64>,
        end: Option<f64>,
    ) -> Result<(), AsrError> {
        self.control.index_unindexed(index, id, start, end)
    }

    pub(crate) fn commit(
        &self,
        index: u64,
        id: impl Into<String>,
        text: impl Into<String>,
        start: Option<f64>,
        end: Option<f64>,
    ) -> Result<(), AsrError> {
        let id = id.into();
        match self.control.preflight_commit(index, &id)? {
            CommitPreflight::Duplicate => Ok(()),
            CommitPreflight::IndexUnindexed => {
                self.control.commit(index, id, String::new(), start, end)
            }
            CommitPreflight::Process => {
                self.control
                    .commit(index, id, self.process_final(text.into()), start, end)
            }
        }
    }

    #[cfg(any(
        feature = "backend-dashscope",
        feature = "backend-openai-http",
        feature = "backend-openai-realtime"
    ))]
    pub(crate) fn deadline(&self) -> Option<std::time::Instant> {
        self.control.deadline()
    }

    #[cfg(any(
        feature = "backend-dashscope",
        feature = "backend-openai-http",
        feature = "backend-openai-realtime"
    ))]
    pub(crate) async fn notified(&self) {
        self.control.0.wake.notified().await;
    }
}

pub(crate) trait Driver: Send {
    fn start(&mut self, sink: &ResultSink) -> Result<(), AsrError> {
        sink.check()
    }

    fn push(&mut self, samples: &[f32], sink: &ResultSink) -> Result<(), AsrError>;

    fn poll(&mut self, sink: &ResultSink) -> Result<(), AsrError> {
        sink.check()
    }

    fn poll_interval(&self) -> Option<Duration> {
        None
    }

    fn finish(&mut self, sink: &ResultSink) -> Result<(), AsrError>;
}
