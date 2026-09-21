use serde::{Deserialize, Serialize};
use std::fmt;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    InvalidInput,
    InvalidModel,
    UnsupportedCapability,
    Busy,
    WouldBlock,
    InputClosed,
    Cancelled,
    Timeout,
    AudioOverrun,
    Backend,
    Protocol,
    Http,
    Io,
    ResourceLimit,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AsrError {
    pub kind: ErrorKind,
    pub stage: String,
    pub message: String,
    pub http_status: Option<u16>,
    pub request_id: Option<String>,
}

impl AsrError {
    pub fn new(kind: ErrorKind, stage: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind,
            stage: stage.into(),
            message: message.into(),
            http_status: None,
            request_id: None,
        }
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidInput, "configuration", message)
    }

    pub(crate) fn backend(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Backend, "backend", message)
    }
}

impl fmt::Display for AsrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}/{}/{}", self.kind, self.stage, self.message)
    }
}

impl std::error::Error for AsrError {}
