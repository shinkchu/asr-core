use super::network::Network;
use crate::{
    session::driver::{Driver, ResultSink},
    *,
};
use futures::TryStreamExt;
use serde_json::Value;
use std::{sync::Arc, task::Poll, time::Instant};

/// Rejects input beyond the configured whole-request upload ceiling before
/// any encoding work; a limit that cannot fit one sample rejects everything.
fn ensure_within_upload_limit(samples: usize, c: &OpenAiHttpConfig) -> Result<(), AsrError> {
    if samples > audio::pcm16_wav_sample_capacity(c.max_upload_bytes).unwrap_or(0) {
        return Err(AsrError::new(
            ErrorKind::ResourceLimit,
            "upload",
            "audio exceeds configured upload limit",
        ));
    }
    Ok(())
}

fn endpoint(c: &OpenAiHttpConfig) -> Result<url::Url, AsrError> {
    let mut url = super::network::parse_clean_url(
        &c.api_root,
        &["http", "https"],
        false,
        "invalid HTTP API root",
        "API root must be an HTTP(S) URL without credentials, query or fragment",
    )?;
    if url
        .path()
        .trim_end_matches('/')
        .ends_with("/audio/transcriptions")
        || url.path().contains("/v1/v1")
    {
        return Err(AsrError::invalid(
            "provide the API root, without a transcription endpoint or repeated version",
        ));
    }
    url.set_path(&format!(
        "{}/audio/transcriptions",
        url.path().trim_end_matches('/')
    ));
    Ok(url)
}
pub(crate) fn validate(c: &OpenAiHttpConfig) -> Result<(), AsrError> {
    endpoint(c)?;
    c.validate_parameters()?;
    if let HttpMode::Utterances(vad) = &c.mode {
        if c.sample_rate != 16000 {
            return Err(AsrError::invalid("VAD utterance mode requires 16000 Hz"));
        }
        #[cfg(feature = "vad-silero")]
        // 与离线后端一致:构建并丢弃一个 detector,把 VAD 模型损坏在
        // prepare 阶段暴露;session 启动时仍按 config 逐个创建实例。
        super::vad::preflight(vad)?;
        #[cfg(not(feature = "vad-silero"))]
        {
            let _ = vad;
            return Err(AsrError::new(
                ErrorKind::UnsupportedCapability,
                "configuration",
                "utterance mode requires vad-silero",
            ));
        }
    }
    Ok(())
}
pub(crate) struct HttpDriver {
    config: OpenAiHttpConfig,
    network: Arc<Network>,
    samples: Vec<f32>,
    index: u64,
    #[cfg(feature = "vad-silero")]
    vad: Option<sherpa_onnx::VoiceActivityDetector>,
}
impl HttpDriver {
    pub fn new(config: OpenAiHttpConfig, network: Arc<Network>) -> Self {
        Self {
            config,
            network,
            samples: Vec::new(),
            index: 0,
            #[cfg(feature = "vad-silero")]
            vad: None,
        }
    }
    fn transcribe(
        &mut self,
        samples: &[f32],
        start: Option<f64>,
        control: &ResultSink,
    ) -> Result<(), AsrError> {
        if samples.is_empty() {
            return Ok(());
        }
        let c = &self.config;
        ensure_within_upload_limit(samples.len(), c)?;
        let wav = audio::encode_wav_pcm16(samples, c.sample_rate)?;
        let part = reqwest::multipart::Part::bytes(wav)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .map_err(|_| AsrError::invalid("invalid WAV MIME type"))?;
        let mut form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("model", c.model.clone())
            .text(
                "response_format",
                if c.response == HttpResponse::Text {
                    "text"
                } else {
                    "json"
                },
            );
        if c.response == HttpResponse::Sse {
            form = form.text("stream", "true");
        }
        if let Some(v) = &c.language {
            form = form.text("language", v.clone());
        }
        if let Some(v) = &c.prompt {
            form = form.text("prompt", v.clone());
        }
        let mut request = self.network.client.post(endpoint(c)?);
        if !c.api_key.0.is_empty() {
            request = request.bearer_auth(&c.api_key.0);
        }
        let id = self.index.to_string();
        let text = self
            .network
            .run(control, "transcription", c.timeouts.response, async {
                let mut response = send_request(request, form, &c.timeouts).await?;
                let status = response.status();
                let request_id = response
                    .headers()
                    .get("x-request-id")
                    .and_then(|h| h.to_str().ok())
                    .map(str::to_owned);
                if !status.is_success() {
                    let mut error = AsrError::new(
                        ErrorKind::Http,
                        "response",
                        format!("provider returned HTTP {}", status.as_u16()),
                    );
                    error.http_status = Some(status.as_u16());
                    error.request_id = request_id;
                    return Err(error);
                }
                let mut received = 0usize;
                let mut body = Vec::new();
                let mut sse = SseParser::default();
                while let Some(chunk) = response.chunk().await.map_err(response_error)? {
                    received = received.saturating_add(chunk.len());
                    if received > c.max_response_bytes {
                        return Err(AsrError::new(
                            ErrorKind::ResourceLimit,
                            "response",
                            "response exceeds configured byte limit",
                        ));
                    }
                    if c.response == HttpResponse::Sse {
                        let progress = sse.push(&chunk)?;
                        if let Some(text) = progress.final_text {
                            return Ok(text);
                        }
                        if progress.partial_changed {
                            control.partial(&id, sse.partial())?;
                        }
                    } else {
                        body.extend_from_slice(&chunk);
                    }
                }
                match c.response {
                    HttpResponse::Text => {
                        String::from_utf8(body).map_err(|_| protocol("response is not UTF-8"))
                    }
                    HttpResponse::Json => {
                        let body = std::str::from_utf8(&body)
                            .map_err(|_| protocol("response is not UTF-8"))?;
                        let value: Value = serde_json::from_str(body)
                            .map_err(|_| protocol("invalid JSON response"))?;
                        value["text"]
                            .as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| protocol("response missing text"))
                    }
                    HttpResponse::Sse => {
                        let progress = sse.finish()?;
                        if progress.partial_changed {
                            control.partial(&id, sse.partial())?;
                        }
                        progress.final_text.ok_or_else(|| {
                            protocol("SSE stream ended without transcript.text.done")
                        })
                    }
                }
            })?;
        control.commit(
            self.index,
            id,
            text,
            start,
            start.map(|s| s + samples.len() as f64 / c.sample_rate as f64),
        )?;
        self.index += 1;
        Ok(())
    }
    #[cfg(feature = "vad-silero")]
    fn drain(&mut self, c: &ResultSink) -> Result<(), AsrError> {
        while let Some(segment) = self.vad.as_ref().unwrap().front() {
            let samples = segment.samples().to_vec();
            let start = segment.start() as f64 / 16000.0;
            self.vad.as_ref().unwrap().pop();
            self.transcribe(&samples, Some(start), c)?;
        }
        Ok(())
    }
}
#[derive(Clone, Copy)]
enum UploadProgress {
    Connecting,
    Sending(Option<Instant>),
    Finished,
}

async fn send_request(
    request: reqwest::RequestBuilder,
    form: reqwest::multipart::Form,
    timeouts: &Timeouts,
) -> Result<reqwest::Response, AsrError> {
    let content_type = format!("multipart/form-data; boundary={}", form.boundary());
    // All parts are already in memory. Retain their Bytes without copying the
    // WAV, and preserve Content-Length for providers that reject chunked uploads.
    let parts: Vec<_> = form
        .into_stream()
        .try_collect()
        .await
        .map_err(request_error)?;
    let length: usize = parts.iter().map(|part| part.len()).sum();
    // Small frames keep transport backpressure visible; yielding the entire WAV
    // as one frame would report completion before a stalled upload can be timed.
    const UPLOAD_FRAME: usize = 16 * 1024;
    let mut chunks = parts.into_iter().flat_map(|part| {
        (0..part.len())
            .step_by(UPLOAD_FRAME)
            .map(move |start| part.slice(start..(start + UPLOAD_FRAME).min(part.len())))
    });
    let (progress_tx, mut progress_rx) = tokio::sync::watch::channel(UploadProgress::Connecting);
    let mut remaining = length;
    let send_timeout = timeouts.send;
    let body = reqwest::Body::wrap_stream(futures::stream::poll_fn(move |_| {
        if remaining == length {
            progress_tx.send_replace(UploadProgress::Sending(
                Instant::now().checked_add(send_timeout),
            ));
        }
        let next = chunks.next();
        if let Some(chunk) = &next {
            remaining -= chunk.len();
        }
        if remaining == 0 {
            progress_tx.send_replace(UploadProgress::Finished);
        }
        Poll::Ready(next.map(Ok::<_, std::convert::Infallible>))
    }));
    let sending = request
        .header(reqwest::header::CONTENT_TYPE, content_type)
        .header(reqwest::header::CONTENT_LENGTH, length)
        .body(body)
        .send();
    tokio::pin!(sending);
    let connect_deadline = crate::deadline::after(timeouts.connect, "HTTP connect")?;
    loop {
        let progress = *progress_rx.borrow_and_update();
        let (deadline, stage) = match progress {
            UploadProgress::Connecting => (connect_deadline, "HTTP connect"),
            UploadProgress::Sending(Some(deadline)) => (deadline, "HTTP upload"),
            UploadProgress::Sending(None) => {
                return Err(AsrError::invalid("HTTP upload timeout is too large"));
            }
            // reqwest's response future also waits for headers. Once the body
            // has been handed to the transport, only the enclosing response
            // and session deadlines apply to server-side recognition.
            UploadProgress::Finished => return sending.await.map_err(request_error),
        };
        tokio::select! {
            result = &mut sending => return result.map_err(request_error),
            changed = progress_rx.changed() => {
                if changed.is_err() {
                    // An early response (e.g. 401) may discard the request body.
                    return sending.await.map_err(request_error);
                }
            },
            _ = tokio::time::sleep_until(deadline.into()) => {
                return Err(crate::deadline::exceeded(stage));
            },
        }
    }
}
fn request_error(error: reqwest::Error) -> AsrError {
    http_error(error, "request")
}
fn response_error(error: reqwest::Error) -> AsrError {
    http_error(error, "response")
}
fn http_error(error: reqwest::Error, stage: &str) -> AsrError {
    let (kind, message) = if error.is_timeout() {
        (ErrorKind::Timeout, "HTTP operation timed out")
    } else if error.is_connect() {
        (ErrorKind::Http, "HTTP connection failed")
    } else if error.is_body() {
        (ErrorKind::Http, "HTTP body transfer failed")
    } else if error.is_decode() {
        (ErrorKind::Http, "HTTP response decoding failed")
    } else if error.is_request() {
        (ErrorKind::Http, "HTTP request could not be sent")
    } else {
        (ErrorKind::Http, "HTTP operation failed")
    };
    AsrError::new(kind, stage, message)
}
fn protocol(message: &str) -> AsrError {
    AsrError::new(ErrorKind::Protocol, "response", message)
}

#[derive(Default)]
struct SseParser {
    pending: Vec<u8>,
    scan: usize,
    line_start: usize,
    partial: String,
    done: bool,
}
#[derive(Default)]
struct SseProgress {
    partial_changed: bool,
    final_text: Option<String>,
}
impl SseParser {
    fn partial(&self) -> &str {
        &self.partial
    }
    fn push(&mut self, chunk: &[u8]) -> Result<SseProgress, AsrError> {
        if self.done {
            return Ok(SseProgress::default());
        }
        self.pending.extend_from_slice(chunk);
        let mut progress = SseProgress::default();
        let mut event_start = 0;
        while self.scan < self.pending.len() {
            if !matches!(self.pending[self.scan], b'\r' | b'\n') {
                self.scan += 1;
                continue;
            }
            let newline_start = self.scan;
            if self.pending[self.scan] == b'\r' {
                if self.scan + 1 == self.pending.len() {
                    // A CR at a chunk boundary may be the first half of CRLF.
                    break;
                }
                self.scan += usize::from(self.pending[self.scan + 1] == b'\n') + 1;
            } else {
                self.scan += 1;
            }
            if newline_start != self.line_start {
                self.line_start = self.scan;
                continue;
            }

            let event = self.pending[event_start..self.line_start].to_vec();
            event_start = self.scan;
            self.line_start = self.scan;
            if let Some(text) = self.event(&event, &mut progress.partial_changed)? {
                self.done = true;
                progress.final_text = Some(text);
                self.pending.clear();
                self.scan = 0;
                self.line_start = 0;
                return Ok(progress);
            }
        }
        if event_start > 0 {
            self.pending.drain(..event_start);
            self.scan -= event_start;
            self.line_start -= event_start;
        }
        if std::str::from_utf8(&self.pending).is_err_and(|error| error.error_len().is_some()) {
            return Err(protocol("response is not UTF-8"));
        }
        Ok(progress)
    }
    fn finish(&mut self) -> Result<SseProgress, AsrError> {
        if self.done {
            return Ok(SseProgress::default());
        }
        let mut progress = SseProgress::default();
        if !self.pending.is_empty() {
            let event = std::mem::take(&mut self.pending);
            self.scan = 0;
            self.line_start = 0;
            if let Some(text) = self.event(&event, &mut progress.partial_changed)? {
                self.done = true;
                progress.final_text = Some(text);
            }
        }
        Ok(progress)
    }
    fn event(
        &mut self,
        bytes: &[u8],
        partial_changed: &mut bool,
    ) -> Result<Option<String>, AsrError> {
        let event = std::str::from_utf8(bytes).map_err(|_| protocol("response is not UTF-8"))?;
        let normalized = event.replace("\r\n", "\n").replace('\r', "\n");
        let mut data = Vec::new();
        for line in normalized.lines() {
            if line.starts_with(':') {
                continue;
            }
            let (field, value) = line.split_once(':').unwrap_or((line, ""));
            if field == "data" {
                data.push(value.strip_prefix(' ').unwrap_or(value));
            }
        }
        let data = data.join("\n");
        if data.is_empty() || data == "[DONE]" {
            return Ok(None);
        }
        let value: Value = serde_json::from_str(&data).map_err(|_| protocol("invalid SSE JSON"))?;
        match value["type"].as_str() {
            Some("transcript.text.delta") => {
                self.partial.push_str(
                    value["delta"]
                        .as_str()
                        .ok_or_else(|| protocol("delta missing text"))?,
                );
                *partial_changed = true;
                Ok(None)
            }
            Some("transcript.text.done") => {
                let text = value["text"]
                    .as_str()
                    .ok_or_else(|| protocol("final event missing text"))?;
                *partial_changed |= self.partial != text;
                text.clone_into(&mut self.partial);
                Ok(Some(text.to_owned()))
            }
            Some("error") => Err(protocol("provider reported an SSE error")),
            _ => Ok(None),
        }
    }
}
#[cfg(test)]
fn parse_sse(body: &str) -> Result<(String, Option<String>), AsrError> {
    let mut parser = SseParser::default();
    let progress = parser.push(body.as_bytes())?;
    Ok((parser.partial, progress.final_text))
}
impl Driver for HttpDriver {
    fn start(&mut self, c: &ResultSink) -> Result<(), AsrError> {
        c.check()?;
        #[cfg(feature = "vad-silero")]
        if let HttpMode::Utterances(config) = &self.config.mode {
            self.vad = Some(super::vad::create(config)?);
        }
        Ok(())
    }
    fn push(&mut self, samples: &[f32], c: &ResultSink) -> Result<(), AsrError> {
        c.check()?;
        #[cfg(feature = "vad-silero")]
        if self.vad.is_some() {
            for chunk in samples.chunks(512) {
                self.vad.as_ref().unwrap().accept_waveform(chunk);
                self.drain(c)?;
            }
            return Ok(());
        }
        ensure_within_upload_limit(
            self.samples.len().saturating_add(samples.len()),
            &self.config,
        )?;
        self.samples.extend_from_slice(samples);
        Ok(())
    }
    fn finish(&mut self, c: &ResultSink) -> Result<(), AsrError> {
        #[cfg(feature = "vad-silero")]
        if self.vad.is_some() {
            self.vad.as_ref().unwrap().flush();
            return self.drain(c);
        }
        let samples = std::mem::take(&mut self.samples);
        self.transcribe(&samples, None, c)
    }
}

#[cfg(test)]
mod tests;
