//! Backend configuration and validation contracts.

#[cfg(feature = "backend-openai-http")]
use crate::audio;
use crate::audio::AudioSpec;
#[cfg(any(
    feature = "backend-dashscope",
    feature = "backend-openai-http",
    feature = "backend-openai-realtime"
))]
use crate::AsrError;
use serde::{Deserialize, Serialize};
use std::{fmt, path::PathBuf, time::Duration};
use zeroize::Zeroize;

/// API key wrapper: redacted by `Debug`, skipped by serde, and zeroized
/// in memory when dropped.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct Secret(pub(crate) String);

impl Secret {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}

fn default_connect_timeout() -> Duration {
    Duration::from_secs(10)
}

fn default_send_timeout() -> Duration {
    Duration::from_secs(5)
}

fn default_response_timeout() -> Duration {
    Duration::from_secs(30)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Timeouts {
    #[serde(default = "default_connect_timeout")]
    pub connect: Duration,
    #[serde(default = "default_send_timeout")]
    pub send: Duration,
    #[serde(default = "default_response_timeout")]
    pub response: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            connect: default_connect_timeout(),
            send: default_send_timeout(),
            response: default_response_timeout(),
        }
    }
}

fn default_vad_threshold() -> f32 {
    0.5
}

fn default_vad_min_silence() -> f32 {
    0.5
}

fn default_vad_min_speech() -> f32 {
    0.25
}

fn default_vad_max_speech() -> f32 {
    15.0
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VadConfig {
    pub model: PathBuf,
    #[serde(default = "default_vad_threshold")]
    pub threshold: f32,
    #[serde(default = "default_vad_min_silence")]
    pub min_silence: f32,
    #[serde(default = "default_vad_min_speech")]
    pub min_speech: f32,
    #[serde(default = "default_vad_max_speech")]
    pub max_speech: f32,
}

impl VadConfig {
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            threshold: default_vad_threshold(),
            min_silence: default_vad_min_silence(),
            min_speech: default_vad_min_speech(),
            max_speech: default_vad_max_speech(),
        }
    }
}

/// 标点恢复模型目录（解压后的 sherpa-onnx punctuation 归档，
/// 如 `sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8/`）。
/// 目录内容决定模型家族：含 `bpe.vocab` 为英文 CNN-BiLSTM，否则为中英 CT-Transformer。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PunctConfig {
    pub model: PathBuf,
}

impl PunctConfig {
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OfflineFamily {
    SenseVoice,
    Paraformer,
    /// 离线 transducer（encoder/decoder/joiner + tokens.txt），支持热词。
    Transducer,
    /// Qwen3-ASR（LLM 型，中英 + 多语言），支持引擎级热词。
    Qwen3Asr,
    /// FunASR-Nano（Fun-ASR-Nano-2512，LLM 型，中英日 + 中文方言），
    /// 支持引擎级热词。
    FunAsrNano,
    /// FireRedASR-AED（attention-encoder-decoder，encoder + decoder +
    /// tokens.txt，中英）。兼容 FireRedASR 1.0-AED-L 与 FireRedASR2 的
    /// sherpa-onnx AED 导出。不支持热词与语言覆盖。
    FireRedAsrAed,
    /// FireRedASR-CTC（单 model.onnx + tokens.txt，中英；FireRedASR2 官方
    /// 导出即此形态）。不支持热词与语言覆盖。
    FireRedAsrCtc,
}

/// Backend-neutral speech phrases. A backend either accepts the complete set
/// for a session or rejects it as an unsupported capability.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SpeechHints {
    pub phrases: Vec<String>,
}

impl SpeechHints {
    pub fn new(phrases: Vec<String>) -> Self {
        Self { phrases }
    }
}

#[derive(Debug, Clone)]
pub struct SessionOptions {
    pub input: AudioSpec,
    pub queue_duration: Duration,
    pub max_chunk_duration: Duration,
    pub max_duration: Duration,
    pub event_capacity: usize,
    pub max_transcript_bytes: usize,
    /// Backend-neutral phrases for backends that support per-session hints.
    pub hints: Option<SpeechHints>,
}

impl SessionOptions {
    pub fn new(input: AudioSpec) -> Self {
        Self {
            input,
            ..Self::default()
        }
    }
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            input: AudioSpec {
                sample_rate: 16_000,
            },
            queue_duration: Duration::from_secs(2),
            max_chunk_duration: Duration::from_secs(1),
            max_duration: Duration::from_secs(1200),
            event_capacity: 256,
            max_transcript_bytes: 8 * 1024 * 1024,
            hints: None,
        }
    }
}

/// A default phrase carried by a transducer bias configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BiasPhrase {
    pub phrase: String,
    pub score: Option<f32>,
}

impl BiasPhrase {
    pub fn new(phrase: impl Into<String>) -> Self {
        Self {
            phrase: phrase.into(),
            score: None,
        }
    }

    pub fn scored(phrase: impl Into<String>, score: f32) -> Self {
        Self {
            phrase: phrase.into(),
            score: Some(score),
        }
    }
}

fn default_bias_score() -> f32 {
    2.0
}

/// Default recognizer ONNX inference threads: conservative so several
/// concurrent sessions stay within a typical core count. Hosts exposing
/// their own thread knob can anchor on this value to keep their default
/// in sync with [`StreamingConfig`] / [`OfflineConfig`].
pub const DEFAULT_NUM_THREADS: usize = 2;

/// Upper bound for recognizer `num_threads`, enforced by
/// [`crate::utils::precheck::validate`] and `Engine::prepare`; also keeps
/// the `usize -> i32` handoff to sherpa-onnx overflow-free.
pub const MAX_NUM_THREADS: usize = 256;

fn default_num_threads() -> usize {
    DEFAULT_NUM_THREADS
}

/// Requested execution provider for local recognition. The native sherpa-onnx
/// and ONNX Runtime libraries must support the selected provider. Upstream may
/// fall back to CPU; this value is not a report of actual device utilization.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    /// NVIDIA CUDA on Linux or Windows; requires a CUDA-enabled native build.
    Cuda,
    /// Apple CoreML; may schedule work on CPU, GPU, or Neural Engine.
    CoreMl,
}

impl ExecutionProvider {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::CoreMl => "coreml",
        }
    }
}

impl std::str::FromStr for ExecutionProvider {
    type Err = crate::AsrError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "coreml" => Ok(Self::CoreMl),
            _ => Err(crate::AsrError::invalid(
                "provider must be cpu, cuda, or coreml",
            )),
        }
    }
}

/// Transducer-only decoding bias. Configuring it selects modified beam search
/// and enables per-session [`SpeechHints`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TransducerBiasConfig {
    pub phrases: Vec<BiasPhrase>,
    #[serde(default = "default_bias_score")]
    pub default_score: f32,
    pub modeling_unit: Option<String>,
}

impl TransducerBiasConfig {
    pub fn new(phrases: Vec<BiasPhrase>) -> Self {
        Self {
            phrases,
            default_score: default_bias_score(),
            modeling_unit: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StreamingConfig {
    pub model_dir: PathBuf,
    /// Recognizer execution provider; VAD and punctuation use CPU.
    #[serde(default)]
    pub provider: ExecutionProvider,
    pub punctuation: Option<PunctConfig>,
    /// Transducer decoding bias; also enables per-session speech hints.
    pub bias: Option<TransducerBiasConfig>,
    /// Recognizer ONNX inference threads; VAD and punctuation models keep
    /// their own fixed settings.
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,
}

impl StreamingConfig {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            provider: ExecutionProvider::default(),
            punctuation: None,
            bias: None,
            num_threads: default_num_threads(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OfflineConfig {
    pub model_dir: PathBuf,
    /// Recognizer execution provider; VAD and punctuation use CPU.
    #[serde(default)]
    pub provider: ExecutionProvider,
    pub family: OfflineFamily,
    pub language: Option<String>,
    pub vad: VadConfig,
    pub punctuation: Option<PunctConfig>,
    /// Transducer-only decoding bias; also enables per-session speech hints.
    pub transducer_bias: Option<TransducerBiasConfig>,
    /// Engine-level prompt phrases for Qwen3-ASR and FunASR-Nano.
    pub prompt_hints: Option<SpeechHints>,
    /// Recognizer ONNX inference threads; VAD and punctuation models keep
    /// their own fixed settings.
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,
}

impl OfflineConfig {
    pub fn new(model_dir: impl Into<PathBuf>, family: OfflineFamily, vad: VadConfig) -> Self {
        Self {
            model_dir: model_dir.into(),
            provider: ExecutionProvider::default(),
            family,
            language: None,
            vad,
            punctuation: None,
            transducer_bias: None,
            prompt_hints: None,
            num_threads: default_num_threads(),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum HttpMode {
    #[default]
    WholeRecording,
    Utterances(VadConfig),
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum HttpResponse {
    #[default]
    Json,
    Text,
    Sse,
}

fn default_http_sample_rate() -> u32 {
    16_000
}

fn default_max_upload_bytes() -> usize {
    24 * 1024 * 1024
}

fn default_max_response_bytes() -> usize {
    1024 * 1024
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenAiHttpConfig {
    pub api_root: String,
    pub model: String,
    #[serde(skip)]
    pub api_key: Secret,
    #[serde(default = "default_http_sample_rate")]
    pub sample_rate: u32,
    #[serde(default)]
    pub mode: HttpMode,
    #[serde(default)]
    pub response: HttpResponse,
    pub language: Option<String>,
    pub prompt: Option<String>,
    #[serde(default = "default_max_upload_bytes")]
    pub max_upload_bytes: usize,
    #[serde(default = "default_max_response_bytes")]
    pub max_response_bytes: usize,
    #[serde(default)]
    pub timeouts: Timeouts,
}

impl OpenAiHttpConfig {
    pub fn new(api_root: impl Into<String>, model: impl Into<String>, api_key: Secret) -> Self {
        Self {
            api_root: api_root.into(),
            model: model.into(),
            api_key,
            sample_rate: default_http_sample_rate(),
            mode: HttpMode::default(),
            response: HttpResponse::default(),
            language: None,
            prompt: None,
            max_upload_bytes: default_max_upload_bytes(),
            max_response_bytes: default_max_response_bytes(),
            timeouts: Timeouts::default(),
        }
    }

    #[cfg(feature = "backend-openai-http")]
    pub(crate) fn validate_parameters(&self) -> Result<(), AsrError> {
        validate_cloud_parameters(&self.model, &self.timeouts, &self.api_key, "OpenAI HTTP")?;
        AudioSpec::mono(self.sample_rate)?;
        if audio::pcm16_wav_sample_capacity(self.max_upload_bytes)
            .is_none_or(|capacity| capacity == 0)
            || self.max_response_bytes == 0
        {
            return Err(AsrError::invalid(
                "upload and response limits must be positive",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DashScopeConfig {
    pub endpoint: String,
    pub model: String,
    #[serde(skip)]
    pub api_key: Secret,
    #[serde(default)]
    pub timeouts: Timeouts,
}

impl DashScopeConfig {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>, api_key: Secret) -> Self {
        Self {
            endpoint: endpoint.into(),
            model: model.into(),
            api_key,
            timeouts: Timeouts::default(),
        }
    }

    #[cfg(feature = "backend-dashscope")]
    pub(crate) fn validate_parameters(&self) -> Result<(), AsrError> {
        validate_cloud_parameters(&self.model, &self.timeouts, &self.api_key, "DashScope")
    }
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenAiRealtimeConfig {
    pub endpoint: String,
    pub model: String,
    #[serde(skip)]
    pub api_key: Secret,
    #[serde(default = "default_true")]
    pub server_vad: bool,
    #[serde(default)]
    pub timeouts: Timeouts,
}

impl OpenAiRealtimeConfig {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>, api_key: Secret) -> Self {
        Self {
            endpoint: endpoint.into(),
            model: model.into(),
            api_key,
            server_vad: true,
            timeouts: Timeouts::default(),
        }
    }

    #[cfg(feature = "backend-openai-realtime")]
    pub(crate) fn validate_parameters(&self) -> Result<(), AsrError> {
        validate_cloud_parameters(
            &self.model,
            &self.timeouts,
            &self.api_key,
            "OpenAI Realtime",
        )
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EngineConfig {
    Streaming(StreamingConfig),
    Offline(OfflineConfig),
    DashScope(DashScopeConfig),
    OpenAiHttp(OpenAiHttpConfig),
    OpenAiRealtime(OpenAiRealtimeConfig),
}

#[cfg(any(
    feature = "backend-dashscope",
    feature = "backend-openai-http",
    feature = "backend-openai-realtime"
))]
pub(crate) fn validate_model(model: &str) -> Result<(), AsrError> {
    if model.trim().is_empty() {
        Err(AsrError::invalid("model is required"))
    } else {
        Ok(())
    }
}

#[cfg(any(
    feature = "backend-dashscope",
    feature = "backend-openai-http",
    feature = "backend-openai-realtime"
))]
pub(crate) fn validate_timeouts(timeouts: &Timeouts) -> Result<(), AsrError> {
    for (name, timeout) in [
        ("connect", timeouts.connect),
        ("send", timeouts.send),
        ("response", timeouts.response),
    ] {
        if timeout.is_zero() {
            return Err(AsrError::invalid(format!(
                "{name} timeout must be positive"
            )));
        }
        if timeout > crate::deadline::MAX_NETWORK_TIMEOUT {
            return Err(AsrError::invalid(format!(
                "{name} timeout must not exceed 24 hours"
            )));
        }
        crate::deadline::after(timeout, name)?;
    }
    Ok(())
}

/// The parameter checks every cloud backend shares: a usable model id,
/// sane timeouts and a non-empty API key, with `service` naming the
/// backend in the API-key error.
#[cfg(any(
    feature = "backend-dashscope",
    feature = "backend-openai-http",
    feature = "backend-openai-realtime"
))]
fn validate_cloud_parameters(
    model: &str,
    timeouts: &Timeouts,
    api_key: &Secret,
    service: &str,
) -> Result<(), AsrError> {
    validate_model(model)?;
    validate_timeouts(timeouts)?;
    if api_key.0.trim().is_empty() {
        return Err(AsrError::invalid(format!("{service} requires an API key")));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
