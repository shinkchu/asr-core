use super::{
    coordinator::{self, Session},
    session::{
        driver::{Driver, FinalTextProcessor},
        state::failure_result,
    },
};
#[cfg(feature = "backend-dashscope")]
use crate::DashScopeConfig;
#[cfg(feature = "backend-openai-realtime")]
use crate::OpenAiRealtimeConfig;
#[cfg(feature = "backend-sherpa")]
use crate::PunctConfig;
use crate::{
    AsrError, AudioBuffer, AudioChunk, AudioSpec, BackendCapabilities, EngineConfig, ErrorKind,
    SessionOptions, SessionOutcome, SessionResult,
};
#[cfg(feature = "backend-openai-http")]
use crate::{HttpMode, HttpResponse, OpenAiHttpConfig};
#[cfg(all(feature = "backend-sherpa", feature = "vad-silero"))]
use crate::{OfflineFamily, VadConfig};
#[cfg(any(
    feature = "backend-dashscope",
    feature = "backend-openai-http",
    feature = "backend-openai-realtime"
))]
use std::sync::Mutex;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::{num::NonZeroUsize, time::Instant};

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub max_active_sessions: NonZeroUsize,
}
impl EngineOptions {
    pub fn new(max_active_sessions: usize) -> Result<Self, AsrError> {
        let max_active_sessions = NonZeroUsize::new(max_active_sessions)
            .ok_or_else(|| AsrError::invalid("max_active_sessions must be positive"))?;
        Ok(Self {
            max_active_sessions,
        })
    }
}
impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            max_active_sessions: NonZeroUsize::new(8).unwrap(),
        }
    }
}

fn load(
    config: &EngineConfig,
) -> Result<(Kind, BackendCapabilities, Option<FinalTextProcessor>), AsrError> {
    // Cloud parameter validation lives in backends::validate_cloud_config so
    // utils::precheck::validate runs the exact same checks; local variants
    // are a no-op there and validate inside their own arms below.
    super::backends::validate_cloud_config(config)?;
    #[allow(unused_variables)]
    match config {
        #[cfg(feature = "backend-sherpa")]
        EngineConfig::Streaming(config) => {
            // Cheap parameter/layout checks before native initialization;
            // utils::precheck delegates to the same functions.
            super::backends::precheck::streaming(config)?;
            let (recognizer, words, vocabulary) =
                super::backends::local::load_stream(&config.model_dir, config.bias.as_ref())?;
            let punctuator = load_punctuator(&config.punctuation)?;
            let supports_session_hints = config.bias.is_some();
            Ok((
                Kind::Streaming(recognizer, words, vocabulary),
                capabilities(
                    "zipformer",
                    16000,
                    true,
                    true,
                    false,
                    punctuator.is_some(),
                    supports_session_hints,
                ),
                punctuator,
            ))
        }
        #[cfg(all(feature = "backend-sherpa", feature = "vad-silero"))]
        EngineConfig::Offline(config) => {
            // Cheap parameter/layout checks (bias, VAD parameters, family
            // capability rules, model layout) before native initialization.
            super::backends::precheck::offline(config)?;
            // VAD 模型损坏属于一次性配置错误:在 prepare 构建并丢弃一个
            // detector 提前暴露,而不是推迟到每个 session 的 start 逐次失败。
            // (内含参数校验,与原先的 vad::validate 同位。)
            super::backends::vad::preflight(&config.vad)?;
            let (recognizer, words, vocabulary) = super::backends::local::load_offline(
                &config.model_dir,
                config.family,
                config.language.as_deref(),
                config.transducer_bias.as_ref(),
                config.prompt_hints.as_ref(),
            )?;
            let punctuator = load_punctuator(&config.punctuation)?;
            let supports_session_hints =
                config.transducer_bias.is_some() && config.family == OfflineFamily::Transducer;
            Ok((
                Kind::Offline(recognizer, config.vad.clone(), words, vocabulary),
                capabilities(
                    "offline",
                    16000,
                    false,
                    false,
                    false,
                    punctuator.is_some(),
                    supports_session_hints,
                ),
                punctuator,
            ))
        }
        #[cfg(feature = "backend-dashscope")]
        EngineConfig::DashScope(cfg) => Ok((
            Kind::DashScope(cfg.clone()),
            capabilities("dashscope", 16000, true, true, true, false, false),
            None,
        )),
        #[cfg(feature = "backend-openai-http")]
        EngineConfig::OpenAiHttp(cfg) => Ok((
            Kind::Http(cfg.clone()),
            capabilities(
                "openai-transcriptions",
                cfg.sample_rate,
                false,
                cfg.response == HttpResponse::Sse,
                true,
                false,
                false,
            ),
            None,
        )),
        #[cfg(feature = "backend-openai-realtime")]
        EngineConfig::OpenAiRealtime(cfg) => Ok((
            Kind::Realtime(cfg.clone()),
            capabilities("openai-realtime", 24000, true, true, true, false, false),
            None,
        )),
        #[allow(unreachable_patterns)]
        _ => Err(AsrError::new(
            ErrorKind::UnsupportedCapability,
            "configuration",
            "requested backend is disabled in this build",
        )),
    }
}

/// 标点模型只在本地后端的 match 臂中加载，因此仅在 backend-sherpa 启用时需要此辅助函数。
#[cfg(feature = "backend-sherpa")]
fn load_punctuator(config: &Option<PunctConfig>) -> Result<Option<FinalTextProcessor>, AsrError> {
    #[cfg(feature = "punct-sherpa")]
    {
        match config {
            Some(c) => Ok(Some(super::backends::punct::create(c)?)),
            None => Ok(None),
        }
    }
    #[cfg(not(feature = "punct-sherpa"))]
    {
        if config.is_some() {
            return Err(AsrError::new(
                ErrorKind::UnsupportedCapability,
                "configuration",
                "punctuation support is disabled in this build",
            ));
        }
        Ok(None)
    }
}
#[cfg(any(
    feature = "backend-sherpa",
    feature = "backend-dashscope",
    feature = "backend-openai-http",
    feature = "backend-openai-realtime"
))]
fn capabilities(
    name: &str,
    rate: u32,
    realtime_audio: bool,
    partial_results: bool,
    remote: bool,
    punctuation: bool,
    supports_session_hints: bool,
) -> BackendCapabilities {
    BackendCapabilities {
        name: name.into(),
        audio: AudioSpec { sample_rate: rate },
        realtime_audio,
        partial_results,
        remote,
        punctuation,
        supports_session_hints,
    }
}

/// 零短语 hints 语义上等价于“无 hints”:`SpeechHints` 可反序列化,配置
/// 驱动的调用方容易携带 `Some(SpeechHints::default())`,若仅判 `is_some()`
/// 会在不支持 hints 的后端被误判为不支持的能力。在进入后端分发前统一
/// 归一化,`start` 与 `transcribe` 及所有后端路径一致受益;非空 hints 不受影响。
fn normalize_session_hints(mut options: SessionOptions) -> SessionOptions {
    if options
        .hints
        .as_ref()
        .is_some_and(|hints| hints.phrases.is_empty())
    {
        options.hints = None;
    }
    options
}
struct Prepared {
    config: EngineConfig,
    capabilities: BackendCapabilities,
    kind: Kind,
    punctuator: Option<FinalTextProcessor>,
    sessions: Arc<AtomicUsize>,
    max_active_sessions: usize,
    #[cfg(any(
        feature = "backend-dashscope",
        feature = "backend-openai-http",
        feature = "backend-openai-realtime"
    ))]
    network: Mutex<Option<Arc<super::backends::network::Network>>>,
}
enum Kind {
    #[cfg(feature = "backend-sherpa")]
    Streaming(
        Arc<sherpa_onnx::OnlineRecognizer>,
        Option<String>,
        super::backends::hotwords::HotwordVocabulary,
    ),
    #[cfg(all(feature = "backend-sherpa", feature = "vad-silero"))]
    Offline(
        Arc<sherpa_onnx::OfflineRecognizer>,
        VadConfig,
        Option<String>,
        super::backends::hotwords::HotwordVocabulary,
    ),
    #[cfg(feature = "backend-dashscope")]
    DashScope(DashScopeConfig),
    #[cfg(feature = "backend-openai-http")]
    Http(OpenAiHttpConfig),
    #[cfg(feature = "backend-openai-realtime")]
    Realtime(OpenAiRealtimeConfig),
}
#[derive(Clone)]
pub struct Engine {
    prepared: Arc<Prepared>,
}
impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("config", &self.prepared.config)
            .field("capabilities", &self.prepared.capabilities)
            .finish()
    }
}
impl Engine {
    pub fn prepare(config: EngineConfig, options: EngineOptions) -> Result<Self, AsrError> {
        let (kind, capabilities, punctuator) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load(&config)))
                .unwrap_or_else(|_| Err(AsrError::backend("model preparation panicked")))?;
        Ok(Self {
            prepared: Arc::new(Prepared {
                config,
                capabilities,
                kind,
                punctuator,
                sessions: Arc::new(AtomicUsize::new(0)),
                max_active_sessions: options.max_active_sessions.get(),
                #[cfg(any(
                    feature = "backend-dashscope",
                    feature = "backend-openai-http",
                    feature = "backend-openai-realtime"
                ))]
                network: Mutex::new(None),
            }),
        })
    }

    #[cfg(all(test, feature = "backend-openai-http"))]
    pub(crate) fn active_sessions(&self) -> usize {
        self.prepared.sessions.load(Ordering::Acquire)
    }

    pub fn config(&self) -> &EngineConfig {
        &self.prepared.config
    }
    pub fn capabilities(&self) -> &BackendCapabilities {
        &self.prepared.capabilities
    }

    pub fn start(&self, options: SessionOptions) -> Result<Session, AsrError> {
        self.start_internal(options, None)
    }

    fn start_internal(
        &self,
        options: SessionOptions,
        deadline: Option<Instant>,
    ) -> Result<Session, AsrError> {
        let options = normalize_session_hints(options);
        if options.hints.is_some() && !self.prepared.capabilities.supports_session_hints {
            return Err(AsrError::new(
                ErrorKind::UnsupportedCapability,
                "start",
                "this engine does not support per-session speech hints",
            ));
        }
        #[cfg(feature = "backend-sherpa")]
        if let Some(hints) = &options.hints {
            super::backends::hotwords::validate_session_hints(hints)
                .map_err(|error| AsrError::new(error.kind, "start", error.message))?;
            if let Some(vocabulary) = self.session_hotword_vocabulary() {
                vocabulary.validate_session(hints)?;
            }
        }
        let permit = Permit::acquire(&self.prepared.sessions, self.prepared.max_active_sessions)?;
        let driver = self.driver(&options)?;
        coordinator::spawn_with_deadline(
            driver,
            self.prepared.capabilities.audio,
            options,
            self.prepared.punctuator.clone(),
            permit,
            deadline,
        )
    }

    #[cfg(any(
        feature = "backend-dashscope",
        feature = "backend-openai-http",
        feature = "backend-openai-realtime"
    ))]
    fn network(&self) -> Result<Arc<super::backends::network::Network>, AsrError> {
        init_on_success(
            &self.prepared.network,
            super::backends::network::Network::new,
        )
    }

    pub fn transcribe(
        &self,
        audio: &AudioBuffer,
        options: SessionOptions,
        deadline: Instant,
    ) -> SessionResult {
        let chunk_frames = match self.validate_transcribe_input(audio, &options, deadline) {
            Ok(chunk_frames) => chunk_frames,
            Err(error) => return failure_result(error, SessionOutcome::default()),
        };
        if Instant::now() >= deadline {
            return failure_result(
                AsrError::new(
                    ErrorKind::Timeout,
                    "transcribe",
                    "transcription deadline expired before session start",
                ),
                SessionOutcome::default(),
            );
        }
        let session = match self.start_internal(options, Some(deadline)) {
            Ok(session) => session,
            Err(error) => return failure_result(error, SessionOutcome::default()),
        };
        let input = session.input();
        for samples in audio.samples.chunks(chunk_frames) {
            if Instant::now() >= deadline {
                return session.finish(deadline);
            }
            let chunk = AudioChunk {
                samples: samples.to_vec(),
                spec: audio.spec,
            };
            // The whole recording was pre-scanned for non-finite /
            // out-of-range samples above, so the internal feed skips the
            // coordinator's redundant per-chunk sample re-scan.
            if let Err(error) = input.push_wait_validated(chunk, deadline) {
                return session.fail_input(error.error);
            }
        }
        session.finish(deadline)
    }

    fn validate_transcribe_input(
        &self,
        audio: &AudioBuffer,
        options: &SessionOptions,
        deadline: Instant,
    ) -> Result<usize, AsrError> {
        if Instant::now() >= deadline {
            return Err(AsrError::new(
                ErrorKind::Timeout,
                "transcribe",
                "transcription deadline already expired",
            ));
        }
        AudioSpec::mono(audio.spec.sample_rate)?;
        if audio.spec != options.input {
            return Err(AsrError::new(
                ErrorKind::InvalidInput,
                "transcribe",
                "audio format does not match SessionOptions input",
            ));
        }
        if audio
            .samples
            .iter()
            .any(|sample| !sample.is_finite() || sample.abs() > 1.0)
        {
            return Err(AsrError::new(
                ErrorKind::InvalidInput,
                "transcribe",
                "audio must contain finite normalized samples in -1..=1",
            ));
        }
        let geometry = coordinator::session_geometry(options)?;
        if audio.samples.len() as u128 > u128::from(geometry.max_session_frames) {
            return Err(AsrError::new(
                ErrorKind::ResourceLimit,
                "transcribe",
                "audio exceeds maximum session duration",
            ));
        }
        self.validate_whole_recording_size(audio)?;
        Ok((options.input.sample_rate as usize / 10)
            .max(1)
            .min(geometry.max_chunk_frames))
    }

    #[cfg(feature = "backend-openai-http")]
    #[allow(irrefutable_let_patterns)]
    fn validate_whole_recording_size(&self, audio: &AudioBuffer) -> Result<(), AsrError> {
        let Kind::Http(config) = &self.prepared.kind else {
            return Ok(());
        };
        if !matches!(config.mode, HttpMode::WholeRecording) {
            return Ok(());
        }
        let numerator = (audio.samples.len() as u128)
            .checked_mul(u128::from(config.sample_rate))
            .ok_or_else(|| {
                AsrError::new(ErrorKind::ResourceLimit, "upload", "audio is too large")
            })?;
        let input_rate = u128::from(audio.spec.sample_rate);
        let output_frames = numerator
            .checked_add(input_rate / 2)
            .map(|value| value / input_rate)
            .ok_or_else(|| {
                AsrError::new(ErrorKind::ResourceLimit, "upload", "audio is too large")
            })?;
        let capacity =
            crate::audio::pcm16_wav_sample_capacity(config.max_upload_bytes).unwrap_or(0);
        if output_frames > capacity as u128 {
            return Err(AsrError::new(
                ErrorKind::ResourceLimit,
                "upload",
                "audio exceeds configured upload limit",
            ));
        }
        Ok(())
    }

    #[cfg(not(feature = "backend-openai-http"))]
    fn validate_whole_recording_size(&self, _: &AudioBuffer) -> Result<(), AsrError> {
        Ok(())
    }
    /// Token vocabulary retained at prepare time for session hint validation.
    #[cfg(all(feature = "backend-sherpa", feature = "vad-silero"))]
    #[allow(unreachable_patterns)]
    fn session_hotword_vocabulary(&self) -> Option<&super::backends::hotwords::HotwordVocabulary> {
        match &self.prepared.kind {
            Kind::Streaming(_, _, vocabulary) | Kind::Offline(_, _, _, vocabulary) => {
                Some(vocabulary)
            }
            _ => None,
        }
    }
    #[cfg(all(feature = "backend-sherpa", not(feature = "vad-silero")))]
    #[allow(unreachable_patterns)]
    fn session_hotword_vocabulary(&self) -> Option<&super::backends::hotwords::HotwordVocabulary> {
        match &self.prepared.kind {
            Kind::Streaming(_, _, vocabulary) => Some(vocabulary),
            _ => None,
        }
    }
    #[allow(unreachable_code, unused_variables)]
    fn driver(&self, options: &SessionOptions) -> Result<Box<dyn Driver>, AsrError> {
        let driver: Box<dyn Driver> = match &self.prepared.kind {
            #[cfg(feature = "backend-sherpa")]
            Kind::Streaming(r, words, _) => Box::new(super::backends::local::Streaming::new(
                r.clone(),
                super::backends::hotwords::merge(words.as_ref(), options.hints.as_ref())?,
            )),
            #[cfg(all(feature = "backend-sherpa", feature = "vad-silero"))]
            Kind::Offline(r, vad, words, _) => Box::new(super::backends::local::Offline::new(
                r.clone(),
                vad.clone(),
                super::backends::hotwords::merge(words.as_ref(), options.hints.as_ref())?,
            )),
            #[cfg(feature = "backend-dashscope")]
            Kind::DashScope(cfg) => Box::new(super::backends::dashscope::DashScopeDriver::new(
                cfg.clone(),
                self.network()?,
            )),
            #[cfg(feature = "backend-openai-http")]
            Kind::Http(cfg) => Box::new(super::backends::http::HttpDriver::new(
                cfg.clone(),
                self.network()?,
            )),
            #[cfg(feature = "backend-openai-realtime")]
            Kind::Realtime(cfg) => Box::new(super::backends::realtime::RealtimeDriver::new(
                cfg.clone(),
                self.network()?,
            )),
            #[allow(unreachable_patterns)]
            _ => {
                return Err(AsrError::new(
                    ErrorKind::UnsupportedCapability,
                    "start",
                    "no enabled backend",
                ))
            }
        };
        Ok(driver)
    }
}
/// 懒加载缓存:首次成功后写入 `slot`,失败不落缓存、由调用方下次重试。
/// 锁内构建即可:`Network::new` 只构造 runtime/HTTP 客户端,无阻塞 I/O,
/// 互斥保证并发时只有一个调用方执行构建。持锁 panic 会毒化锁,但槽位
/// 只在成功后写入、panic 后仍为 `None`,因此恢复锁后重试语义不变。
#[cfg(any(
    feature = "backend-dashscope",
    feature = "backend-openai-http",
    feature = "backend-openai-realtime"
))]
fn init_on_success<T>(
    slot: &Mutex<Option<Arc<T>>>,
    build: impl FnOnce() -> Result<T, AsrError>,
) -> Result<Arc<T>, AsrError> {
    let mut cached = slot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(value) = &*cached {
        return Ok(value.clone());
    }
    let value = Arc::new(build()?);
    *cached = Some(value.clone());
    Ok(value)
}
struct Permit(Arc<AtomicUsize>);
impl Permit {
    fn acquire(count: &Arc<AtomicUsize>, max: usize) -> Result<Self, AsrError> {
        count
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < max).then_some(n + 1)
            })
            .map(|_| Self(count.clone()))
            .map_err(|_| {
                AsrError::new(
                    ErrorKind::Busy,
                    "executor",
                    "configured concurrency limit reached",
                )
            })
    }
}
impl Drop for Permit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(all(test, feature = "backend-openai-http"))]
mod tests;

#[cfg(all(
    test,
    any(
        feature = "backend-dashscope",
        feature = "backend-openai-http",
        feature = "backend-openai-realtime"
    )
))]
mod network_cache_tests;

#[cfg(test)]
mod hint_normalization_tests;

#[cfg(all(test, feature = "backend-sherpa"))]
mod sherpa_hint_normalization_tests;
