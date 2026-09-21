use crate::{AsrError, ErrorKind, VadConfig};
use sherpa_onnx::{VadModelConfig, VoiceActivityDetector};

pub(crate) fn validate(c: &VadConfig) -> Result<(), AsrError> {
    if !c.model.is_file() || c.model.metadata().map_or(true, |m| m.len() == 0) {
        return Err(AsrError::new(
            ErrorKind::InvalidModel,
            "vad",
            "VAD model must be a nonempty file",
        ));
    }
    if !c.threshold.is_finite()
        || !(0.0..1.0).contains(&c.threshold)
        || [c.min_silence, c.min_speech, c.max_speech]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        || c.min_silence > 60.0
        || c.min_speech > 60.0
        || c.max_speech > 60.0
        || c.min_speech >= c.max_speech
    {
        return Err(AsrError::invalid(
            "invalid VAD threshold or speech durations",
        ));
    }
    Ok(())
}
pub(crate) fn create(c: &VadConfig) -> Result<VoiceActivityDetector, AsrError> {
    validate(c)?;
    let mut config = VadModelConfig::default();
    config.silero_vad.model = Some(c.model.to_string_lossy().into_owned());
    config.silero_vad.threshold = c.threshold;
    config.silero_vad.min_silence_duration = c.min_silence;
    config.silero_vad.min_speech_duration = c.min_speech;
    config.silero_vad.max_speech_duration = c.max_speech;
    config.silero_vad.window_size = 512;
    config.sample_rate = 16000;
    config.num_threads = 1;
    config.provider = Some("cpu".into());
    VoiceActivityDetector::create(&config, c.max_speech + c.min_silence + 5.0)
        .ok_or_else(|| AsrError::backend("failed to initialize VAD"))
}

/// 在 prepare 阶段构建并丢弃一个 detector,把模型文件损坏、ONNX 初始化失败
/// 等错误一次性暴露,而不是推迟到每个 session 的 start 逐次失败。
///
/// `VoiceActivityDetector` 虽声明为 `Send + Sync`,但它是 per-session 的
/// 有状态对象(音频环形缓冲、语音分段队列、模型循环状态),且 crate 不提供
/// 从已加载模型派生新实例的 API,因此无法像识别器那样跨 session 共享;
/// 运行期仍由各 session 调用 [`create`] 自行构建。
pub(crate) fn preflight(c: &VadConfig) -> Result<(), AsrError> {
    create(c).map(drop)
}
