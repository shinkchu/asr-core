//! Cheap pre-native parameter and layout checks for local backends.
//!
//! [`streaming`] and [`offline`] run every check `Engine::prepare` performs
//! before native model initialization: family capability rules (which
//! families accept a language override, transducer bias, or engine-level
//! prompt hints), VAD parameters, hotword bias against the model's own
//! vocabulary, filesystem layout, and punctuation layout. `prepare` calls
//! them first, and `crate::utils::precheck::validate` delegates to the same
//! functions, so both entry points apply the same canonical pre-native
//! validation rules. They never construct native models.

use super::model_error;
#[cfg(all(feature = "backend-sherpa", feature = "punct-sherpa"))]
use super::punctuation_error;
use crate::{AsrError, StreamingConfig};
#[cfg(feature = "vad-silero")]
use crate::{OfflineFamily, SpeechHints, TransducerBiasConfig};

/// Recognizer thread-count rule shared by [`streaming`] and [`offline`]:
/// `1..=MAX_NUM_THREADS`. Runs before any filesystem access so a bad value
/// fails without a model on disk.
fn validate_num_threads(num_threads: usize) -> Result<(), AsrError> {
    if num_threads == 0 || num_threads > crate::MAX_NUM_THREADS {
        return Err(AsrError::invalid(format!(
            "num_threads must be between 1 and {}",
            crate::MAX_NUM_THREADS
        )));
    }
    Ok(())
}

/// Platform checks only: native provider availability depends on the linked
/// runtime and is not exposed by sherpa-onnx's Rust API.
pub(crate) fn validate_provider(provider: crate::ExecutionProvider) -> Result<(), AsrError> {
    use crate::ExecutionProvider;
    let supported = match provider {
        ExecutionProvider::Cpu => true,
        ExecutionProvider::Cuda => cfg!(any(target_os = "linux", target_os = "windows")),
        ExecutionProvider::CoreMl => cfg!(target_vendor = "apple"),
    };
    if !supported {
        return Err(AsrError::new(
            crate::ErrorKind::UnsupportedCapability,
            "configuration",
            format!(
                "provider {} is unsupported on this platform",
                provider.as_str()
            ),
        ));
    }
    Ok(())
}

/// Matches the stage `punct::create` reports for the same layout failure, so
/// precheck does not change observable error semantics.
/// Cheap precheck of a [`StreamingConfig`]: hotword bias syntax, model
/// layout, modeling-unit resolution (bpe.vocab presence), bias phrases
/// against the model tokens, and punctuation layout. Mirrors the order in
/// which `Engine::prepare`'s streaming arm applies the same checks.
#[cfg(feature = "backend-sherpa")]
pub(crate) fn streaming(config: &StreamingConfig) -> Result<(), AsrError> {
    validate_num_threads(config.num_threads)?;
    validate_provider(config.provider)?;
    let files = super::model_layout::find_model_files(&config.model_dir).map_err(model_error)?;
    if let Some(bias) = &config.bias {
        super::hotwords::prepare_bias(bias, &files)?;
    }
    validate_punct(config.punctuation.as_ref())
}

/// Family capability rules that hold before any filesystem access. Shared
/// by `load_offline` and [`offline`], so a family rule has exactly one home.
#[cfg(feature = "vad-silero")]
pub(crate) fn validate_family_parameters(
    family: OfflineFamily,
    language: Option<&str>,
    bias: Option<&TransducerBiasConfig>,
    prompt_hints: Option<&SpeechHints>,
) -> Result<(), AsrError> {
    match family {
        OfflineFamily::SenseVoice | OfflineFamily::Paraformer => {
            if bias.is_some() || prompt_hints.is_some() {
                return Err(AsrError::new(
                    crate::ErrorKind::UnsupportedCapability,
                    "configuration",
                    format!("{family:?} does not support speech hints or transducer bias"),
                ));
            }
            if let Some(language) = language {
                match family {
                    OfflineFamily::SenseVoice => {
                        if !["auto", "zh", "en", "ja", "ko", "yue"].contains(&language) {
                            return Err(AsrError::invalid("unsupported SenseVoice language"));
                        }
                    }
                    _ => {
                        return Err(AsrError::invalid(
                            "Paraformer does not accept a language override",
                        ));
                    }
                }
            }
        }
        OfflineFamily::Transducer => {
            if language.is_some() {
                return Err(AsrError::invalid(
                    "Transducer does not accept a language override",
                ));
            }
            if prompt_hints.is_some() {
                return Err(AsrError::new(
                    crate::ErrorKind::UnsupportedCapability,
                    "configuration",
                    "Transducer does not support engine-level prompt hints",
                ));
            }
        }
        OfflineFamily::Qwen3Asr | OfflineFamily::FunAsrNano => {
            if language.is_some() {
                return Err(AsrError::invalid(format!(
                    "{family:?} does not accept a language override"
                )));
            }
            if bias.is_some() {
                return Err(AsrError::new(
                    crate::ErrorKind::UnsupportedCapability,
                    "configuration",
                    format!("{family:?} does not support transducer bias"),
                ));
            }
        }
        OfflineFamily::FireRedAsrAed | OfflineFamily::FireRedAsrCtc => {
            if bias.is_some() || prompt_hints.is_some() {
                return Err(AsrError::new(
                    crate::ErrorKind::UnsupportedCapability,
                    "configuration",
                    format!("{family:?} does not support speech hints or transducer bias"),
                ));
            }
            if language.is_some() {
                return Err(AsrError::invalid(format!(
                    "{family:?} does not accept a language override"
                )));
            }
        }
    }
    Ok(())
}

/// Cheap precheck of an [`OfflineConfig`]: hotword bias syntax, VAD
/// parameters, family capability rules, model layout, family contradiction,
/// prompt-hint rendering rules, and punctuation layout — in the same order
/// `Engine::prepare`'s offline arm applies them.
#[cfg(all(feature = "backend-sherpa", feature = "vad-silero"))]
pub(crate) fn offline(config: &crate::OfflineConfig) -> Result<(), AsrError> {
    validate_num_threads(config.num_threads)?;
    validate_provider(config.provider)?;
    // The cheap half of prepare's `vad::preflight`; the native detector is
    // still built once in prepare.
    super::vad::validate(&config.vad)?;
    validate_family_parameters(
        config.family,
        config.language.as_deref(),
        config.transducer_bias.as_ref(),
        config.prompt_hints.as_ref(),
    )?;
    match config.family {
        // 三个家族共享"单 onnx + tokens.txt"布局发现与 SenseVoice 矛盾守卫。
        OfflineFamily::SenseVoice | OfflineFamily::Paraformer | OfflineFamily::FireRedAsrCtc => {
            let files = super::model_layout::find_offline_model_files(&config.model_dir)
                .map_err(model_error)?;
            super::model_layout::ensure_family_not_contradicted(config.family, &files.tokens)
                .map_err(model_error)?;
        }
        OfflineFamily::FireRedAsrAed => {
            super::model_layout::find_fire_red_aed_files(&config.model_dir).map_err(model_error)?;
        }
        OfflineFamily::Transducer => {
            let files =
                super::model_layout::find_model_files(&config.model_dir).map_err(model_error)?;
            if let Some(bias) = &config.transducer_bias {
                super::hotwords::prepare_bias(bias, &files)?;
            }
        }
        OfflineFamily::Qwen3Asr => {
            super::model_layout::find_qwen3_files(&config.model_dir).map_err(model_error)?;
        }
        OfflineFamily::FunAsrNano => {
            super::model_layout::find_funasr_nano_files(&config.model_dir).map_err(model_error)?;
        }
    }
    if let Some(hints) = &config.prompt_hints {
        match config.family {
            OfflineFamily::Qwen3Asr => {
                super::hotwords::qwen3_prompt(hints)?;
            }
            OfflineFamily::FunAsrNano => {
                super::hotwords::funasr_nano_prompt(hints)?;
            }
            // Rejected by validate_family_parameters above.
            _ => {}
        }
    }
    validate_punct(config.punctuation.as_ref())
}

#[cfg(all(feature = "backend-sherpa", feature = "punct-sherpa"))]
fn validate_punct(config: Option<&crate::PunctConfig>) -> Result<(), AsrError> {
    match config {
        Some(config) => super::model_layout::find_punct_model_files(&config.model)
            .map_err(punctuation_error)
            .map(|_| ()),
        None => Ok(()),
    }
}

#[cfg(all(feature = "backend-sherpa", not(feature = "punct-sherpa")))]
fn validate_punct(config: Option<&crate::PunctConfig>) -> Result<(), AsrError> {
    if config.is_some() {
        return Err(AsrError::new(
            crate::ErrorKind::UnsupportedCapability,
            "configuration",
            "punctuation support is disabled in this build",
        ));
    }
    Ok(())
}
