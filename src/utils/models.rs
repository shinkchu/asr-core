//! Local model directory detection.
//!
//! `Engine::prepare` remains the authority: it loads the native models and
//! rejects anything they reject. [`detect`] answers the cheaper question
//! before an expensive prepare — what a directory contains. For the
//! configuration-level precheck see [`super::precheck`].

use std::path::Path;

use crate::{backends::model_error, AsrError};

/// What a local model directory contains, decided from its file layout.
///
/// Detection is one-way evidence, mirroring the engine's own family
/// precheck: a positive marker proves a family; the absence of markers is
/// never proof of a specific family.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalModel {
    /// Transducer layout (encoder/decoder/joiner + tokens.txt). Streaming
    /// zipformer and offline transducer archives share this layout, so the
    /// intent is configured separately via `StreamingConfig` or
    /// `OfflineConfig` with `OfflineFamily::Transducer`.
    Transducer,
    /// Flat layout (model*.onnx + tokens.txt) whose tokens carry SenseVoice
    /// language markers — definitive SenseVoice evidence.
    SenseVoice,
    /// Flat layout without markers: Paraformer, a markerless SenseVoice
    /// variant, or a FireRedASR2 CTC export. The directory alone cannot
    /// decide the family; the host must configure it explicitly via
    /// `OfflineFamily::Paraformer` / `FireRedAsrCtc`.
    Flat,
    /// Qwen3-ASR (conv_frontend + encoder/decoder + tokenizer directory).
    Qwen3Asr,
    /// FunASR-Nano (encoder_adaptor/llm/embedding + tokenizer directory).
    FunAsrNano,
    /// FireRedASR-AED (encoder/decoder + tokens.txt, no joiner). Covers both
    /// FireRedASR 1.0-AED-L and FireRedASR2 sherpa-onnx AED exports. The CTC
    /// export shares the flat model*.onnx + tokens.txt layout with Paraformer
    /// and is reported as [`LocalModel::Flat`].
    FireRedAsrAed,
    /// Punctuation model (model*.onnx, optionally with bpe.vocab). Only
    /// reported with the `punct-sherpa` feature.
    Punct,
}

/// Detects the model family a directory contains. The error of a recognized
/// but incomplete family directory names the missing file — including
/// archives wrapped in a single child directory; a directory that matches no
/// known layout fails with a generic message.
pub fn detect(dir: &Path) -> Result<LocalModel, AsrError> {
    use crate::backends::model_layout as layout;

    // Flat check first: a complete ASR directory (model*.onnx + tokens.txt)
    // must never be reported as a punctuation model, whose layout it
    // strictly contains.
    if let Ok(files) = layout::find_offline_model_files(dir) {
        return match layout::is_sense_voice_tokens(&files.tokens) {
            Ok(true) => Ok(LocalModel::SenseVoice),
            Ok(false) => Ok(LocalModel::Flat),
            Err(error) => Err(model_error(error)),
        };
    }
    if layout::find_model_files(dir).is_ok() {
        return Ok(LocalModel::Transducer);
    }
    // The find_* helpers descend into a single wrapper child themselves, but
    // the family-marker guard must test the same resolved directory —
    // checking the original `dir` would let a wrapped-but-incomplete archive
    // lose its specific error to the generic fallback.
    let qwen3_dir = layout::descend_where(dir, layout::has_conv_frontend);
    match layout::find_qwen3_files(&qwen3_dir) {
        Ok(_) => return Ok(LocalModel::Qwen3Asr),
        // conv_frontend在场说明是 Qwen3 目录，报它自己的缺失项。
        Err(error) if layout::has_conv_frontend(&qwen3_dir) => return Err(model_error(error)),
        Err(_) => {}
    }
    let nano_dir = layout::descend_where(dir, layout::has_encoder_adaptor);
    match layout::find_funasr_nano_files(&nano_dir) {
        Ok(_) => return Ok(LocalModel::FunAsrNano),
        // encoder_adaptor在场说明是 FunASR-Nano 目录，报它自己的缺失项。
        Err(error) if layout::has_encoder_adaptor(&nano_dir) => return Err(model_error(error)),
        Err(_) => {}
    }
    // FireRed-AED 的 encoder 标记（encoder*.onnx 且无 joiner）也命中 Qwen3
    // 目录（encoder/decoder 同名），故必须放在 Qwen3/FunASR 之后裁决。
    let aed_dir = layout::descend_where(dir, layout::has_fire_red_aed_layout_marker);
    match layout::find_fire_red_aed_files(&aed_dir) {
        Ok(_) => return Ok(LocalModel::FireRedAsrAed),
        // encoder 在场且无 joiner 说明是 FireRed-AED 目录，报它自己的缺失项。
        Err(error) if layout::has_fire_red_aed_layout_marker(&aed_dir) => {
            return Err(model_error(error))
        }
        Err(_) => {}
    }
    #[cfg(feature = "punct-sherpa")]
    if layout::find_punct_model_files(dir).is_ok() {
        return Ok(LocalModel::Punct);
    }
    Err(model_error(
        "unrecognized model directory layout; expected a transducer, FireRedASR-AED, \
         SenseVoice/Paraformer, Qwen3-ASR, FunASR-Nano or punctuation model directory",
    ))
}

#[cfg(test)]
mod tests;
