//! Temp-directory builders for model layout tests, shared by
//! `utils::models`, `utils::precheck` and the local backend's tests so the
//! family directory shapes exist once. Each builder returns the kept
//! `TempDir` (dropping it removes the directory) and its path.

use std::path::PathBuf;

/// A directory containing exactly `names` as small placeholder files.
pub(crate) fn dir(names: &[&str]) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    for name in names {
        std::fs::write(dir.path().join(name), [0u8; 8]).unwrap();
    }
    let path = dir.path().to_path_buf();
    (dir, path)
}

/// A "single onnx + tokens" directory (SenseVoice / Paraformer /
/// FireRedAsrCtc shape) with the given tokens content.
pub(crate) fn flat_dir(tokens: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("model.int8.onnx"), [0u8; 8]).unwrap();
    std::fs::write(dir.path().join("tokens.txt"), tokens).unwrap();
    let path = dir.path().to_path_buf();
    (dir, path)
}

/// A streaming / offline transducer directory (encoder/decoder/joiner +
/// tokens).
pub(crate) fn transducer_dir(tokens: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    for name in ["encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx"] {
        std::fs::write(dir.path().join(name), [0u8; 8]).unwrap();
    }
    std::fs::write(dir.path().join("tokens.txt"), tokens).unwrap();
    let path = dir.path().to_path_buf();
    (dir, path)
}

/// A Qwen3-ASR directory (conv_frontend/encoder/decoder + tokenizer dir).
pub(crate) fn qwen3_dir() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("conv_frontend.onnx"), [0u8; 8]).unwrap();
    std::fs::write(dir.path().join("encoder.int8.onnx"), [0u8; 8]).unwrap();
    std::fs::write(dir.path().join("decoder.int8.onnx"), [0u8; 8]).unwrap();
    let tokenizer = dir.path().join("tokenizer");
    std::fs::create_dir(&tokenizer).unwrap();
    std::fs::write(tokenizer.join("merges.txt"), "#version").unwrap();
    std::fs::write(tokenizer.join("vocab.json"), "{}").unwrap();
    let path = dir.path().to_path_buf();
    (dir, path)
}

/// A FunASR-Nano directory (encoder_adaptor/llm/embedding + tokenizer dir).
pub(crate) fn funasr_dir() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    for name in [
        "encoder_adaptor.int8.onnx",
        "llm.int8.onnx",
        "embedding.int8.onnx",
    ] {
        std::fs::write(dir.path().join(name), [0u8; 8]).unwrap();
    }
    let tokenizer = dir.path().join("Qwen3-0.6B");
    std::fs::create_dir(&tokenizer).unwrap();
    for name in ["vocab.json", "merges.txt", "tokenizer.json"] {
        std::fs::write(tokenizer.join(name), [0u8; 8]).unwrap();
    }
    let path = dir.path().to_path_buf();
    (dir, path)
}
