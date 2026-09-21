use super::*;
use crate::{ErrorKind, DEFAULT_NUM_THREADS};

#[test]
fn ambiguous_model_candidates_surface_as_invalid_model() {
    let dir = tempfile::tempdir().unwrap();
    for name in [
        "encoder-a.int8.onnx",
        "encoder-b.int8.onnx",
        "decoder.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    ] {
        std::fs::write(dir.path().join(name), [1u8; 8]).unwrap();
    }

    let error = match load_stream(dir.path(), None, DEFAULT_NUM_THREADS) {
        Ok(_) => panic!("ambiguous model layout must fail before native initialization"),
        Err(error) => error,
    };
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(error.message.contains("encoder-a.int8.onnx"));
    assert!(error.message.contains("encoder-b.int8.onnx"));
    assert!(!error.message.contains(&dir.path().display().to_string()));
}

#[test]
fn empty_bpe_vocabulary_surfaces_as_invalid_model() {
    let dir = tempfile::tempdir().unwrap();
    for name in [
        "encoder.int8.onnx",
        "decoder.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    ] {
        std::fs::write(dir.path().join(name), [1u8; 8]).unwrap();
    }
    std::fs::write(dir.path().join("bpe.vocab"), []).unwrap();

    let error = match load_stream(dir.path(), None, DEFAULT_NUM_THREADS) {
        Ok(_) => panic!("empty bpe.vocab must fail before native initialization"),
        Err(error) => error,
    };
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(error.message.contains("bpe.vocab is empty"));
}

/// 接线测试：load_offline 确实经过家族预检。预检位于 OfflineRecognizer::create
/// 之前，垃圾模型文件即可触发，无需真实模型（CI 可跑）。
#[cfg(feature = "vad-silero")]
#[test]
fn offline_family_precheck_fires_before_native_initialization() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("model.int8.onnx"), [1u8; 8]).unwrap();
    std::fs::write(dir.path().join("tokens.txt"), "foo 1\n<|zh|> 2\n").unwrap();

    let error = match load_offline(
        dir.path(),
        OfflineFamily::Paraformer,
        None,
        None,
        None,
        DEFAULT_NUM_THREADS,
    ) {
        Ok(_) => panic!("marked tokens must reject a Paraformer configuration"),
        Err(error) => error,
    };
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(error.message.contains("SenseVoice language markers"));

    // FireRedAsrCtc 走同一"单 onnx + tokens"布局发现与矛盾守卫。
    let error = match load_offline(
        dir.path(),
        OfflineFamily::FireRedAsrCtc,
        None,
        None,
        None,
        DEFAULT_NUM_THREADS,
    ) {
        Ok(_) => panic!("marked tokens must reject a FireRedAsrCtc configuration"),
        Err(error) => error,
    };
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(error.message.contains("SenseVoice language markers"));

    // FireRedAsrAed 缺 decoder 在原生初始化前以 InvalidModel 点名。
    let aed = tempfile::tempdir().unwrap();
    std::fs::write(aed.path().join("encoder.int8.onnx"), [1u8; 8]).unwrap();
    std::fs::write(aed.path().join("tokens.txt"), "foo 1\n").unwrap();
    let error = match load_offline(
        aed.path(),
        OfflineFamily::FireRedAsrAed,
        None,
        None,
        None,
        DEFAULT_NUM_THREADS,
    ) {
        Ok(_) => panic!("missing decoder must fail before native initialization"),
        Err(error) => error,
    };
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(error.message.contains("decoder"), "{error:?}");
}
