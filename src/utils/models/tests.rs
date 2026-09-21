use super::*;
use crate::backends::model_layout::fixtures::{dir, flat_dir, funasr_dir, qwen3_dir};
use crate::ErrorKind;

#[test]
fn directories_detect_as_their_layout_families() {
    let sense = flat_dir("foo 1\n<|zh|> 2\n<|itn|> 3\n");
    assert_eq!(detect(&sense.1).unwrap(), LocalModel::SenseVoice);
    let para = flat_dir("foo 1\nbar 2\n");
    assert_eq!(detect(&para.1).unwrap(), LocalModel::Flat);

    let transducer = dir(&["encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx"]);
    std::fs::write(transducer.1.join("tokens.txt"), "a 1\n").unwrap();
    assert_eq!(detect(&transducer.1).unwrap(), LocalModel::Transducer);

    let qwen3 = qwen3_dir();
    assert_eq!(detect(&qwen3.1).unwrap(), LocalModel::Qwen3Asr);
    let funasr = funasr_dir();
    assert_eq!(detect(&funasr.1).unwrap(), LocalModel::FunAsrNano);

    let aed = dir(&["encoder.int8.onnx", "decoder.int8.onnx"]);
    std::fs::write(aed.1.join("tokens.txt"), "a 1\n").unwrap();
    assert_eq!(detect(&aed.1).unwrap(), LocalModel::FireRedAsrAed);
    // CTC 导出与 Paraformer 同形（model*.onnx + tokens.txt，无标记）：
    // 单向证据原则下报告为 Flat，由宿主显式配置家族。
    let ctc = flat_dir("foo 1\nbar 2\n");
    assert_eq!(detect(&ctc.1).unwrap(), LocalModel::Flat);
}

#[cfg(feature = "punct-sherpa")]
#[test]
fn model_only_directories_detect_as_punct_and_asr_directories_do_not() {
    let punct = dir(&["model.int8.onnx"]);
    assert_eq!(detect(&punct.1).unwrap(), LocalModel::Punct);
    // An ASR directory strictly contains the punctuation layout; the
    // flat-first detection order must keep it SenseVoice/Flat.
    let sense = flat_dir("<|zh|> 1\n");
    assert_ne!(detect(&sense.1).unwrap(), LocalModel::Punct);
}

#[test]
fn incomplete_family_directories_name_the_missing_file_and_unknown_ones_fail() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("conv_frontend.onnx"), [0u8; 8]).unwrap();
    let error = detect(dir.path()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(error.message.contains("encoder"), "{error}");

    let empty = tempfile::tempdir().unwrap();
    let error = detect(empty.path()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(error.message.contains("unrecognized"), "{error}");
}

#[test]
fn wrapped_incomplete_family_directories_still_name_the_missing_file() {
    // A Qwen3 archive unpacked into a single child directory with the
    // encoder missing: the family marker lives in the child, so the
    // error must come from the resolved directory, not the generic
    // fallback for the wrapper.
    let dir = tempfile::tempdir().unwrap();
    let child = dir.path().join("sherpa-onnx-qwen3-asr-0.6B-int8");
    std::fs::create_dir(&child).unwrap();
    std::fs::write(child.join("conv_frontend.onnx"), [0u8; 8]).unwrap();
    let error = detect(dir.path()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(error.message.contains("encoder"), "{error}");

    // Same for a wrapped FunASR-Nano directory missing its llm.
    let dir = tempfile::tempdir().unwrap();
    let child = dir.path().join("sherpa-onnx-funasr-nano");
    std::fs::create_dir(&child).unwrap();
    std::fs::write(child.join("encoder_adaptor.onnx"), [0u8; 8]).unwrap();
    let error = detect(dir.path()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(error.message.contains("llm"), "{error}");

    // Same for a wrapped FireRedASR-AED directory missing its decoder.
    let dir = tempfile::tempdir().unwrap();
    let child = dir
        .path()
        .join("sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26");
    std::fs::create_dir(&child).unwrap();
    std::fs::write(child.join("encoder.int8.onnx"), [0u8; 8]).unwrap();
    std::fs::write(child.join("tokens.txt"), "a 1\n").unwrap();
    let error = detect(dir.path()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(error.message.contains("decoder"), "{error}");
}
