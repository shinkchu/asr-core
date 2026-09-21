// Imports follow the gates of the tests that need them, so builds with
// only cloud or no backend features compile without unused warnings.
#[cfg(any(
    feature = "backend-sherpa",
    feature = "backend-dashscope",
    feature = "backend-openai-http",
    feature = "backend-openai-realtime"
))]
use super::*;

#[cfg(all(feature = "backend-sherpa", not(feature = "punct-sherpa")))]
use crate::backends::model_layout::fixtures::dir;
#[cfg(feature = "backend-sherpa")]
use crate::backends::model_layout::fixtures::transducer_dir;
#[cfg(all(feature = "backend-sherpa", feature = "vad-silero"))]
use crate::backends::model_layout::fixtures::{flat_dir, funasr_dir, qwen3_dir};

#[cfg(feature = "backend-sherpa")]
#[test]
fn validate_checks_streaming_and_punct_paths() {
    // Streaming requires the transducer layout (encoder/decoder/joiner).
    let (_guard, streaming_path) = transducer_dir("a 1\n");
    let mut config = crate::StreamingConfig::new(&streaming_path);
    assert!(validate(&EngineConfig::Streaming(config.clone())).is_ok());

    // Punctuation pointing at the transducer directory must fail: the
    // punct layout is model*.onnx only, and encoder/decoder/joiner do
    // not match it. (A flat ASR directory, in contrast, *is* a valid
    // punct layout — use `detect` to keep the two apart.)
    #[cfg(feature = "punct-sherpa")]
    {
        config.punctuation = Some(crate::PunctConfig::new(&streaming_path));
        let error = validate(&EngineConfig::Streaming(config.clone())).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidModel);
        // Shared precheck must keep the stage the load path has always
        // reported for this failure (`punct::create`).
        assert_eq!(error.stage, "punctuation");
        assert_precheck_matches_prepare(EngineConfig::Streaming(config));
    }
    #[cfg(not(feature = "punct-sherpa"))]
    {
        // Mirrors Engine::prepare: punctuation is compiled out.
        let (punct_guard, punct_path) = dir(&["model.int8.onnx"]);
        config.punctuation = Some(crate::PunctConfig::new(&punct_path));
        let error = validate(&EngineConfig::Streaming(config)).unwrap_err();
        assert_eq!(error.kind, ErrorKind::UnsupportedCapability);
        drop(punct_guard);
    }
}

/// The whole point of the shared validators: a configuration the
/// precheck accepts must not fail `Engine::prepare` on a cheap check.
#[cfg(feature = "backend-sherpa")]
#[test]
fn validate_applies_the_same_hotword_bias_rules_as_prepare() {
    let (_guard, path) = transducer_dir("a 1\n");
    let mut bias = crate::TransducerBiasConfig::new(vec![crate::BiasPhrase::new("张三")]);
    bias.default_score = f32::NAN;
    let mut config = crate::StreamingConfig::new(&path);
    config.bias = Some(bias);
    let error = validate(&EngineConfig::Streaming(config)).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert!(error.message.contains("finite"), "{error}");
}

#[cfg(all(feature = "backend-sherpa", feature = "vad-silero"))]
#[test]
fn validate_applies_the_same_vad_parameter_rules_as_prepare() {
    // The TempDir must stay bound for the whole test: it owns the
    // directory the files live in.
    let (_guard, path) = flat_dir("foo 1\n<|zh|> 2\n");
    let vad_path = path.join("silero_vad.onnx");
    std::fs::write(&vad_path, [0u8; 8]).unwrap();
    let mut vad = crate::VadConfig::new(&vad_path);
    vad.threshold = f32::NAN;
    let config = crate::OfflineConfig::new(&path, crate::OfflineFamily::SenseVoice, vad);
    let error = validate(&EngineConfig::Offline(config)).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert!(error.message.contains("VAD"), "{error}");
}

#[cfg(all(feature = "backend-sherpa", feature = "vad-silero"))]
#[test]
fn validate_offline_enforces_family_contradiction_and_vad_presence() {
    let (_guard, path) = flat_dir("foo 1\n<|zh|> 2\n");
    let vad_path = path.join("silero_vad.onnx");
    std::fs::write(&vad_path, [0u8; 8]).unwrap();
    let vad = crate::VadConfig::new(&vad_path);

    let config = crate::OfflineConfig::new(&path, crate::OfflineFamily::SenseVoice, vad.clone());
    assert!(validate(&EngineConfig::Offline(config)).is_ok());

    // SenseVoice markers are one-way evidence: configuring Paraformer on
    // a marked directory must be rejected before any model load.
    let config = crate::OfflineConfig::new(&path, crate::OfflineFamily::Paraformer, vad.clone());
    let error = validate(&EngineConfig::Offline(config)).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(error.message.contains("SenseVoice"), "{error}");

    // Missing VAD file fails cheaply.
    let config = crate::OfflineConfig::new(&path, crate::OfflineFamily::SenseVoice, {
        let mut vad = vad;
        vad.model = path.join("missing.onnx");
        vad
    });
    let error = validate(&EngineConfig::Offline(config)).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidModel);
}

#[cfg(feature = "backend-dashscope")]
#[test]
fn validate_delegates_cloud_parameter_checks() {
    let config = EngineConfig::DashScope(crate::DashScopeConfig::new(
        "http://localhost/v1",
        "model",
        crate::Secret::new("k"),
    ));
    let error = validate(&config).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
}

#[cfg(feature = "backend-dashscope")]
#[test]
fn validate_rejects_invalid_dashscope_endpoints_and_parameters() {
    let endpoint = validate(&EngineConfig::DashScope(crate::DashScopeConfig::new(
        "not a url",
        "model",
        crate::Secret::new("k"),
    )))
    .unwrap_err();
    assert_eq!(endpoint.kind, ErrorKind::InvalidInput);
    assert_eq!(endpoint.stage, "configuration");

    let parameters = validate(&EngineConfig::DashScope(crate::DashScopeConfig::new(
        "ws://localhost/v1",
        " ",
        crate::Secret::new("k"),
    )))
    .unwrap_err();
    assert_eq!(parameters.kind, ErrorKind::InvalidInput);
    assert_eq!(parameters.stage, "configuration");
    assert_eq!(parameters.message, "model is required");
}

#[cfg(feature = "backend-openai-http")]
#[test]
fn validate_rejects_invalid_openai_http_parameters() {
    let api_root = validate(&EngineConfig::OpenAiHttp(crate::OpenAiHttpConfig::new(
        "not a url",
        "model",
        crate::Secret::new("k"),
    )))
    .unwrap_err();
    assert_eq!(api_root.kind, ErrorKind::InvalidInput);

    let api_key = validate(&EngineConfig::OpenAiHttp(crate::OpenAiHttpConfig::new(
        "https://localhost/v1",
        "model",
        crate::Secret::new("  "),
    )))
    .unwrap_err();
    assert_eq!(api_key.kind, ErrorKind::InvalidInput);
    assert_eq!(api_key.message, "OpenAI HTTP requires an API key");
}

#[cfg(feature = "backend-openai-realtime")]
#[test]
fn validate_rejects_invalid_openai_realtime_endpoints_and_parameters() {
    let endpoint = validate(&EngineConfig::OpenAiRealtime(
        crate::OpenAiRealtimeConfig::new("not a url", "model", crate::Secret::new("k")),
    ))
    .unwrap_err();
    assert_eq!(endpoint.kind, ErrorKind::InvalidInput);
    assert_eq!(endpoint.stage, "configuration");

    let api_key = validate(&EngineConfig::OpenAiRealtime(
        crate::OpenAiRealtimeConfig::new("ws://localhost/v1", "model", crate::Secret::new("")),
    ))
    .unwrap_err();
    assert_eq!(api_key.kind, ErrorKind::InvalidInput);
    assert_eq!(api_key.message, "OpenAI Realtime requires an API key");
}

/// 防漂移哨兵：对每个非法配置，`utils::precheck::validate` 与
/// `Engine::prepare` 必须同时失败且 kind/stage/message 一致。两者共享
/// 同一份 canonical pre-native 校验入口；loader 后半段仍会重复执行部分
/// 廉价校验，新增这类校验时须同步进共享 precheck。precheck 若漏检
/// （返回 Ok），下面的 `unwrap_err` 会当场 panic。
#[cfg(feature = "backend-sherpa")]
fn assert_precheck_matches_prepare(config: EngineConfig) {
    let precheck = validate(&config).unwrap_err();
    let prepare = crate::Engine::prepare(config, crate::EngineOptions::default()).unwrap_err();
    assert_eq!(precheck.kind, prepare.kind, "{precheck} vs {prepare}");
    assert_eq!(precheck.stage, prepare.stage, "{precheck} vs {prepare}");
    assert_eq!(precheck.message, prepare.message, "{precheck} vs {prepare}");
}

#[cfg(feature = "backend-sherpa")]
#[test]
fn precheck_fails_exactly_where_prepare_fails_for_streaming_bias() {
    // modeling_unit 'bpe' without a bpe.vocab in the model directory:
    // rejected through resolve_modeling_unit, before native init.
    let (_guard, path) = transducer_dir("a 1\n");
    let mut bias = crate::TransducerBiasConfig::new(Vec::new());
    bias.modeling_unit = Some("bpe".into());
    let mut config = crate::StreamingConfig::new(&path);
    config.bias = Some(bias);
    let error = validate(&EngineConfig::Streaming(config.clone())).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(error.message.contains("bpe.vocab"), "{error}");
    assert_precheck_matches_prepare(EngineConfig::Streaming(config));

    // A bias phrase whose characters the model tokens do not know.
    let (_guard, path) = transducer_dir("语 1\n音 2\n");
    let bias = crate::TransducerBiasConfig::new(vec![crate::BiasPhrase::new("语音识别")]);
    let mut config = crate::StreamingConfig::new(&path);
    config.bias = Some(bias);
    let error = validate(&EngineConfig::Streaming(config.clone())).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(
        error.message.contains("missing from the model tokens"),
        "{error}"
    );
    assert_precheck_matches_prepare(EngineConfig::Streaming(config));
}

#[cfg(all(feature = "backend-sherpa", feature = "vad-silero"))]
#[test]
fn precheck_fails_exactly_where_prepare_fails_for_offline_family_rules() {
    let vad_dir = tempfile::tempdir().unwrap();
    let vad_path = vad_dir.path().join("silero_vad.onnx");
    std::fs::write(&vad_path, [0u8; 8]).unwrap();
    let vad = crate::VadConfig::new(&vad_path);

    // Capability rules reject before any filesystem access, so a bare
    // directory is enough for these fixtures.
    let bare = tempfile::tempdir().unwrap();
    let with_bias = || crate::TransducerBiasConfig::new(vec![crate::BiasPhrase::new("张三")]);
    let with_hints = || crate::SpeechHints::new(vec!["张三".into()]);
    macro_rules! case {
        ($name:literal, $family:expr, $kind:expr, $message:literal, $field:ident, $value:expr) => {{
            let mut config = crate::OfflineConfig::new(bare.path(), $family, vad.clone());
            config.$field = $value;
            let error = validate(&EngineConfig::Offline(config.clone())).unwrap_err();
            assert_eq!(error.kind, $kind, "{}: {error}", $name);
            assert!(error.message.contains($message), "{}: {error}", $name);
            assert_precheck_matches_prepare(EngineConfig::Offline(config));
        }};
    }
    case!(
        "sensevoice bias",
        crate::OfflineFamily::SenseVoice,
        ErrorKind::UnsupportedCapability,
        "does not support speech hints or transducer bias",
        transducer_bias,
        Some(with_bias())
    );
    case!(
        "sensevoice prompt hints",
        crate::OfflineFamily::SenseVoice,
        ErrorKind::UnsupportedCapability,
        "does not support speech hints or transducer bias",
        prompt_hints,
        Some(with_hints())
    );
    case!(
        "sensevoice language",
        crate::OfflineFamily::SenseVoice,
        ErrorKind::InvalidInput,
        "unsupported SenseVoice language",
        language,
        Some("fr".into())
    );
    case!(
        "paraformer language",
        crate::OfflineFamily::Paraformer,
        ErrorKind::InvalidInput,
        "Paraformer does not accept a language override",
        language,
        Some("zh".into())
    );
    case!(
        "transducer language",
        crate::OfflineFamily::Transducer,
        ErrorKind::InvalidInput,
        "Transducer does not accept a language override",
        language,
        Some("zh".into())
    );
    case!(
        "transducer prompt hints",
        crate::OfflineFamily::Transducer,
        ErrorKind::UnsupportedCapability,
        "does not support engine-level prompt hints",
        prompt_hints,
        Some(with_hints())
    );
    case!(
        "qwen3 bias",
        crate::OfflineFamily::Qwen3Asr,
        ErrorKind::UnsupportedCapability,
        "does not support transducer bias",
        transducer_bias,
        Some(with_bias())
    );
    case!(
        "qwen3 language",
        crate::OfflineFamily::Qwen3Asr,
        ErrorKind::InvalidInput,
        "does not accept a language override",
        language,
        Some("zh".into())
    );
    case!(
        "funasr bias",
        crate::OfflineFamily::FunAsrNano,
        ErrorKind::UnsupportedCapability,
        "does not support transducer bias",
        transducer_bias,
        Some(with_bias())
    );
    case!(
        "funasr language",
        crate::OfflineFamily::FunAsrNano,
        ErrorKind::InvalidInput,
        "does not accept a language override",
        language,
        Some("zh".into())
    );
    case!(
        "fire-red aed bias",
        crate::OfflineFamily::FireRedAsrAed,
        ErrorKind::UnsupportedCapability,
        "does not support speech hints or transducer bias",
        transducer_bias,
        Some(with_bias())
    );
    case!(
        "fire-red aed prompt hints",
        crate::OfflineFamily::FireRedAsrAed,
        ErrorKind::UnsupportedCapability,
        "does not support speech hints or transducer bias",
        prompt_hints,
        Some(with_hints())
    );
    case!(
        "fire-red aed language",
        crate::OfflineFamily::FireRedAsrAed,
        ErrorKind::InvalidInput,
        "does not accept a language override",
        language,
        Some("zh".into())
    );
    case!(
        "fire-red ctc bias",
        crate::OfflineFamily::FireRedAsrCtc,
        ErrorKind::UnsupportedCapability,
        "does not support speech hints or transducer bias",
        transducer_bias,
        Some(with_bias())
    );
    case!(
        "fire-red ctc prompt hints",
        crate::OfflineFamily::FireRedAsrCtc,
        ErrorKind::UnsupportedCapability,
        "does not support speech hints or transducer bias",
        prompt_hints,
        Some(with_hints())
    );
    case!(
        "fire-red ctc language",
        crate::OfflineFamily::FireRedAsrCtc,
        ErrorKind::InvalidInput,
        "does not accept a language override",
        language,
        Some("zh".into())
    );

    // Layout rules fire after capability rules on a bare directory:
    // the finders reject before native init and both engines agree.
    for (name, family, missing) in [
        (
            "fire-red aed layout",
            crate::OfflineFamily::FireRedAsrAed,
            "missing encoder*.onnx",
        ),
        (
            "fire-red ctc layout",
            crate::OfflineFamily::FireRedAsrCtc,
            "missing model*.onnx",
        ),
    ] {
        let config = crate::OfflineConfig::new(bare.path(), family, vad.clone());
        let error = validate(&EngineConfig::Offline(config.clone())).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidModel, "{name}: {error}");
        assert!(error.message.contains(missing), "{name}: {error}");
        assert_precheck_matches_prepare(EngineConfig::Offline(config));
    }

    // CTC 与 SenseVoice/Paraformer 共享矛盾守卫：带 <|zh|> 标记的 tokens
    // 配成 FireRedAsrCtc 必须被 validate（而非仅 prepare）拒绝。
    let (marked_guard, marked_path) = flat_dir("foo 1\n<|zh|> 2\n");
    let config = crate::OfflineConfig::new(
        &marked_path,
        crate::OfflineFamily::FireRedAsrCtc,
        vad.clone(),
    );
    let error = validate(&EngineConfig::Offline(config.clone())).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(
        error.message.contains("SenseVoice language markers"),
        "{error}"
    );
    assert_precheck_matches_prepare(EngineConfig::Offline(config));
    drop(marked_guard);

    // 无标记的完整 flat 目录配成 FireRedAsrCtc 应当通过预检（防误拒）。
    let (flat_guard, flat_path) = flat_dir("foo 1\nbar 2\n");
    let config =
        crate::OfflineConfig::new(&flat_path, crate::OfflineFamily::FireRedAsrCtc, vad.clone());
    validate(&EngineConfig::Offline(config)).unwrap();
    drop(flat_guard);

    // Prompt rendering rules fire after the family layout resolves, so
    // these fixtures must be complete directories.
    let (qwen3_guard, qwen3_path) = qwen3_dir();
    let mut config =
        crate::OfflineConfig::new(&qwen3_path, crate::OfflineFamily::Qwen3Asr, vad.clone());
    config.prompt_hints = Some(crate::SpeechHints::new(vec!["张三,李四".into()]));
    let error = validate(&EngineConfig::Offline(config.clone())).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert!(error.message.contains("','"), "{error}");
    assert_precheck_matches_prepare(EngineConfig::Offline(config));
    drop(qwen3_guard);

    let (funasr_guard, funasr_path) = funasr_dir();
    let mut config = crate::OfflineConfig::new(&funasr_path, crate::OfflineFamily::FunAsrNano, vad);
    config.prompt_hints = Some(crate::SpeechHints::new(vec!["沪；语".into()]));
    let error = validate(&EngineConfig::Offline(config.clone())).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert!(error.message.contains('；'), "{error}");
    assert_precheck_matches_prepare(EngineConfig::Offline(config));
    drop(funasr_guard);
}
