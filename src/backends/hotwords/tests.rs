use super::*;
use crate::ErrorKind;

#[test]
fn bias_validation_and_rendering_are_transducer_specific() {
    let bias = TransducerBiasConfig {
        phrases: vec![
            BiasPhrase::new("语音识别"),
            BiasPhrase::scored("深度学习", 3.5),
        ],
        default_score: 2.0,
        modeling_unit: Some("cjkchar+bpe".into()),
    };
    validate_bias(&bias).unwrap();
    assert_eq!(render_bias(&bias).unwrap(), "语音识别/深度学习 :3.5");
    assert_eq!(
        merge(
            render_bias(&bias).as_ref(),
            Some(&SpeechHints::new(vec!["张三".into()])),
        )
        .unwrap()
        .as_deref(),
        Some("语音识别/深度学习 :3.5/张三")
    );
}

#[test]
fn invalid_transducer_syntax_and_limits_are_rejected() {
    for bad in ["", "语/音", "语:音", "语,音", "c#", "@name", "a\nb"] {
        assert!(validate_session_hints(&SpeechHints::new(vec![bad.into()])).is_err());
    }
    let too_many = SpeechHints::new(vec!["word".into(); MAX_PHRASES + 1]);
    assert!(validate_session_hints(&too_many).is_err());
    let mut bias = TransducerBiasConfig::new(Vec::new());
    bias.default_score = f32::NAN;
    assert!(validate_bias(&bias).is_err());
}

#[cfg(feature = "vad-silero")]
#[test]
fn prompt_hints_have_family_specific_validation() {
    let hints = SpeechHints::new(vec!["骨质疏松症患者".into(), "张三".into()]);
    assert_eq!(
        qwen3_prompt(&hints).unwrap().unwrap(),
        "骨质疏松症患者,张三"
    );
    assert_eq!(
        funasr_nano_prompt(&hints).unwrap().unwrap(),
        "骨质疏松症患者,张三"
    );
    assert!(funasr_nano_prompt(&SpeechHints::new(vec!["沪；语".into()])).is_err());
}

#[test]
fn merged_hints_exceeding_the_limit_are_rejected() {
    let bias = TransducerBiasConfig::new(
        (0..MAX_PHRASES)
            .map(|index| BiasPhrase::new(format!("热词{index}")))
            .collect::<Vec<_>>(),
    );
    validate_bias(&bias).unwrap();
    let defaults = render_bias(&bias).unwrap();
    let session = SpeechHints::new(vec!["补充热词".into()]);
    validate_session_hints(&session).unwrap();
    let error = merge(Some(&defaults), Some(&session)).unwrap_err();
    assert!(error
        .message
        .contains("engine and session speech hints combined"));
}

/// prepare_bias 只消费 tokens 与 bpe_vocab 两个路径，其余字段填占位值。
fn bias_files(dir: &std::path::Path, tokens: &str) -> ModelFiles {
    let tokens_path = dir.join("tokens.txt");
    std::fs::write(&tokens_path, tokens).unwrap();
    ModelFiles {
        encoder: dir.join("encoder.onnx"),
        decoder: dir.join("decoder.onnx"),
        joiner: dir.join("joiner.onnx"),
        tokens: tokens_path,
        bpe_vocab: None,
    }
}

#[test]
fn prepare_bias_pipeline_orders_syntax_unit_then_vocabulary() {
    // 语法非法的 bias 在进入建模单元/词表阶段之前先失败：tokens 本身
    // 是合法的 cjkchar 词表，报错只能是 bias 语法校验。
    let dir = tempfile::tempdir().unwrap();
    let files = bias_files(dir.path(), "语 1\n音 2\n");
    let syntax = TransducerBiasConfig::new(vec![BiasPhrase::new("语/音")]);
    let error = prepare_bias(&syntax, &files).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert!(error.message.contains('/'), "{error}");

    // 显式 bpe 建模单元但没有 bpe.vocab：resolve 阶段指名拒绝。
    let bpe_dir = tempfile::tempdir().unwrap();
    let bpe_files = bias_files(bpe_dir.path(), "hello 1\nworld 2\n");
    let mut bpe = TransducerBiasConfig::new(vec![BiasPhrase::new("hello")]);
    bpe.modeling_unit = Some("bpe".into());
    let error = prepare_bias(&bpe, &bpe_files).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(error.message.contains("requires bpe.vocab"), "{error}");

    // 合法配置产出预期的建模单元与词表：cjkchar、无 bpe.vocab，词表
    // 接受 tokens 内字符的短语、拒绝 OOV。
    let bias = TransducerBiasConfig::new(vec![BiasPhrase::new("语音")]);
    let (unit, bpe_vocab, vocabulary) = prepare_bias(&bias, &files).unwrap();
    assert_eq!(unit, "cjkchar");
    assert_eq!(bpe_vocab, None);
    vocabulary
        .validate_session(&SpeechHints::new(vec!["语音".into()]))
        .unwrap();
    assert!(vocabulary
        .validate_session(&SpeechHints::new(vec!["㮝".into()]))
        .is_err());
}

#[test]
fn model_vocabulary_checks_default_and_session_phrases() {
    let dir = tempfile::tempdir().unwrap();
    let tokens = dir.path().join("tokens.txt");
    std::fs::write(&tokens, "语 1\n音 2\n▁hello 3\n").unwrap();
    let invalid = TransducerBiasConfig::new(vec![BiasPhrase::new("语音识别")]);
    assert!(HotwordVocabulary::prepare(&invalid, "cjkchar", &tokens).is_err());
    let valid = TransducerBiasConfig::new(vec![BiasPhrase::new("语音")]);
    let vocabulary = HotwordVocabulary::prepare(&valid, "cjkchar", &tokens).unwrap();
    assert!(vocabulary
        .validate_session(&SpeechHints::new(vec!["语音".into()]))
        .is_ok());
    assert!(vocabulary
        .validate_session(&SpeechHints::new(vec!["语音识别".into()]))
        .is_err());
}
