use super::*;
#[test]
fn sensevoice_marker_after_twenty_thousand_tokens_is_detected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokens.txt");
    std::fs::write(&path, format!("{}<|zh|> 24884\n", "word 1\n".repeat(24884))).unwrap();
    assert!(is_sense_voice_tokens(&path).unwrap());
}

/// 家族预检四组合：标记存在只在"配成非 SenseVoice"时拒绝；
/// 标记缺失（Fun-ASR-Nano 等 markerless 变体的形态）一律放行。
#[cfg(feature = "vad-silero")]
#[test]
fn family_precheck_is_one_way_evidence() {
    use crate::OfflineFamily;
    let dir = tempfile::tempdir().unwrap();
    let marked = dir.path().join("marked.txt");
    std::fs::write(&marked, "foo 1\n<|zh|> 2\n<|itn|> 3\n").unwrap();
    let markerless = dir.path().join("markerless.txt");
    // Fun-ASR-Nano 形态：base64 字节级词表 + 尾部 <blk>，无任何语言标记
    std::fs::write(&markerless, "IQ== 1\nJg== 2\nPGJsaz4= 60514\n").unwrap();

    ensure_family_not_contradicted(OfflineFamily::SenseVoice, &marked).unwrap();
    ensure_family_not_contradicted(OfflineFamily::SenseVoice, &markerless).unwrap();
    ensure_family_not_contradicted(OfflineFamily::Paraformer, &markerless).unwrap();
    let err = ensure_family_not_contradicted(OfflineFamily::Paraformer, &marked).unwrap_err();
    assert!(err.contains("SenseVoice language markers"), "{err}");
    // 读取失败对两个家族都必须暴露，不得被 SenseVoice 分支短路吞掉。
    let invalid_utf8 = dir.path().join("invalid.txt");
    std::fs::write(&invalid_utf8, [0xFF, 0xFE]).unwrap();
    assert!(ensure_family_not_contradicted(OfflineFamily::SenseVoice, &invalid_utf8).is_err());
    assert!(ensure_family_not_contradicted(OfflineFamily::Paraformer, &invalid_utf8).is_err());
}
#[test]
fn directories_and_empty_models_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("tokens.txt"), "token 1").unwrap();
    std::fs::create_dir(dir.path().join("model.onnx")).unwrap();
    assert!(find_offline_model_files(dir.path()).is_err());
    std::fs::write(dir.path().join("model.int8.onnx"), []).unwrap();
    assert!(find_offline_model_files(dir.path()).is_err());
}
#[test]
fn equally_preferred_candidates_are_rejected_with_safe_names() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("encoder-a.int8.onnx"), [0u8; 8]).unwrap();
    std::fs::write(dir.path().join("encoder-b.int8.onnx"), [0u8; 8]).unwrap();
    std::fs::write(dir.path().join("decoder.onnx"), [0u8; 8]).unwrap();
    std::fs::write(dir.path().join("joiner.int8.onnx"), [0u8; 8]).unwrap();
    std::fs::write(dir.path().join("tokens.txt"), "token 1\n").unwrap();

    let error = find_model_files(dir.path()).unwrap_err();
    assert!(error.contains("encoder-a.int8.onnx"), "{error}");
    assert!(error.contains("encoder-b.int8.onnx"), "{error}");
    assert!(error.contains("keep exactly one"), "{error}");
    assert!(
        !error.contains(&dir.path().display().to_string()),
        "{error}"
    );
}
#[cfg(feature = "punct-sherpa")]
#[test]
fn punct_model_family_and_int8_preference_are_detected() {
    // CT-Transformer: model*.onnx only, int8 preferred.
    let ct = tempfile::tempdir().unwrap();
    std::fs::write(ct.path().join("model.onnx"), [0u8; 8]).unwrap();
    std::fs::write(ct.path().join("model.int8.onnx"), [0u8; 8]).unwrap();
    std::fs::write(ct.path().join("tokens.json"), "{}").unwrap();
    let files = find_punct_model_files(ct.path()).unwrap();
    assert!(files.model.ends_with("model.int8.onnx"));
    assert!(files.vocab.is_none());
    // CNN-BiLSTM: bpe.vocab selects the English online family.
    let en = tempfile::tempdir().unwrap();
    std::fs::write(en.path().join("model.int8.onnx"), [0u8; 8]).unwrap();
    std::fs::write(en.path().join("bpe.vocab"), "a 0\n").unwrap();
    let files = find_punct_model_files(en.path()).unwrap();
    assert!(files.vocab.is_some());
    // Archives unpack under a single child directory.
    let wrapped = tempfile::tempdir().unwrap();
    let child = wrapped.path().join("sherpa-onnx-punct-x");
    std::fs::create_dir(&child).unwrap();
    std::fs::write(child.join("model.int8.onnx"), [0u8; 8]).unwrap();
    let files = find_punct_model_files(wrapped.path()).unwrap();
    assert!(files.model.starts_with(&child));
    // An empty vocabulary is ambiguous and must not silently switch families.
    let empty_vocab = tempfile::tempdir().unwrap();
    std::fs::write(empty_vocab.path().join("model.int8.onnx"), [0u8; 8]).unwrap();
    std::fs::write(empty_vocab.path().join("bpe.vocab"), []).unwrap();
    let error = find_punct_model_files(empty_vocab.path()).unwrap_err();
    assert!(error.contains("bpe.vocab is empty"), "{error}");
    assert!(find_punct_model_files(tempfile::tempdir().unwrap().path()).is_err());
}
#[cfg(feature = "backend-sherpa")]
#[test]
fn hotword_modeling_unit_is_detected_from_bpe_vocab_and_tokens() {
    let files = |dir: &Path, bpe_vocab: Option<PathBuf>| ModelFiles {
        encoder: dir.join("encoder.onnx"),
        decoder: dir.join("decoder.onnx"),
        joiner: dir.join("joiner.onnx"),
        tokens: dir.join("tokens.txt"),
        bpe_vocab,
    };
    // 双语 zh-en：bpe.vocab + tokens 含 CJK → cjkchar+bpe
    let bilingual = tempfile::tempdir().unwrap();
    std::fs::write(bilingual.path().join("tokens.txt"), "中 1\n▁hello 2\n").unwrap();
    std::fs::write(bilingual.path().join("bpe.vocab"), "▁hello 0\n").unwrap();
    let (unit, vocab) = detect_hotword_modeling_unit(&files(
        bilingual.path(),
        find_bpe_vocab(bilingual.path()).unwrap(),
    ));
    assert_eq!(unit, "cjkchar+bpe");
    assert_eq!(
        vocab.as_deref(),
        Some(bilingual.path().join("bpe.vocab").as_path())
    );
    // 纯英文：bpe.vocab + 纯 ASCII tokens → bpe
    let english = tempfile::tempdir().unwrap();
    std::fs::write(english.path().join("tokens.txt"), "▁hello 1\n").unwrap();
    std::fs::write(english.path().join("bpe.vocab"), "▁hello 0\n").unwrap();
    let (unit, _) = detect_hotword_modeling_unit(&files(
        english.path(),
        find_bpe_vocab(english.path()).unwrap(),
    ));
    assert_eq!(unit, "bpe");
    // 中文模型（无 bpe.vocab，如 conformer-zh）→ cjkchar
    let chinese = tempfile::tempdir().unwrap();
    std::fs::write(chinese.path().join("tokens.txt"), "中 1\n").unwrap();
    let (unit, vocab) = detect_hotword_modeling_unit(&files(
        chinese.path(),
        find_bpe_vocab(chinese.path()).unwrap(),
    ));
    assert_eq!(unit, "cjkchar");
    assert_eq!(vocab, None);
    // 空 bpe.vocab 明确拒绝，不能静默改用 cjkchar。
    let empty = tempfile::tempdir().unwrap();
    std::fs::write(empty.path().join("tokens.txt"), "中 1\n").unwrap();
    std::fs::write(empty.path().join("bpe.vocab"), []).unwrap();
    assert!(find_bpe_vocab(empty.path()).is_err());
    // 扩展区与兼容表意区的汉字也按 CJK 判定（含罕见字的 tokens 不再误判为纯 bpe）。
    assert!(is_cjk_char('\u{20000}')); // 扩展 B
    assert!(is_cjk_char('\u{2A700}')); // 扩展 C
    assert!(is_cjk_char('\u{2CEB0}')); // 扩展 F
    assert!(is_cjk_char('\u{31350}')); // 扩展 H
    assert!(is_cjk_char('\u{F900}')); // 兼容表意
    assert!(is_cjk_char('\u{2F800}')); // 兼容表意补充
    assert!(!is_cjk_char('a'));
    assert!(!is_cjk_char('。')); // CJK 标点不属于表意字符
    let rare = tempfile::tempdir().unwrap();
    // U+20000 𠀀（扩展 B 区首字）。
    std::fs::write(rare.path().join("tokens.txt"), "𠀀 1\n").unwrap();
    std::fs::write(rare.path().join("bpe.vocab"), "▁hello 0\n").unwrap();
    let (unit, _) =
        detect_hotword_modeling_unit(&files(rare.path(), find_bpe_vocab(rare.path()).unwrap()));
    assert_eq!(unit, "cjkchar+bpe");
}
#[test]
fn qwen3_layout_needs_conv_frontend_encoder_decoder_and_tokenizer() {
    let dir = tempfile::tempdir().unwrap();
    assert!(find_qwen3_files(dir.path()).is_err());
    std::fs::write(dir.path().join("conv_frontend.onnx"), [0u8; 8]).unwrap();
    std::fs::write(dir.path().join("encoder.int8.onnx"), [0u8; 8]).unwrap();
    std::fs::write(dir.path().join("decoder.int8.onnx"), [0u8; 8]).unwrap();
    let tokenizer = dir.path().join("tokenizer");
    std::fs::create_dir(&tokenizer).unwrap();
    assert!(find_qwen3_files(dir.path()).is_err());
    std::fs::write(tokenizer.join("merges.txt"), "#version").unwrap();
    assert!(find_qwen3_files(dir.path()).is_err());
    std::fs::write(tokenizer.join("vocab.json"), "{}").unwrap();
    let files = find_qwen3_files(dir.path()).unwrap();
    assert!(files.conv_frontend.ends_with("conv_frontend.onnx"));
    assert!(files.encoder.ends_with("encoder.int8.onnx"));
    assert!(files.decoder.ends_with("decoder.int8.onnx"));
    assert_eq!(files.tokenizer, tokenizer);
    // 解压归档的单层子目录同样可探测。
    let wrapped = tempfile::tempdir().unwrap();
    let child = wrapped
        .path()
        .join("sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25");
    std::fs::create_dir(&child).unwrap();
    std::fs::write(child.join("conv_frontend.onnx"), [0u8; 8]).unwrap();
    std::fs::write(child.join("encoder.int8.onnx"), [0u8; 8]).unwrap();
    std::fs::write(child.join("decoder.int8.onnx"), [0u8; 8]).unwrap();
    std::fs::create_dir(child.join("tokenizer")).unwrap();
    std::fs::write(child.join("tokenizer/merges.txt"), "#version").unwrap();
    std::fs::write(child.join("tokenizer/vocab.json"), "{}").unwrap();
    assert!(find_qwen3_files(wrapped.path()).is_ok());
}
#[test]
fn funasr_nano_layout_needs_all_models_and_tokenizer_trio() {
    let dir = tempfile::tempdir().unwrap();
    assert!(find_funasr_nano_files(dir.path()).is_err());
    std::fs::write(dir.path().join("encoder_adaptor.int8.onnx"), [0u8; 8]).unwrap();
    std::fs::write(dir.path().join("llm.int8.onnx"), [0u8; 8]).unwrap();
    // int8 优先（与 fp32 并存时）。
    std::fs::write(dir.path().join("embedding.onnx"), [0u8; 8]).unwrap();
    std::fs::write(dir.path().join("embedding.int8.onnx"), [0u8; 8]).unwrap();
    // tokenizer 三件套缺一不可（上游 funasr-nano-tokenizer.cc 缺一即退出）。
    let tokenizer = dir.path().join("Qwen3-0.6B");
    std::fs::create_dir(&tokenizer).unwrap();
    assert!(find_funasr_nano_files(dir.path()).is_err());
    std::fs::write(tokenizer.join("vocab.json"), "{}").unwrap();
    assert!(find_funasr_nano_files(dir.path()).is_err());
    std::fs::write(tokenizer.join("merges.txt"), "#version").unwrap();
    assert!(find_funasr_nano_files(dir.path()).is_err());
    // 空文件视为缺失。
    std::fs::write(tokenizer.join("tokenizer.json"), []).unwrap();
    assert!(find_funasr_nano_files(dir.path()).is_err());
    std::fs::write(tokenizer.join("tokenizer.json"), "{}").unwrap();
    let files = find_funasr_nano_files(dir.path()).unwrap();
    assert!(files.encoder_adaptor.ends_with("encoder_adaptor.int8.onnx"));
    assert!(files.llm.ends_with("llm.int8.onnx"));
    assert!(files.embedding.ends_with("embedding.int8.onnx"));
    assert_eq!(files.tokenizer, tokenizer);
    // 第二个完整 tokenizer 目录即歧义：无法确定性选择。
    let second = dir.path().join("another-tokenizer");
    std::fs::create_dir(&second).unwrap();
    for name in ["vocab.json", "merges.txt", "tokenizer.json"] {
        std::fs::write(second.join(name), [0u8; 8]).unwrap();
    }
    assert!(find_funasr_nano_files(dir.path()).is_err());
}
#[test]
fn funasr_nano_archive_wrapped_in_single_child_is_detected() {
    let wrapped = tempfile::tempdir().unwrap();
    let child = wrapped
        .path()
        .join("sherpa-onnx-funasr-nano-int8-2025-12-30");
    std::fs::create_dir(&child).unwrap();
    for name in [
        "encoder_adaptor.int8.onnx",
        "llm.int8.onnx",
        "embedding.int8.onnx",
    ] {
        std::fs::write(child.join(name), [0u8; 8]).unwrap();
    }
    // test_wavs/ 等兄弟目录不干扰：tokenizer 探测按内容三件套判定。
    std::fs::create_dir(child.join("test_wavs")).unwrap();
    let tokenizer = child.join("Qwen3-0.6B");
    std::fs::create_dir(&tokenizer).unwrap();
    for name in ["vocab.json", "merges.txt", "tokenizer.json"] {
        std::fs::write(tokenizer.join(name), [0u8; 8]).unwrap();
    }
    let files = find_funasr_nano_files(wrapped.path()).unwrap();
    assert!(files.encoder_adaptor.starts_with(&child));
    assert!(files.tokenizer.starts_with(&child));
}
#[test]
fn fire_red_aed_layout_needs_encoder_decoder_and_tokens() {
    let dir = tempfile::tempdir().unwrap();
    assert!(find_fire_red_aed_files(dir.path()).is_err());
    std::fs::write(dir.path().join("encoder.int8.onnx"), [0u8; 8]).unwrap();
    assert!(find_fire_red_aed_files(dir.path()).is_err());
    // decoder 也是 int8 优先（与 fp32 并存时）。
    std::fs::write(dir.path().join("decoder.onnx"), [0u8; 8]).unwrap();
    std::fs::write(dir.path().join("decoder.int8.onnx"), [0u8; 8]).unwrap();
    assert!(find_fire_red_aed_files(dir.path()).is_err());
    std::fs::write(dir.path().join("tokens.txt"), "a 1\n").unwrap();
    let files = find_fire_red_aed_files(dir.path()).unwrap();
    assert!(files.encoder.ends_with("encoder.int8.onnx"));
    assert!(files.decoder.ends_with("decoder.int8.onnx"));
    assert!(files.tokens.ends_with("tokens.txt"));
    // 解压归档的单层子目录同样可探测。
    let wrapped = tempfile::tempdir().unwrap();
    let child = wrapped
        .path()
        .join("sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26");
    std::fs::create_dir(&child).unwrap();
    for name in ["encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"] {
        std::fs::write(child.join(name), [0u8; 8]).unwrap();
    }
    std::fs::create_dir(child.join("test_wavs")).unwrap();
    let files = find_fire_red_aed_files(wrapped.path()).unwrap();
    assert!(files.encoder.starts_with(&child));
    assert!(files.decoder.starts_with(&child));
    assert!(files.tokens.starts_with(&child));
}
#[test]
fn fire_red_aed_marker_requires_absent_joiner() {
    // encoder + decoder + tokens + joiner 是流式 transducer 布局：
    // FireRed 标记必须为假，家族判定交给各自 finder。
    let dir = tempfile::tempdir().unwrap();
    for name in [
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    ] {
        std::fs::write(dir.path().join(name), [0u8; 8]).unwrap();
    }
    assert!(!has_fire_red_aed_layout_marker(dir.path()));
    assert!(find_model_files(dir.path()).is_ok());
    // 无 joiner 的同形目录标记为真，且 transducer finder 因缺 joiner 失败。
    std::fs::remove_file(dir.path().join("joiner.int8.onnx")).unwrap();
    assert!(has_fire_red_aed_layout_marker(dir.path()));
    assert!(find_model_files(dir.path()).is_err());
    assert!(find_fire_red_aed_files(dir.path()).is_ok());
    // fp32 命名同样参与标记判定（不限于 .int8 字面名）。
    std::fs::write(dir.path().join("joiner.onnx"), [0u8; 8]).unwrap();
    assert!(!has_fire_red_aed_layout_marker(dir.path()));
    std::fs::remove_file(dir.path().join("joiner.onnx")).unwrap();
}
/// 反向错配守卫：把完整 transducer 目录（含 joiner）配成 FireRedAsrAed
/// 必须在原生初始化前指名拒绝——encoder/decoder/tokens 前缀在该布局
/// 全部命中，放行会让 sherpa 原生层以进程退出收场而非返回错误。
#[test]
fn fire_red_aed_rejects_transducer_layout_before_native_init() {
    for names in [
        [
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "joiner.int8.onnx",
            "tokens.txt",
        ],
        [
            // 官方 zipformer 归档的 epoch 后缀命名（前缀口径）。
            "encoder-epoch-99-avg-1.onnx",
            "decoder-epoch-99-avg-1.onnx",
            "joiner-epoch-99-avg-1.onnx",
            "tokens.txt",
        ],
        [
            // fp32 joiner 变体（如 joiner.fp16.onnx）同样排除。
            "encoder.onnx",
            "decoder.onnx",
            "joiner.fp16.onnx",
            "tokens.txt",
        ],
    ] {
        let dir = tempfile::tempdir().unwrap();
        for name in names {
            std::fs::write(dir.path().join(name), [0u8; 8]).unwrap();
        }
        assert!(!has_fire_red_aed_layout_marker(dir.path()));
        let error = find_fire_red_aed_files(dir.path()).unwrap_err();
        assert!(
            error.contains("joiner") && error.contains("transducer layout"),
            "{error}"
        );
    }
}
