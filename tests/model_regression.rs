#![cfg(feature = "backend-sherpa")]
use asr_core::*;
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
/// 大模型回归层开关：Qwen3-ASR、FunASR-Nano、FireRedASR2、sense-voice
/// nano 变体等未入 CI fixture 的模型测试由 `ASR_RUN_LARGE_MODEL_TEST=1`
/// 显式启用。守卫放在测试体内而非 `#[cfg]`/Cargo feature：这些测试在
/// 普通 `cargo test` 下仍参与编译与类型检查，未启用时打印一行 skipped
/// 即返回，workflow 无需维护 `--skip` 列表。
#[cfg(feature = "vad-silero")]
fn large_model_tier() -> bool {
    large_model_tier_from(std::env::var("ASR_RUN_LARGE_MODEL_TEST").ok().as_deref())
}
/// 纯函数接缝：开关语义在此固定并直接测试，不触碰进程级环境变量
/// （集成测试并行运行，set_var 会与守卫读变量竞争）。
/// 仅 `1` 或 `true` 启用：空值、`false`、`0` 等一律不启用，避免 CI 误
/// 注入空变量或用户设 `false` 期望关闭反而开启。
fn large_model_tier_from(value: Option<&str>) -> bool {
    matches!(value, Some("1" | "true"))
}
#[test]
fn large_model_tier_requires_explicit_enable() {
    assert!(!large_model_tier_from(None));
    assert!(!large_model_tier_from(Some("")));
    assert!(!large_model_tier_from(Some("0")));
    assert!(!large_model_tier_from(Some("false")));
    assert!(!large_model_tier_from(Some("enabled")));
    assert!(large_model_tier_from(Some("1")));
    assert!(large_model_tier_from(Some("true")));
}
fn prepare(config: EngineConfig) -> Result<Engine, AsrError> {
    Engine::prepare(config, EngineOptions::default())
}
/// Offline 回归测试的基线配置契约：语言覆盖、标点、热词、提示词
/// 默认全关，需要覆盖的字段用结构体更新语法在调用点叠加。
#[cfg(feature = "vad-silero")]
fn offline_config(dir: impl Into<PathBuf>, family: OfflineFamily, vad: VadConfig) -> OfflineConfig {
    OfflineConfig {
        model_dir: dir.into(),
        family,
        language: None,
        vad,
        punctuation: None,
        transducer_bias: None,
        prompt_hints: None,
        provider: asr_core::ExecutionProvider::Cpu,
        num_threads: DEFAULT_NUM_THREADS,
    }
}
fn start(engine: &Engine, audio: &AudioBuffer, hints: Option<SpeechHints>) -> Session {
    engine
        .start(SessionOptions {
            input: audio.spec,
            hints,
            ..Default::default()
        })
        .unwrap()
}
fn push_all(session: &Session, audio: &AudioBuffer, chunk_frames: usize) -> SessionOutcome {
    let input = session.input();
    for samples in audio.samples.chunks(chunk_frames) {
        input
            .push_wait(
                AudioChunk {
                    samples: samples.to_vec(),
                    spec: audio.spec,
                },
                Instant::now() + Duration::from_secs(60),
            )
            .unwrap();
    }
    session
        .finish(Instant::now() + Duration::from_secs(60))
        .unwrap()
}
fn run(engine: &Engine, audio: &AudioBuffer, chunk_frames: usize) -> SessionOutcome {
    push_all(&start(engine, audio, None), audio, chunk_frames)
}
fn run_with_hints(
    engine: &Engine,
    audio: &AudioBuffer,
    chunk_frames: usize,
    phrases: Vec<String>,
) -> SessionOutcome {
    push_all(
        &start(engine, audio, Some(SpeechHints::new(phrases))),
        audio,
        chunk_frames,
    )
}
/// 能力校验契约：会话级 hints 在不支持的后端必须快速失败；返回错误
/// 供调用点断言 kind/stage/message 等具体契约。
fn start_with_hints_must_fail(
    engine: &Engine,
    input: AudioSpec,
    phrase: &str,
    context: &str,
) -> AsrError {
    engine
        .start(SessionOptions {
            input,
            hints: Some(SpeechHints::new(vec![phrase.into()])),
            ..Default::default()
        })
        .err()
        .unwrap_or_else(|| panic!("{context}: expected start() to fail but it succeeded"))
}
/// ASR 行为不变式：纯静音经 VAD 过滤后不产生分句（streaming 家族
/// 则无语音活动）。
fn assert_silence_yields_no_segments(engine: &Engine) {
    assert!(run(
        engine,
        &AudioBuffer::mono(vec![0.0; 16000], 16000).unwrap(),
        1600,
    )
    .transcript
    .segments
    .is_empty());
}
/// 从基线转写提取热词：热词冒烟断言用它保证偏置不会破坏既有识别。
/// 中文转写无空白分隔，`split_whitespace().next()` 会把整句当词，整句
/// 超过 validate_bias 的 64 字符上限即报 InvalidInput，表象像产品 bug。
/// 因此跳过开头标点/空白后取第一段连续的非标点、非空白字符（至多
/// 4 个，与 Qwen3 用例的截标点、限长写法一致）：热词非空、远低于
/// 上限，且仍是基线转写的连续子串，`contains` 断言不会被截断位置
/// 撕裂（双语 zipformer 输出带空格的中英混排）。
fn hotword_from(text: &str) -> String {
    fn is_separator(c: char) -> bool {
        matches!(
            c,
            '，' | '。' | '！' | '？' | '、' | '；' | '：' | ',' | '.' | '!' | '?' | ';' | ':'
        ) || c.is_whitespace()
    }
    text.chars()
        .skip_while(|c| is_separator(*c))
        .take_while(|c| !is_separator(*c))
        .take(4)
        .collect()
}
#[test]
fn hotword_extraction_is_short_punctuation_free_and_contiguous() {
    // 中文整句无空白分隔：不得把整句当热词（超 64 字符上限会被
    // validate_bias 报 InvalidInput），截到前 4 个字符。
    let long_sentence = "这是一个远超六十四字符上限的中文整句样例。".repeat(5);
    assert_eq!(hotword_from(&long_sentence), "这是一个");
    // 标点不进热词，且截断不会跨过被删除的标点撕裂连续子串。
    assert_eq!(hotword_from("广西壮族自治区，爱吃柠檬"), "广西壮族");
    assert_eq!(hotword_from("，你好，世界！"), "你好");
    // 空白分隔的中英混排（双语 zipformer）：取首个空白前的字符。
    assert_eq!(
        hotword_from("昨天是 MONDAY礼拜二 THE DAY AFTER TOMORROW是星期三"),
        "昨天是"
    );
    // 纯标点或空转写返回空串，由调用方的非空断言拦截。
    assert_eq!(hotword_from("。，！"), "");
    assert_eq!(hotword_from(""), "");
}
#[test]
#[ignore = "requires ASR_STREAMING_MODEL pointing to the registered official model"]
fn streaming_chunk_boundaries_tail_and_silence() {
    let dir = PathBuf::from(std::env::var("ASR_STREAMING_MODEL").unwrap());
    let engine = prepare(EngineConfig::Streaming(StreamingConfig {
        model_dir: dir.clone(),
        punctuation: None,
        bias: None,
        provider: asr_core::ExecutionProvider::Cpu,
        // 非默认线程数：真实模型层确认参数端到端传到 sherpa-onnx 且识别正常。
        num_threads: 4,
    }))
    .unwrap();
    let mut audio = audio::read_wav_pcm16(dir.join("test_wavs/0.wav")).unwrap();
    if let Some(last) = audio.samples.iter().rposition(|v| v.abs() > 0.01) {
        audio
            .samples
            .truncate((last + 320).min(audio.samples.len()));
    }
    let a = run(&engine, &audio, 1600);
    let b = run(&engine, &audio, 317);
    assert!(!a.transcript.text().is_empty());
    assert_eq!(a.transcript.text(), b.transcript.text());
    audio.samples.extend(vec![0.0; 16000]);
    let padded = run(&engine, &audio, 1600);
    assert_eq!(a.transcript.text(), padded.transcript.text());
    eprintln!("streaming: {}", a.transcript.text());
    assert_silence_yields_no_segments(&engine);
}
#[test]
#[ignore = "requires ASR_STREAMING_MODEL; session hotwords must fail fast on a greedy engine"]
fn session_hotwords_fail_without_engine_hotwords() {
    let dir = PathBuf::from(std::env::var("ASR_STREAMING_MODEL").unwrap());
    let engine = prepare(EngineConfig::Streaming(StreamingConfig {
        model_dir: dir.clone(),
        punctuation: None,
        bias: None,
        provider: asr_core::ExecutionProvider::Cpu,
        num_threads: DEFAULT_NUM_THREADS,
    }))
    .unwrap();
    assert!(!engine.capabilities().supports_session_hints);
    let audio = audio::read_wav_pcm16(dir.join("test_wavs/0.wav")).unwrap();
    let error = start_with_hints_must_fail(
        &engine,
        audio.spec,
        "hello",
        "session hotwords must fail fast",
    );
    assert_eq!(error.kind, ErrorKind::UnsupportedCapability);
    assert_eq!(error.stage, "start");
}
#[test]
#[ignore = "requires ASR_STREAMING_BILINGUAL_MODEL (zh-en zipformer whose archive ships bpe.vocab)"]
fn streaming_hotwords_engine_level_and_session_override() {
    let dir = PathBuf::from(std::env::var("ASR_STREAMING_BILINGUAL_MODEL").unwrap());
    // 空词表仅启用热词机制（beam search + 建模单元），词表走会话级。
    let engine = prepare(EngineConfig::Streaming(StreamingConfig {
        model_dir: dir.clone(),
        punctuation: None,
        bias: Some(TransducerBiasConfig::new(Vec::new())),
        provider: asr_core::ExecutionProvider::Cpu,
        num_threads: DEFAULT_NUM_THREADS,
    }))
    .unwrap();
    assert!(engine.capabilities().supports_session_hints);
    let audio = audio::read_wav_pcm16(dir.join("test_wavs/0.wav")).unwrap();
    let baseline = run(&engine, &audio, 1600);
    let word = hotword_from(&baseline.transcript.text());
    assert!(!word.is_empty());
    let boosted = run_with_hints(&engine, &audio, 1600, vec![word.clone()]);
    eprintln!("streaming baseline: {}", baseline.transcript.text());
    eprintln!(
        "streaming session hotword '{word}': {}",
        boosted.transcript.text()
    );
    assert!(boosted.transcript.text().contains(&word));
    // 会话级词表对照引擎 prepare 时加载的 tokens 表校验：OOV 字符在
    // start 报 InvalidModel，而不是静默编出错位热词路径。双语模型为
    // cjkchar+bpe，仅 ASCII 字母经 BPE；找一个必不在 tokens 表的汉字
    // （生僻字 U+3B9D）即可触发。
    let error = start_with_hints_must_fail(
        &engine,
        audio.spec,
        "㮝",
        "session hotword with OOV characters must fail at start",
    );
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(error.message.contains("㮝"), "{error}");

    // 引擎级词表对所有会话生效，且可与会话级词表叠加。
    let engine = prepare(EngineConfig::Streaming(StreamingConfig {
        model_dir: dir.clone(),
        punctuation: None,
        bias: Some(TransducerBiasConfig::new(vec![BiasPhrase::new(
            word.clone(),
        )])),
        provider: asr_core::ExecutionProvider::Cpu,
        num_threads: DEFAULT_NUM_THREADS,
    }))
    .unwrap();
    let baked = run(&engine, &audio, 1600);
    eprintln!(
        "streaming engine hotword '{word}': {}",
        baked.transcript.text()
    );
    assert!(baked.transcript.text().contains(&word));
    let extra = run_with_hints(&engine, &audio, 1600, vec![word.clone()]);
    assert!(extra.transcript.text().contains(&word));
}
#[cfg(feature = "vad-silero")]
#[test]
#[ignore = "requires ASR_SENSEVOICE_MODEL and ASR_VAD_MODEL"]
fn sensevoice_official_samples_and_silence() {
    let dir = PathBuf::from(std::env::var("ASR_SENSEVOICE_MODEL").unwrap());
    let engine = prepare(EngineConfig::Offline(offline_config(
        dir.clone(),
        OfflineFamily::SenseVoice,
        VadConfig::new(std::env::var("ASR_VAD_MODEL").unwrap()),
    )))
    .unwrap();
    for name in ["zh", "en"] {
        let audio = audio::read_wav_pcm16(dir.join(format!("test_wavs/{name}.wav"))).unwrap();
        let result = run(&engine, &audio, 1600);
        assert!(!result.transcript.text().is_empty());
        assert_eq!(result.received_frames, result.processed_frames);
        eprintln!("SenseVoice {name}: {}", result.transcript.text());
    }
    assert_silence_yields_no_segments(&engine);
}
#[cfg(feature = "vad-silero")]
#[test]
#[ignore = "requires ASR_PARAFORMER_MODEL, ASR_PARAFORMER_WAV and ASR_VAD_MODEL"]
fn paraformer_official_sample() {
    let engine = prepare(EngineConfig::Offline(offline_config(
        std::env::var("ASR_PARAFORMER_MODEL").unwrap(),
        OfflineFamily::Paraformer,
        VadConfig::new(std::env::var("ASR_VAD_MODEL").unwrap()),
    )))
    .unwrap();
    let audio = audio::read_wav_pcm16(std::env::var("ASR_PARAFORMER_WAV").unwrap()).unwrap();
    let result = run(&engine, &audio, 1600);
    assert!(!result.transcript.text().is_empty());
    eprintln!("Paraformer: {}", result.transcript.text());
}
#[cfg(feature = "vad-silero")]
#[test]
#[ignore = "requires ASR_OFFLINE_TRANSDUCER_MODEL (encoder/decoder/joiner + tokens) and ASR_VAD_MODEL"]
fn offline_transducer_hotwords() {
    let dir = PathBuf::from(std::env::var("ASR_OFFLINE_TRANSDUCER_MODEL").unwrap());
    let vad = VadConfig::new(std::env::var("ASR_VAD_MODEL").unwrap());
    // 空词表仅启用热词机制（beam search + 建模单元）。
    let engine = prepare(EngineConfig::Offline(OfflineConfig {
        transducer_bias: Some(TransducerBiasConfig::new(Vec::new())),
        ..offline_config(dir.clone(), OfflineFamily::Transducer, vad.clone())
    }))
    .unwrap();
    assert!(engine.capabilities().supports_session_hints);
    let wav = if dir.join("test_wavs/0.wav").exists() {
        dir.join("test_wavs/0.wav")
    } else {
        dir.join("test_wavs/1.wav")
    };
    let audio = audio::read_wav_pcm16(wav).unwrap();
    // 会话级 OOV 校验：conformer-zh 为 cjkchar 单元，生僻字（U+3B9D）
    // 不在 tokens 表，start 必须报 InvalidModel。
    let error = start_with_hints_must_fail(
        &engine,
        audio.spec,
        "㮝",
        "session hotword with OOV characters must fail at start",
    );
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    let baseline = run(&engine, &audio, 1600);
    let word = hotword_from(&baseline.transcript.text());
    assert!(!word.is_empty());
    let engine = prepare(EngineConfig::Offline(OfflineConfig {
        transducer_bias: Some(TransducerBiasConfig::new(vec![BiasPhrase::new(
            word.clone(),
        )])),
        ..offline_config(dir.clone(), OfflineFamily::Transducer, vad.clone())
    }))
    .unwrap();
    let boosted = run(&engine, &audio, 1600);
    eprintln!(
        "offline transducer baseline: {}",
        baseline.transcript.text()
    );
    eprintln!(
        "offline transducer hotword '{word}': {}",
        boosted.transcript.text()
    );
    assert!(boosted.transcript.text().contains(&word));
    // 无热词配置：greedy 解码的 plain transducer 家族也要可用。
    let plain = prepare(EngineConfig::Offline(offline_config(
        dir.clone(),
        OfflineFamily::Transducer,
        vad,
    )))
    .unwrap();
    assert!(!plain.capabilities().supports_session_hints);
    let error = start_with_hints_must_fail(
        &plain,
        audio.spec,
        "hello",
        "session hotwords must fail on a plain transducer engine",
    );
    assert_eq!(error.kind, ErrorKind::UnsupportedCapability);
    let transcript = run(&plain, &audio, 1600).transcript.text();
    eprintln!("offline transducer plain: {transcript}");
    assert!(!transcript.is_empty());
}
#[cfg(feature = "vad-silero")]
#[test]
#[ignore = "large-model tier (ASR_RUN_LARGE_MODEL_TEST=1); requires ASR_QWEN3_MODEL (conv_frontend/encoder/decoder + tokenizer dir) and ASR_VAD_MODEL"]
fn qwen3_hotwords_and_language_rejection() {
    if !large_model_tier() {
        eprintln!("skipped: large-model tier (ASR_RUN_LARGE_MODEL_TEST unset)");
        return;
    }
    let dir = PathBuf::from(std::env::var("ASR_QWEN3_MODEL").unwrap());
    let vad = VadConfig::new(std::env::var("ASR_VAD_MODEL").unwrap());
    // Qwen3-ASR 拒绝语言覆盖（多语言自动识别）。
    let Err(error) = prepare(EngineConfig::Offline(OfflineConfig {
        language: Some("zh".into()),
        ..offline_config(dir.clone(), OfflineFamily::Qwen3Asr, vad.clone())
    })) else {
        panic!("Qwen3Asr must reject a language override");
    };
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    let wav = if dir.join("test_wavs/raokouling.wav").exists() {
        dir.join("test_wavs/raokouling.wav")
    } else {
        dir.join("test_wavs/0.wav")
    };
    let audio = audio::read_wav_pcm16(wav).unwrap();
    let plain = prepare(EngineConfig::Offline(offline_config(
        dir.clone(),
        OfflineFamily::Qwen3Asr,
        vad.clone(),
    )))
    .unwrap();
    let baseline = run(&plain, &audio, 1600);
    // 中文输出无空格且带标点：截标点、限长取词作热词，完整长句热词
    // 会挤爆 Qwen3 的提示词预算（max_total_len 警告）。
    let word = hotword_from(&baseline.transcript.text());
    assert!(!word.is_empty());
    // Qwen3 热词为引擎级（逗号串），会话级必须失败。
    assert!(!plain.capabilities().supports_session_hints);
    let error = start_with_hints_must_fail(
        &plain,
        audio.spec,
        &word,
        "session hotwords must fail on Qwen3Asr",
    );
    assert_eq!(error.kind, ErrorKind::UnsupportedCapability);
    // 官方演示用例：raokouling.wav 把"骨质疏松症患者"识别成近音乱码，
    // 引擎级热词应当纠正它——这是热词偏置真正生效的断言。
    let engine = prepare(EngineConfig::Offline(OfflineConfig {
        prompt_hints: Some(SpeechHints::new(vec![
            word.clone(),
            "骨质疏松症患者".into(),
        ])),
        ..offline_config(dir.clone(), OfflineFamily::Qwen3Asr, vad)
    }))
    .unwrap();
    let boosted = run(&engine, &audio, 1600);
    let boosted_text = boosted.transcript.text();
    eprintln!("qwen3 baseline: {}", baseline.transcript.text());
    eprintln!("qwen3 hotword '{word}': {boosted_text}");
    assert!(boosted_text.contains(&word));
    assert!(
        boosted_text.contains("骨质疏松"),
        "hotword should correct the near-homophone garble: {boosted_text}"
    );
}
#[cfg(feature = "vad-silero")]
#[test]
#[ignore = "large-model tier (ASR_RUN_LARGE_MODEL_TEST=1); requires ASR_FUNASR_NANO_MODEL (encoder_adaptor/llm/embedding + tokenizer dir) and ASR_VAD_MODEL"]
fn funasr_nano_hotwords_and_language_rejection() {
    if !large_model_tier() {
        eprintln!("skipped: large-model tier (ASR_RUN_LARGE_MODEL_TEST unset)");
        return;
    }
    let dir = PathBuf::from(std::env::var("ASR_FUNASR_NANO_MODEL").unwrap());
    let vad = VadConfig::new(std::env::var("ASR_VAD_MODEL").unwrap());
    // FunASR-Nano 拒绝语言覆盖（中英日自动识别）。
    let Err(error) = prepare(EngineConfig::Offline(OfflineConfig {
        language: Some("zh".into()),
        ..offline_config(dir.clone(), OfflineFamily::FunAsrNano, vad.clone())
    })) else {
        panic!("FunAsrNano must reject a language override");
    };
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    // 该归档的样本为命名文件（dia_* 方言、ja、lyrics、rag_*），无 0/1.wav；
    // dia_sh.wav（上海话）输出中文基线。其他家族归档仍命中 0.wav/1.wav。
    let wav = ["0.wav", "1.wav", "dia_sh.wav"]
        .iter()
        .map(|name| dir.join("test_wavs").join(name))
        .find(|p| p.is_file())
        .expect("test_wavs sample");
    let audio = audio::read_wav_pcm16(wav).unwrap();
    let plain = prepare(EngineConfig::Offline(offline_config(
        dir.clone(),
        OfflineFamily::FunAsrNano,
        vad.clone(),
    )))
    .unwrap();
    let baseline = run(&plain, &audio, 1600);
    assert_eq!(baseline.received_frames, baseline.processed_frames);
    // 中文输出可能带标点：截标点、限长取词作热词（完整长句会挤爆
    // 提示词预算）。
    let word = hotword_from(&baseline.transcript.text());
    assert!(!word.is_empty(), "baseline must be non-empty");
    // FunASR-Nano 热词为引擎级（逗号串注入用户提示），会话级必须失败。
    assert!(!plain.capabilities().supports_session_hints);
    let error = start_with_hints_must_fail(
        &plain,
        audio.spec,
        &word,
        "session hotwords must fail on FunAsrNano",
    );
    assert_eq!(error.kind, ErrorKind::UnsupportedCapability);
    let engine = prepare(EngineConfig::Offline(OfflineConfig {
        prompt_hints: Some(SpeechHints::new(vec![word.clone()])),
        ..offline_config(dir.clone(), OfflineFamily::FunAsrNano, vad)
    }))
    .unwrap();
    let boosted = run(&engine, &audio, 1600);
    let boosted_text = boosted.transcript.text();
    eprintln!("funasr-nano baseline: {}", baseline.transcript.text());
    eprintln!("funasr-nano hotword '{word}': {boosted_text}");
    // 冒烟级断言：word 取自基线转写，contains 基本恒真，只能证明热词路径
    // 端到端可跑通。qwen3 测试那样"无热词乱码、有热词修正"的强断言，需要
    // 先在满足内存条件且非 AVX2-only 的机器上找出真实 garble 样本（见
    // providers.md 的 FunASR-Nano 量化数值警示）。
    assert!(boosted_text.contains(&word));
    // 纯静音经 VAD 过滤后不产生分句（上游 #3921 修复静音幻觉后）。
    assert_silence_yields_no_segments(&plain);
}
#[cfg(feature = "vad-silero")]
#[test]
#[ignore = "large-model tier (ASR_RUN_LARGE_MODEL_TEST=1); requires ASR_SENSEVOICE_NANO_MODEL (sense-voice-funasr-nano 单文件目录) \
and ASR_SENSEVOICE_NANO_WAV and ASR_VAD_MODEL"]
fn sensevoice_markerless_funasr_nano_variant_loads() {
    if !large_model_tier() {
        eprintln!("skipped: large-model tier (ASR_RUN_LARGE_MODEL_TEST unset)");
        return;
    }
    // `sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17`，
    // base64 字节级词表、无 <|zh|> 语言标记的 SenseVoice 系变体，配 SenseVoice
    // 家族加载（tokens 无标记不再构成拒绝证据）。
    let dir = PathBuf::from(std::env::var("ASR_SENSEVOICE_NANO_MODEL").unwrap());
    let engine = prepare(EngineConfig::Offline(offline_config(
        dir,
        OfflineFamily::SenseVoice,
        VadConfig::new(std::env::var("ASR_VAD_MODEL").unwrap()),
    )))
    .unwrap();
    let audio = audio::read_wav_pcm16(std::env::var("ASR_SENSEVOICE_NANO_WAV").unwrap()).unwrap();
    let result = run(&engine, &audio, 1600);
    assert_eq!(result.received_frames, result.processed_frames);
    eprintln!("sense-voice funasr-nano: {}", result.transcript.text());
    assert!(
        !result.transcript.text().is_empty(),
        "transcript must be non-empty"
    );
}
#[cfg(feature = "punct-sherpa")]
#[test]
#[ignore = "requires ASR_STREAMING_MODEL and ASR_PUNCT_MODEL (ct-transformer zh-en)"]
fn streaming_punctuation_ct_transformer() {
    let dir = PathBuf::from(std::env::var("ASR_STREAMING_MODEL").unwrap());
    let raw_engine = prepare(EngineConfig::Streaming(StreamingConfig::new(dir.clone()))).unwrap();
    let engine = prepare(EngineConfig::Streaming(StreamingConfig {
        model_dir: dir.clone(),
        punctuation: Some(PunctConfig::new(std::env::var("ASR_PUNCT_MODEL").unwrap())),
        bias: None,
        provider: asr_core::ExecutionProvider::Cpu,
        num_threads: DEFAULT_NUM_THREADS,
    }))
    .unwrap();
    assert!(engine.capabilities().punctuation);
    let audio = audio::read_wav_pcm16(dir.join("test_wavs/0.wav")).unwrap();
    let raw = run(&raw_engine, &audio, 1600);
    let punctuated = run(&engine, &audio, 1600);
    let raw = raw.transcript.text();
    let punctuated = punctuated.transcript.text();
    eprintln!("streaming ct-transformer raw: {raw}");
    eprintln!("streaming ct-transformer punctuated: {punctuated}");
    assert!(!raw.chars().any(|c| "，。？！,.?!".contains(c)));
    // The zh-en model emits CJK full-width punctuation even for English input.
    assert!(punctuated.chars().any(|c| "，。？！".contains(c)));
    assert_ne!(raw, punctuated);
}
#[cfg(feature = "punct-sherpa")]
#[test]
#[ignore = "requires ASR_STREAMING_MODEL (english zipformer) and ASR_PUNCT_EN_MODEL (cnn-bilstm)"]
fn streaming_punctuation_en_cnn_bilstm() {
    let dir = PathBuf::from(std::env::var("ASR_STREAMING_MODEL").unwrap());
    let raw_engine = prepare(EngineConfig::Streaming(StreamingConfig::new(dir.clone()))).unwrap();
    let engine = prepare(EngineConfig::Streaming(StreamingConfig {
        model_dir: dir.clone(),
        punctuation: Some(PunctConfig::new(
            std::env::var("ASR_PUNCT_EN_MODEL").unwrap(),
        )),
        bias: None,
        provider: asr_core::ExecutionProvider::Cpu,
        num_threads: DEFAULT_NUM_THREADS,
    }))
    .unwrap();
    assert!(engine.capabilities().punctuation);
    let audio = audio::read_wav_pcm16(dir.join("test_wavs/0.wav")).unwrap();
    let raw = run(&raw_engine, &audio, 1600);
    let punctuated = run(&engine, &audio, 1600);
    let raw = raw.transcript.text();
    let punctuated = punctuated.transcript.text();
    eprintln!("streaming cnn-bilstm raw: {raw}");
    eprintln!("streaming cnn-bilstm punctuated: {punctuated}");
    // The streaming zipformer emits uppercase text. The CNN-BiLSTM model
    // leaves fully uppercase input unchanged (its BPE vocabulary only knows
    // lowercase words), so lowercase before punctuating: half-width
    // punctuation and capitalization are then restored.
    let normalized = punctuated.to_lowercase();
    assert!(!raw.chars().any(|c| "，。？！,.?!".contains(c)));
    assert!(normalized.chars().any(|c| ",.?!".contains(c)));
    assert_ne!(raw.to_lowercase(), normalized);
}
#[cfg(feature = "vad-silero")]
#[test]
#[ignore = "large-model tier (ASR_RUN_LARGE_MODEL_TEST=1); requires ASR_FIRE_RED_AED_MODEL (encoder/decoder int8 + tokens.txt) and ASR_VAD_MODEL"]
fn fire_red_aed_baseline_and_capability_rejection() {
    if !large_model_tier() {
        eprintln!("skipped: large-model tier (ASR_RUN_LARGE_MODEL_TEST unset)");
        return;
    }
    fire_red_regression(OfflineFamily::FireRedAsrAed, "ASR_FIRE_RED_AED_MODEL");
}
#[cfg(feature = "vad-silero")]
#[test]
#[ignore = "large-model tier (ASR_RUN_LARGE_MODEL_TEST=1); requires ASR_FIRE_RED_CTC_MODEL (model.int8.onnx + tokens.txt) and ASR_VAD_MODEL"]
fn fire_red_ctc_baseline_and_capability_rejection() {
    if !large_model_tier() {
        eprintln!("skipped: large-model tier (ASR_RUN_LARGE_MODEL_TEST unset)");
        return;
    }
    fire_red_regression(OfflineFamily::FireRedAsrCtc, "ASR_FIRE_RED_CTC_MODEL");
}
/// FireRedASR2 家族共用回归体：能力全拒（无热词/语言覆盖）、基线转写
/// 非空、会话级 hints 拒绝、纯静音经 VAD 过滤后无分句。
#[cfg(feature = "vad-silero")]
fn fire_red_regression(family: OfflineFamily, env_model: &str) {
    let dir = PathBuf::from(std::env::var(env_model).unwrap());
    let vad = VadConfig::new(std::env::var("ASR_VAD_MODEL").unwrap());
    let offline = |language: Option<String>,
                   bias: Option<TransducerBiasConfig>,
                   hints: Option<SpeechHints>| {
        prepare(EngineConfig::Offline(OfflineConfig {
            language,
            transducer_bias: bias,
            prompt_hints: hints,
            ..offline_config(dir.clone(), family, vad.clone())
        }))
    };
    // 中英自动识别：语言覆盖必须拒绝。
    let Err(error) = offline(Some("zh".into()), None, None) else {
        panic!("{family:?} must reject a language override");
    };
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    // 无热词通道：引擎级 bias 与 prompt_hints、会话级 hints 都必须拒绝。
    let Err(error) = offline(
        None,
        Some(TransducerBiasConfig::new(vec![BiasPhrase::new("张三")])),
        None,
    ) else {
        panic!("{family:?} must reject transducer bias");
    };
    assert_eq!(error.kind, ErrorKind::UnsupportedCapability);
    let Err(error) = offline(None, None, Some(SpeechHints::new(vec!["张三".into()]))) else {
        panic!("{family:?} must reject engine-level prompt hints");
    };
    assert_eq!(error.kind, ErrorKind::UnsupportedCapability);
    // 官方归档自带 0.wav（中英混说）；方言/8k 样本刻意不进断言（见
    // docs/validation.md 的验证限制），列表兜底只为容错非标准目录。
    let wav = [
        "0.wav",
        "1.wav",
        "2.wav",
        "3.wav",
        "3-sichuan.wav",
        "4-tianjin.wav",
        "5-henan.wav",
    ]
    .iter()
    .map(|name| dir.join("test_wavs").join(name))
    .find(|p| p.is_file())
    .expect("test_wavs sample");
    let audio = audio::read_wav_pcm16(wav).unwrap();
    let engine = offline(None, None, None).unwrap();
    assert!(!engine.capabilities().supports_session_hints);
    let error = start_with_hints_must_fail(
        &engine,
        audio.spec,
        "张三",
        &format!("session hotwords must fail on {family:?}"),
    );
    assert_eq!(error.kind, ErrorKind::UnsupportedCapability);
    let outcome = run(&engine, &audio, 1600);
    assert_eq!(outcome.received_frames, outcome.processed_frames);
    eprintln!("{family:?} baseline: {}", outcome.transcript.text());
    assert!(
        !outcome.transcript.text().trim().is_empty(),
        "baseline must be non-empty"
    );
    // 纯静音经 VAD 过滤后不产生分句。
    assert_silence_yields_no_segments(&engine);
}
