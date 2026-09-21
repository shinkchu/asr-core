use super::*;

#[test]
fn secrets_never_serialize_or_debug() {
    let config = OpenAiHttpConfig::new(
        "https://example.com/v1",
        "model",
        Secret::new("private-key"),
    );
    assert!(!format!("{config:?}").contains("private-key"));
    let json = serde_json::to_string(&config).unwrap();
    assert!(!json.contains("private-key"));
    assert!(!json.contains("api_key"));
}

#[test]
fn secret_zeroizes_backing_buffer_and_wipes_on_drop() {
    use zeroize::Zeroize as _;

    // Drop 使用的 String::zeroize 在清零后会 clear，清空后的缓冲无法
    // 再读出断言；这里对同一擦除原语的就地清零语义直接验证。
    let mut buffer = *b"private-key";
    buffer.zeroize();
    assert!(buffer.iter().all(|byte| *byte == 0));

    // Secret 实现 Drop,离开作用域即触发上述擦除(类型级验证;
    // 堆缓冲释放后的内存归属不可安全读取断言)。
    #[allow(drop_bounds)] // 有意用作 `Secret: Drop` 的类型级断言
    fn assert_wipes_on_drop<T: Drop>(_: &T) {}
    assert_wipes_on_drop(&Secret::new("x"));
}

#[test]
fn timeouts_and_vad_deserialize_with_constructor_defaults() {
    let timeouts: Timeouts = serde_json::from_str("{}").unwrap();
    assert_eq!(timeouts, Timeouts::default());
    assert_eq!(timeouts.connect, Duration::from_secs(10));
    assert_eq!(timeouts.send, Duration::from_secs(5));
    assert_eq!(timeouts.response, Duration::from_secs(30));

    let explicit: Timeouts = serde_json::from_str(
        r#"{"connect":{"secs":1,"nanos":0},"send":{"secs":2,"nanos":0},"response":{"secs":3,"nanos":0}}"#,
    )
    .unwrap();
    assert_eq!(explicit.connect, Duration::from_secs(1));
    assert_eq!(explicit.send, Duration::from_secs(2));
    assert_eq!(explicit.response, Duration::from_secs(3));

    let vad: VadConfig = serde_json::from_str(r#"{"model":"/v"}"#).unwrap();
    assert_eq!(vad, VadConfig::new("/v"));
    assert_eq!(vad.threshold, 0.5);
    assert_eq!(vad.min_silence, 0.5);
    assert_eq!(vad.min_speech, 0.25);
    assert_eq!(vad.max_speech, 15.0);

    // deny_unknown_fields 对最小 JSON 之外的多余字段仍然拒绝。
    let unknown_timeouts = r#"{"connect":{"secs":1,"nanos":0},"ignored":true}"#;
    assert!(serde_json::from_str::<Timeouts>(unknown_timeouts).is_err());
    let unknown_vad = r#"{"model":"/v","ignored":true}"#;
    assert!(serde_json::from_str::<VadConfig>(unknown_vad).is_err());
}

#[test]
fn concrete_local_configs_have_minimal_0_4_json() {
    let streaming: EngineConfig =
        serde_json::from_str(r#"{"Streaming":{"model_dir":"/m"}}"#).unwrap();
    assert_eq!(
        streaming,
        EngineConfig::Streaming(StreamingConfig::new("/m"))
    );

    let offline: EngineConfig = serde_json::from_str(
        r#"{"Offline":{"model_dir":"/m","family":"SenseVoice","vad":{"model":"/v","threshold":0.5,"min_silence":0.5,"min_speech":0.25,"max_speech":15.0}}}"#,
    )
    .unwrap();
    assert_eq!(
        offline,
        EngineConfig::Offline(OfflineConfig::new(
            "/m",
            OfflineFamily::SenseVoice,
            VadConfig::new("/v"),
        ))
    );

    let biased: EngineConfig = serde_json::from_str(
        r#"{"Streaming":{"model_dir":"/m","bias":{"phrases":[{"phrase":"张三","score":3.5}]}}}"#,
    )
    .unwrap();
    let EngineConfig::Streaming(biased) = biased else {
        panic!("wrong variant")
    };
    assert_eq!(biased.bias.unwrap().default_score, 2.0);

    let prompted: EngineConfig = serde_json::from_str(
        r#"{"Offline":{"model_dir":"/m","family":"Qwen3Asr","vad":{"model":"/v","threshold":0.5,"min_silence":0.5,"min_speech":0.25,"max_speech":15.0},"prompt_hints":{"phrases":["骨质疏松症患者"]}}}"#,
    )
    .unwrap();
    let EngineConfig::Offline(prompted) = prompted else {
        panic!("wrong variant")
    };
    assert_eq!(
        prompted.prompt_hints.unwrap().phrases,
        vec!["骨质疏松症患者"]
    );

    for (family_json, family) in [
        ("FireRedAsrAed", OfflineFamily::FireRedAsrAed),
        ("FireRedAsrCtc", OfflineFamily::FireRedAsrCtc),
    ] {
        let fire_red: EngineConfig = serde_json::from_str(&format!(
            r#"{{"Offline":{{"model_dir":"/m","family":"{family_json}","vad":{{"model":"/v","threshold":0.5,"min_silence":0.5,"min_speech":0.25,"max_speech":15.0}}}}}}"#,
        ))
        .unwrap();
        assert_eq!(
            fire_red,
            EngineConfig::Offline(OfflineConfig::new("/m", family, VadConfig::new("/v")))
        );
    }
}

#[test]
fn cloud_json_uses_0_4_fields_and_requires_secret_injection() {
    let http: EngineConfig = serde_json::from_str(
        r#"{"OpenAiHttp":{"api_root":"https://example.com/v1","model":"whisper-1"}}"#,
    )
    .unwrap();
    let EngineConfig::OpenAiHttp(http) = http else {
        panic!("wrong variant")
    };
    assert_eq!(
        http,
        OpenAiHttpConfig::new("https://example.com/v1", "whisper-1", Secret::default(),)
    );
    assert_eq!(http.api_key, Secret::default());

    let realtime: EngineConfig = serde_json::from_str(
        r#"{"OpenAiRealtime":{"endpoint":"wss://example.com/v1/realtime","model":"gpt-realtime"}}"#,
    )
    .unwrap();
    let EngineConfig::OpenAiRealtime(realtime) = realtime else {
        panic!("wrong variant")
    };
    assert!(realtime.server_vad);
    assert_eq!(realtime.api_key, Secret::default());
}

#[test]
fn removed_or_unknown_fields_are_rejected() {
    let old_http = r#"{"OpenAiHttp":{"api_root":"https://example.com/v1","model":"whisper-1","supports_sse":true}}"#;
    assert!(serde_json::from_str::<EngineConfig>(old_http).is_err());

    let old_realtime = r#"{"OpenAiRealtime":{"endpoint":"wss://example.com/v1/realtime","model":"gpt-realtime","sample_rate":24000}}"#;
    assert!(serde_json::from_str::<EngineConfig>(old_realtime).is_err());

    let unknown_local = r#"{"Streaming":{"model_dir":"/m","ignored":true}}"#;
    assert!(serde_json::from_str::<EngineConfig>(unknown_local).is_err());

    let old_hotwords = r#"{"Streaming":{"model_dir":"/m","hotwords":null}}"#;
    assert!(serde_json::from_str::<EngineConfig>(old_hotwords).is_err());
}

#[cfg(feature = "backend-dashscope")]
#[test]
fn dashscope_rejects_blank_api_keys() {
    for key in ["", "   "] {
        let error = DashScopeConfig::new("wss://example.com", "model", Secret::new(key))
            .validate_parameters()
            .expect_err("blank API key must be rejected");
        assert_eq!(error.kind, crate::ErrorKind::InvalidInput);
        assert_eq!(error.stage, "configuration");
    }
    DashScopeConfig::new("wss://example.com", "model", Secret::new("key"))
        .validate_parameters()
        .expect("non-blank API key must be accepted");
}

#[cfg(feature = "backend-openai-http")]
#[test]
fn openai_http_rejects_blank_api_keys_like_dashscope() {
    for key in ["", "   "] {
        let error = OpenAiHttpConfig::new("https://example.com/v1", "model", Secret::new(key))
            .validate_parameters()
            .expect_err("blank API key must be rejected");
        assert_eq!(error.kind, crate::ErrorKind::InvalidInput);
        assert_eq!(error.stage, "configuration");
    }
    OpenAiHttpConfig::new("https://example.com/v1", "model", Secret::new("key"))
        .validate_parameters()
        .expect("non-blank API key must be accepted");
}

#[cfg(feature = "backend-openai-realtime")]
#[test]
fn openai_realtime_rejects_blank_api_keys_like_dashscope() {
    for key in ["", "   "] {
        let error =
            OpenAiRealtimeConfig::new("wss://example.com/v1/realtime", "model", Secret::new(key))
                .validate_parameters()
                .expect_err("blank API key must be rejected");
        assert_eq!(error.kind, crate::ErrorKind::InvalidInput);
        assert_eq!(error.stage, "configuration");
    }
    OpenAiRealtimeConfig::new("wss://example.com/v1/realtime", "model", Secret::new("key"))
        .validate_parameters()
        .expect("non-blank API key must be accepted");
}
