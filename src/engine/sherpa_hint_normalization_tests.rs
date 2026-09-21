use super::normalize_session_hints;
use crate::backends::hotwords;
use crate::{BiasPhrase, SessionOptions, SpeechHints, TransducerBiasConfig};

/// 支持 hints 的后端(引擎级 bias 的 streaming/transducer)由 driver
/// 经 `hotwords::merge(words, options.hints)` 消费会话 hints:归一化
/// 后的空短语 hints 必须与 None 得到完全相同的偏置(无偏置叠加)。
#[test]
fn normalized_empty_hints_bias_exactly_like_none_on_supported_backends() {
    let bias = TransducerBiasConfig::new(vec![BiasPhrase::new("语音识别")]);
    let defaults = hotwords::render_bias(&bias).unwrap();

    let options = normalize_session_hints(SessionOptions {
        hints: Some(SpeechHints::default()),
        ..Default::default()
    });
    assert!(options.hints.is_none());
    let with_empty_hints = hotwords::merge(Some(&defaults), options.hints.as_ref()).unwrap();
    let without_hints = hotwords::merge(Some(&defaults), None).unwrap();
    assert_eq!(with_empty_hints, without_hints);
    assert_eq!(with_empty_hints.as_deref(), Some(defaults.as_str()));
}
