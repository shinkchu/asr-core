use super::punctuation_error;
use crate::{session::driver::FinalTextProcessor, AsrError, PunctConfig};
use sherpa_onnx::{
    OfflinePunctuation, OfflinePunctuationConfig, OnlinePunctuation, OnlinePunctuationConfig,
};
use std::sync::Arc;

enum Punct {
    CtTransformer(OfflinePunctuation),
    CnnBilstm(OnlinePunctuation),
}

/// CNN-BiLSTM 的 BPE 词表只含小写词形；流式 zipformer 输出全大写文本时
/// 模型会原样返回。送入模型前先转小写，让标点与大小写恢复正常。
fn lowercase_ascii(text: &str) -> String {
    text.chars().map(|c| c.to_ascii_lowercase()).collect()
}

impl Punct {
    fn add(&self, text: &str) -> Option<String> {
        match self {
            Punct::CtTransformer(p) => p.add_punctuation(text),
            Punct::CnnBilstm(p) => p.add_punctuation(&lowercase_ascii(text)),
        }
    }
}

/// 在引擎 prepare 阶段创建一次；sherpa 两类对象均无状态且 `Send + Sync`，
/// 返回的 `Arc` 可跨并发 session 共享。
pub(crate) fn create(c: &PunctConfig) -> Result<FinalTextProcessor, AsrError> {
    let files = super::model_layout::find_punct_model_files(&c.model).map_err(punctuation_error)?;
    let model = files.model.to_string_lossy().into_owned();
    let punct = match files.vocab {
        // zh-en CT-Transformer：仅 model*.onnx，词表内嵌于模型元数据。
        None => {
            let mut config = OfflinePunctuationConfig::default();
            config.model.ct_transformer = Some(model);
            config.model.num_threads = 2;
            config.model.provider = Some("cpu".into());
            Punct::CtTransformer(OfflinePunctuation::create(&config).ok_or_else(|| {
                AsrError::backend("failed to initialize CT-Transformer punctuation model")
            })?)
        }
        // 英文 CNN-BiLSTM：model*.onnx + bpe.vocab，额外恢复大小写。
        Some(vocab) => {
            let mut config = OnlinePunctuationConfig::default();
            config.model.cnn_bilstm = Some(model);
            config.model.bpe_vocab = Some(vocab.to_string_lossy().into_owned());
            config.model.num_threads = 2;
            config.model.provider = Some("cpu".into());
            Punct::CnnBilstm(OnlinePunctuation::create(&config).ok_or_else(|| {
                AsrError::backend("failed to initialize CNN-BiLSTM punctuation model")
            })?)
        }
    };
    Ok(Arc::new(move |text: &str| {
        // 推理失败（含内部 NUL）回退原文：后处理是 best-effort，不是会话故障点。
        punct.add(text).unwrap_or_else(|| text.to_string())
    }))
}
