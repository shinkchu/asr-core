use super::{
    hotwords::{self, HotwordVocabulary},
    model_error,
};
use crate::{
    session::driver::{Driver, ResultSink},
    AsrError, ExecutionProvider, TransducerBiasConfig,
};
#[cfg(feature = "vad-silero")]
use crate::{OfflineFamily, SpeechHints};
#[cfg(feature = "vad-silero")]
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig};
use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig, OnlineStream};
use std::{path::Path, sync::Arc};

pub(crate) fn load_stream(
    dir: &Path,
    bias: Option<&TransducerBiasConfig>,
    num_threads: usize,
    provider: ExecutionProvider,
) -> Result<(Arc<OnlineRecognizer>, Option<String>, HotwordVocabulary), AsrError> {
    let (config, words, vocabulary) = stream_config(dir, bias, num_threads, provider)?;
    let recognizer = OnlineRecognizer::create(&config).ok_or_else(|| {
        model_error(format!(
            "failed to initialize streaming model with provider {}",
            provider.as_str()
        ))
    })?;
    Ok((Arc::new(recognizer), words, vocabulary))
}

#[cfg(feature = "vad-silero")]
pub(crate) fn load_offline(
    dir: &Path,
    family: OfflineFamily,
    language: Option<&str>,
    bias: Option<&TransducerBiasConfig>,
    prompt_hints: Option<&SpeechHints>,
    num_threads: usize,
    provider: ExecutionProvider,
) -> Result<(Arc<OfflineRecognizer>, Option<String>, HotwordVocabulary), AsrError> {
    let (config, words, vocabulary) = offline_config(
        dir,
        family,
        language,
        bias,
        prompt_hints,
        num_threads,
        provider,
    )?;
    let recognizer = OfflineRecognizer::create(&config).ok_or_else(|| {
        model_error(format!(
            "failed to initialize offline model with provider {}",
            provider.as_str()
        ))
    })?;
    Ok((Arc::new(recognizer), words, vocabulary))
}

fn stream_config(
    dir: &Path,
    bias: Option<&TransducerBiasConfig>,
    num_threads: usize,
    provider: ExecutionProvider,
) -> Result<(OnlineRecognizerConfig, Option<String>, HotwordVocabulary), AsrError> {
    super::precheck::validate_provider(provider)?;
    let files = super::model_layout::find_model_files(dir).map_err(model_error)?;
    let mut c = OnlineRecognizerConfig::default();
    c.model_config.transducer.encoder = Some(files.encoder.to_string_lossy().into_owned());
    c.model_config.transducer.decoder = Some(files.decoder.to_string_lossy().into_owned());
    c.model_config.transducer.joiner = Some(files.joiner.to_string_lossy().into_owned());
    c.model_config.tokens = Some(files.tokens.to_string_lossy().into_owned());
    c.model_config.num_threads = num_threads as i32;
    c.model_config.provider = Some(provider.as_str().into());
    let mut vocabulary = HotwordVocabulary::empty();
    let words = match bias {
        Some(bias) => {
            let (unit, bpe_vocab, prepared) = hotwords::prepare_bias(bias, &files)?;
            vocabulary = prepared;
            c.decoding_method = Some("modified_beam_search".into());
            c.max_active_paths = 4;
            c.hotwords_score = bias.default_score;
            c.model_config.modeling_unit = Some(unit);
            c.model_config.bpe_vocab = bpe_vocab;
            hotwords::render_bias(bias)
        }
        None => {
            c.decoding_method = Some("greedy_search".into());
            None
        }
    };
    c.enable_endpoint = true;
    c.rule1_min_trailing_silence = 1.0;
    c.rule2_min_trailing_silence = 2.0;
    c.rule3_min_utterance_length = 20.0;
    Ok((c, words, vocabulary))
}
// 离线路径的执行模型是"VAD 切段 → 逐段离线识别"（Offline driver 在
// start 构建 VAD、push/drain 消费 VAD 段），运行时依赖 vad-silero，
// 与 backend-sherpa 并非天然绑定而是本 crate 的设计选择。保持该门控：
// sherpa-only 构建不去编译一条必然不可用的路径，也防止未来被误删。
#[cfg(feature = "vad-silero")]
fn offline_config(
    dir: &Path,
    family: OfflineFamily,
    language: Option<&str>,
    bias: Option<&TransducerBiasConfig>,
    prompt_hints: Option<&SpeechHints>,
    num_threads: usize,
    provider: ExecutionProvider,
) -> Result<(OfflineRecognizerConfig, Option<String>, HotwordVocabulary), AsrError> {
    super::precheck::validate_provider(provider)?;
    // 家族能力规则（language/bias/prompt_hints 的取舍）唯一归宿在
    // backends::precheck，Engine::prepare 与 utils::precheck 共用。
    super::precheck::validate_family_parameters(family, language, bias, prompt_hints)?;
    let mut c = OfflineRecognizerConfig::default();
    c.model_config.num_threads = num_threads as i32;
    c.model_config.provider = Some(provider.as_str().into());
    let mut words = None;
    let mut vocabulary = HotwordVocabulary::empty();
    match family {
        // 三个家族共享"单 onnx + tokens.txt"布局发现与 SenseVoice 矛盾守卫
        //（find_offline_model_files 把 model 路径放在 files.encoder 字段里）。
        OfflineFamily::SenseVoice | OfflineFamily::Paraformer | OfflineFamily::FireRedAsrCtc => {
            let files = super::model_layout::find_offline_model_files(dir).map_err(model_error)?;
            super::model_layout::ensure_family_not_contradicted(family, &files.tokens)
                .map_err(model_error)?;
            c.model_config.tokens = Some(files.tokens.to_string_lossy().into_owned());
            let model = Some(files.encoder.to_string_lossy().into_owned());
            match family {
                OfflineFamily::SenseVoice => {
                    // 语言白名单已在 validate_family_parameters 校验。
                    let language = language.unwrap_or("auto");
                    c.model_config.sense_voice.model = model;
                    c.model_config.sense_voice.language = Some(language.into());
                    c.model_config.sense_voice.use_itn = true;
                }
                OfflineFamily::Paraformer => {
                    c.model_config.paraformer.model = model;
                }
                OfflineFamily::FireRedAsrCtc => {
                    c.model_config.fire_red_asr_ctc.model = model;
                }
                _ => unreachable!("handled above"),
            }
        }
        OfflineFamily::Transducer => {
            let files = super::model_layout::find_model_files(dir).map_err(model_error)?;
            c.model_config.tokens = Some(files.tokens.to_string_lossy().into_owned());
            c.model_config.transducer.encoder = Some(files.encoder.to_string_lossy().into_owned());
            c.model_config.transducer.decoder = Some(files.decoder.to_string_lossy().into_owned());
            c.model_config.transducer.joiner = Some(files.joiner.to_string_lossy().into_owned());
            if let Some(bias) = bias {
                let (unit, bpe_vocab, prepared) = hotwords::prepare_bias(bias, &files)?;
                vocabulary = prepared;
                c.decoding_method = Some("modified_beam_search".into());
                c.hotwords_score = bias.default_score;
                c.model_config.modeling_unit = Some(unit);
                c.model_config.bpe_vocab = bpe_vocab;
                words = hotwords::render_bias(bias);
            }
        }
        OfflineFamily::Qwen3Asr => {
            let files = super::model_layout::find_qwen3_files(dir).map_err(model_error)?;
            c.model_config.qwen3_asr.conv_frontend =
                Some(files.conv_frontend.to_string_lossy().into_owned());
            c.model_config.qwen3_asr.encoder = Some(files.encoder.to_string_lossy().into_owned());
            c.model_config.qwen3_asr.decoder = Some(files.decoder.to_string_lossy().into_owned());
            c.model_config.qwen3_asr.tokenizer =
                Some(files.tokenizer.to_string_lossy().into_owned());
            if let Some(hints) = prompt_hints {
                c.model_config.qwen3_asr.hotwords = hotwords::qwen3_prompt(hints)?;
            }
        }
        OfflineFamily::FunAsrNano => {
            let files = super::model_layout::find_funasr_nano_files(dir).map_err(model_error)?;
            c.model_config.funasr_nano.encoder_adaptor =
                Some(files.encoder_adaptor.to_string_lossy().into_owned());
            c.model_config.funasr_nano.llm = Some(files.llm.to_string_lossy().into_owned());
            c.model_config.funasr_nano.embedding =
                Some(files.embedding.to_string_lossy().into_owned());
            c.model_config.funasr_nano.tokenizer =
                Some(files.tokenizer.to_string_lossy().into_owned());
            // crate 的 Default 与 C++/Python 上游相反：temperature/top_p=1.0 跑成
            // 全随机采样、itn=0 关闭文本规整。显式固定为上游 C++ 默认：贪心解码
            // （temperature <= 1e-6 走 argmax）+ ITN 开。
            c.model_config.funasr_nano.temperature = 1e-6;
            c.model_config.funasr_nano.top_p = 0.8;
            c.model_config.funasr_nano.seed = 42;
            c.model_config.funasr_nano.max_new_tokens = 512;
            c.model_config.funasr_nano.itn = 1;
            if let Some(hints) = prompt_hints {
                c.model_config.funasr_nano.hotwords = hotwords::funasr_nano_prompt(hints)?;
            }
        }
        OfflineFamily::FireRedAsrAed => {
            let files = super::model_layout::find_fire_red_aed_files(dir).map_err(model_error)?;
            c.model_config.tokens = Some(files.tokens.to_string_lossy().into_owned());
            c.model_config.fire_red_asr.encoder =
                Some(files.encoder.to_string_lossy().into_owned());
            c.model_config.fire_red_asr.decoder =
                Some(files.decoder.to_string_lossy().into_owned());
        }
    }
    Ok((c, words, vocabulary))
}
pub(crate) struct Streaming {
    recognizer: Arc<OnlineRecognizer>,
    hotwords: Option<String>,
    stream: Option<OnlineStream>,
    index: u64,
    last: String,
    received: bool,
}
impl Streaming {
    pub(crate) fn new(recognizer: Arc<OnlineRecognizer>, hotwords: Option<String>) -> Self {
        Self {
            recognizer,
            hotwords,
            stream: None,
            index: 0,
            last: String::new(),
            received: false,
        }
    }
    fn decode(&mut self, control: &ResultSink, final_chunk: bool) -> Result<(), AsrError> {
        let stream = self.stream.as_ref().unwrap();
        while self.recognizer.is_ready(stream) {
            control.check()?;
            self.recognizer.decode(stream);
        }
        control.check()?;
        let text = self
            .recognizer
            .get_result(stream)
            .ok_or_else(|| AsrError::backend("streaming result unavailable"))?
            .text;
        if final_chunk || self.recognizer.is_endpoint(stream) {
            control.commit(self.index, self.index.to_string(), text, None, None)?;
            self.index += 1;
            self.last.clear();
            if !final_chunk {
                self.recognizer.reset(stream);
            }
        } else if self.last != text {
            control.partial(self.index.to_string(), &text)?;
            self.last = text;
        }
        Ok(())
    }
}
impl Driver for Streaming {
    fn start(&mut self, c: &ResultSink) -> Result<(), AsrError> {
        c.check()?;
        self.stream = Some(match &self.hotwords {
            Some(words) => self.recognizer.create_stream_with_hotwords(words),
            None => self.recognizer.create_stream(),
        });
        Ok(())
    }
    fn push(&mut self, samples: &[f32], c: &ResultSink) -> Result<(), AsrError> {
        self.received |= !samples.is_empty();
        self.stream
            .as_ref()
            .unwrap()
            .accept_waveform(16000, samples);
        self.decode(c, false)
    }
    fn finish(&mut self, c: &ResultSink) -> Result<(), AsrError> {
        if !self.received {
            return c.check();
        }
        let stream = self.stream.as_ref().unwrap();
        // Supply right context before closing the feature pipeline. It is not counted as input.
        stream.accept_waveform(16000, &[0.0; 12800]);
        stream.input_finished();
        self.decode(c, true)
    }
}
#[cfg(feature = "vad-silero")]
pub(crate) struct Offline {
    recognizer: Arc<OfflineRecognizer>,
    config: crate::VadConfig,
    hotwords: Option<String>,
    vad: Option<sherpa_onnx::VoiceActivityDetector>,
    index: u64,
}
#[cfg(feature = "vad-silero")]
impl Offline {
    pub(crate) fn new(
        recognizer: Arc<OfflineRecognizer>,
        config: crate::VadConfig,
        hotwords: Option<String>,
    ) -> Self {
        Self {
            recognizer,
            config,
            hotwords,
            vad: None,
            index: 0,
        }
    }
    fn drain(&mut self, c: &ResultSink) -> Result<(), AsrError> {
        let vad = self.vad.as_ref().unwrap();
        while let Some(segment) = vad.front() {
            c.check()?;
            let stream = match &self.hotwords {
                Some(words) => self.recognizer.create_stream_with_hotwords(words),
                None => self.recognizer.create_stream(),
            };
            stream.accept_waveform(16000, segment.samples());
            self.recognizer.decode(&stream);
            c.check()?;
            let text = stream
                .get_result()
                .ok_or_else(|| AsrError::backend("offline result unavailable"))?
                .text;
            let start = segment.start() as f64 / 16000.0;
            c.commit(
                self.index,
                self.index.to_string(),
                text,
                Some(start),
                Some(start + segment.samples().len() as f64 / 16000.0),
            )?;
            self.index += 1;
            vad.pop();
        }
        Ok(())
    }
}
#[cfg(feature = "vad-silero")]
impl Driver for Offline {
    fn start(&mut self, c: &ResultSink) -> Result<(), AsrError> {
        c.check()?;
        self.vad = Some(super::vad::create(&self.config)?);
        Ok(())
    }
    fn push(&mut self, samples: &[f32], c: &ResultSink) -> Result<(), AsrError> {
        for chunk in samples.chunks(512) {
            c.check()?;
            self.vad.as_ref().unwrap().accept_waveform(chunk);
            self.drain(c)?;
        }
        Ok(())
    }
    fn finish(&mut self, c: &ResultSink) -> Result<(), AsrError> {
        self.vad.as_ref().unwrap().flush();
        self.drain(c)
    }
}

#[cfg(test)]
mod tests;
