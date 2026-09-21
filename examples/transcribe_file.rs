use asr_core::{
    audio::read_wav_pcm16,
    utils::models::{detect, LocalModel},
    BiasPhrase, Engine, EngineConfig, EngineOptions, OfflineConfig, OfflineFamily, PunctConfig,
    SessionOptions, StreamingConfig, TransducerBiasConfig, VadConfig,
};
use std::{
    error::Error,
    path::PathBuf,
    time::{Duration, Instant},
};
fn main() -> Result<(), Box<dyn Error>> {
    let mut positional: Vec<String> = Vec::new();
    let mut hotwords: Vec<BiasPhrase> = Vec::new();
    let mut vad: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--hotwords" {
            let list = args.next().ok_or("--hotwords requires a value")?;
            for word in list.split(',').map(str::trim).filter(|w| !w.is_empty()) {
                hotwords.push(BiasPhrase::new(word));
            }
        } else if arg == "--vad" {
            vad = Some(PathBuf::from(args.next().ok_or("--vad requires a value")?));
        } else {
            positional.push(arg);
        }
    }
    if positional.len() != 2 && positional.len() != 3 {
        return Err(
            "usage: transcribe_file MODEL_DIRECTORY AUDIO.wav [PUNCTUATION_MODEL_DIRECTORY] [--hotwords WORD1,WORD2,...] [--vad silero_vad.onnx]"
                .into(),
        );
    }
    let audio = read_wav_pcm16(&positional[1])?;
    let model_dir = PathBuf::from(&positional[0]);
    let punctuation = positional.get(2).map(|dir| PunctConfig::new(dir.clone()));
    let config = match detect(&model_dir)? {
        // transducer 布局（encoder/decoder/joiner）走流式识别；离线 transducer
        // 意图须用 transcribe 示例的 JSON 配置显式指定 OfflineFamily::Transducer。
        LocalModel::Transducer => {
            if vad.is_some() {
                return Err("--vad applies to offline model families only".into());
            }
            EngineConfig::Streaming(StreamingConfig {
                model_dir,
                punctuation,
                // 引擎级热词：启用 modified_beam_search 并对所有会话生效。
                bias: (!hotwords.is_empty()).then(|| TransducerBiasConfig::new(hotwords)),
            })
        }
        family => {
            if !hotwords.is_empty() {
                return Err("hotwords need a streaming transducer model".into());
            }
            EngineConfig::Offline(OfflineConfig {
                model_dir,
                family: offline_family(family)?,
                language: None,
                vad: VadConfig::new(offline_vad(vad, family)?),
                punctuation,
                transducer_bias: None,
                prompt_hints: None,
            })
        }
    };
    let engine = Engine::prepare(config, EngineOptions::default())?;
    let mut options = SessionOptions::new(audio.spec);
    options.max_duration = Duration::from_secs(60 * 60);
    options.max_transcript_bytes = 2 * 1024 * 1024;
    let result = engine.transcribe(&audio, options, Instant::now() + Duration::from_secs(30))?;
    println!("{}", result.transcript.text());
    Ok(())
}

/// 目录布局 → 离线家族。流式家族已在调用方分流，这里只裁决可判定的
/// 离线布局；Flat 与 Punct 无法对应，返回带原因的错误。
fn offline_family(model: LocalModel) -> Result<OfflineFamily, Box<dyn Error>> {
    Ok(match model {
        LocalModel::SenseVoice => OfflineFamily::SenseVoice,
        LocalModel::FireRedAsrAed => OfflineFamily::FireRedAsrAed,
        LocalModel::Qwen3Asr => OfflineFamily::Qwen3Asr,
        LocalModel::FunAsrNano => OfflineFamily::FunAsrNano,
        // 扁平布局（model*.onnx + tokens.txt）为 Paraformer、FireRedASR-CTC
        // 与无标记 SenseVoice 变体共享，目录本身无法裁决，须显式配置。
        LocalModel::Flat => {
            return Err(
                "flat model*.onnx layout is shared by Paraformer, FireRedASR-CTC and \
                 markerless SenseVoice variants; use the transcribe example with a JSON config \
                 that names the family"
                    .into(),
            )
        }
        LocalModel::Punct => {
            return Err("model directory is a punctuation model, not an ASR model".into())
        }
        _ => return Err("unsupported model directory layout".into()),
    })
}

/// 离线家族按 "VAD 切段 → 逐段识别" 执行，须要 silero VAD 模型文件。
fn offline_vad(vad: Option<PathBuf>, model: LocalModel) -> Result<PathBuf, Box<dyn Error>> {
    if !cfg!(feature = "vad-silero") {
        return Err(
            "offline model families need the vad-silero feature; rebuild with \
             --features backend-sherpa,vad-silero"
                .into(),
        );
    }
    vad.ok_or_else(|| {
        format!(
            "{} needs the silero VAD model file; pass --vad silero_vad.onnx",
            match model {
                LocalModel::SenseVoice => "SenseVoice",
                LocalModel::FireRedAsrAed => "FireRedASR-AED",
                LocalModel::Qwen3Asr => "Qwen3-ASR",
                LocalModel::FunAsrNano => "FunASR-Nano",
                _ => "this model family",
            }
        )
        .into()
    })
}
