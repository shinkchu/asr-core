//! 转录本地音频文件。识别过程实时反馈到 stderr：每段定稿立即打印（带
//! 时间戳），流式模型的部分结果在交互式终端上单行原地刷新；完整转写
//! 仍只在结束后打印到 stdout，重定向管道拿到的是干净结果。

use asr_core::{
    audio::read_wav_pcm16,
    utils::models::{detect, LocalModel},
    AudioChunk, BiasPhrase, Engine, EngineConfig, EngineOptions, ExecutionProvider, OfflineConfig,
    OfflineFamily, PunctConfig, SessionOptions, StreamingConfig, Subscription,
    TransducerBiasConfig, Update, VadConfig, DEFAULT_NUM_THREADS, MAX_NUM_THREADS,
};
use std::{
    error::Error,
    io::IsTerminal,
    path::PathBuf,
    time::{Duration, Instant},
};
fn main() -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let mut positional: Vec<String> = Vec::new();
    let mut hotwords: Vec<BiasPhrase> = Vec::new();
    let mut vad: Option<PathBuf> = None;
    let mut max_duration_secs: Option<u64> = None;
    let mut deadline_secs: Option<u64> = None;
    let mut threads: Option<usize> = None;
    let mut provider = ExecutionProvider::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--hotwords" {
            let list = args.next().ok_or("--hotwords requires a value")?;
            for word in list.split(',').map(str::trim).filter(|w| !w.is_empty()) {
                hotwords.push(BiasPhrase::new(word));
            }
        } else if arg == "--vad" {
            vad = Some(PathBuf::from(args.next().ok_or("--vad requires a value")?));
        } else if arg == "--max-duration" {
            max_duration_secs = Some(flag_seconds("--max-duration", args.next())?);
        } else if arg == "--deadline" {
            deadline_secs = Some(flag_seconds("--deadline", args.next())?);
        } else if arg == "--provider" {
            provider = args.next().ok_or("--provider requires a value")?.parse()?;
        } else if arg == "--threads" {
            threads = Some(flag_threads(args.next())?);
        } else {
            positional.push(arg);
        }
    }
    if positional.len() != 2 && positional.len() != 3 {
        return Err(
            "usage: transcribe_file MODEL_DIRECTORY AUDIO.wav [PUNCTUATION_MODEL_DIRECTORY] [--hotwords WORD1,WORD2,...] [--vad silero_vad.onnx] [--max-duration SECONDS] [--deadline SECONDS] [--threads N] [--provider cpu|cuda|coreml]"
                .into(),
        );
    }
    let audio = read_wav_pcm16(&positional[1])?;
    let model_dir = PathBuf::from(&positional[0]);
    let punctuation = positional.get(2).map(|dir| PunctConfig::new(dir.clone()));
    let threads = threads.unwrap_or(DEFAULT_NUM_THREADS);
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
                num_threads: threads,
                provider,
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
                num_threads: threads,
                provider,
            })
        }
    };
    eprintln!("加载模型（请求 provider: {}）…", provider.as_str());
    let engine = Engine::prepare(config, EngineOptions::default())?;
    let mut options = SessionOptions::new(audio.spec);
    // 会话时长上限：默认取库内硬顶 24 小时（coordinator 24 小时校验），
    // 示例自身不叠加人为限制。
    options.max_duration = Duration::from_secs(max_duration_secs.unwrap_or(24 * 60 * 60));
    options.max_transcript_bytes = 2 * 1024 * 1024;
    // 整体截止：默认按"音频时长 + 5 分钟"兜底；离线识别通常快于实时，
    // 机器较慢时可用 --deadline 显式放宽。
    let audio_secs = (audio.samples.len() as u64).div_ceil(u64::from(audio.spec.sample_rate));
    let deadline =
        Instant::now() + Duration::from_secs(deadline_secs.unwrap_or(audio_secs + 5 * 60));
    eprintln!("音频 {audio_secs} 秒，开始识别…");
    // 会话级流程替代 Engine::transcribe：先订阅事件流再喂音频，段定稿与
    // 部分结果经渲染线程实时到达终端，而不是等整段转写结束才输出。
    let session = engine.start(options)?;
    let subscription = session
        .subscribe()
        .ok_or("session subscription slot unavailable")?;
    let display = std::thread::spawn(move || render_live(subscription));
    let input = session.input();
    // 与 Engine::transcribe 相同的 100ms 分块；识别慢于喂入时 push_wait 在
    // 队列水位上背压等待，不丢音频。
    let chunk_frames = (audio.spec.sample_rate as usize / 10).max(1);
    for samples in audio.samples.chunks(chunk_frames) {
        let chunk = AudioChunk {
            samples: samples.to_vec(),
            spec: audio.spec,
        };
        if let Err(error) = input.push_wait(chunk, deadline) {
            session.cancel();
            display.join().map_err(|_| "event display failed")?;
            return Err(error.into());
        }
    }
    let result = session.finish(deadline);
    // 结束前定稿的段可能还排在订阅队列里：先 join 渲染线程再打印统计，
    // 保证 stderr 上的输出顺序。
    display.join().map_err(|_| "event display failed")?;
    let outcome = result?;
    eprintln!(
        "识别完成：{} 段，总耗时 {:.1} 秒",
        outcome.transcript.segments.len(),
        started.elapsed().as_secs_f64()
    );
    println!("{}", outcome.transcript.text());
    Ok(())
}

/// 订阅事件 → stderr。段定稿立即成行打印；流式部分结果仅在交互式终端
/// 上渲染（单行原地刷新），重定向时跳过部分结果、只保留定稿段，避免
/// 转义序列和中间文本污染落盘内容。
fn render_live(mut subscription: Subscription) {
    let interactive = std::io::stderr().is_terminal();
    // 终端上是否有未换行的部分结果行，等待被段定稿或结束清行。
    let mut live_line = false;
    while let Some(update) = subscription.recv() {
        match update {
            Update::Partial { text, .. } if interactive => {
                eprint!("\r\x1b[K{}", partial_tail(&text));
                live_line = true;
            }
            Update::Segment(segment) => {
                if live_line {
                    eprint!("\r\x1b[K");
                    live_line = false;
                }
                let text = segment.text.trim();
                if text.is_empty() {
                    continue;
                }
                match (segment.start_seconds, segment.end_seconds) {
                    (Some(start), Some(end)) => {
                        eprintln!("[{} - {}] {}", stamp(start), stamp(end), text)
                    }
                    _ => eprintln!("{text}"),
                }
            }
            Update::Reset(_) | Update::Phase(_) => {}
            _ => {}
        }
    }
    if live_line {
        eprint!("\r\x1b[K");
    }
}

/// 部分结果单行尾部截断：折行后 `\r` + 清行只能清掉最后一行、留下残影，
/// 超过宽度时仅保留尾部并加省略号。
fn partial_tail(text: &str) -> String {
    const WIDTH: usize = 60;
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= WIDTH {
        return text.to_string();
    }
    let mut tail = String::from("…");
    tail.extend(chars[chars.len() - WIDTH..].iter());
    tail
}

/// 秒 → "mm:ss.ss" 段时间戳。
fn stamp(seconds: f64) -> String {
    format!(
        "{:02}:{:05.2}",
        seconds.div_euclid(60.0) as u64,
        seconds.rem_euclid(60.0)
    )
}

/// "--flag SECONDS" 解析：值缺失、非正整数均按 flag 名报错。
fn flag_seconds(flag: &str, value: Option<String>) -> Result<u64, Box<dyn Error>> {
    let value = value.ok_or_else(|| format!("{flag} requires a value"))?;
    let secs = value
        .parse()
        .map_err(|_| format!("{flag} expects a whole number of seconds, got {value:?}"))?;
    if secs == 0 {
        return Err(format!("{flag} must be greater than zero").into());
    }
    Ok(secs)
}

/// "--threads N" 解析：值缺失、非整数、超范围均在 CLI 层报错——范围与库的
/// precheck 同一份上限（MAX_NUM_THREADS），不让无效值走到 wav 读取与模型
/// 准备之后。
fn flag_threads(value: Option<String>) -> Result<usize, Box<dyn Error>> {
    let value = value.ok_or("--threads requires a value")?;
    let threads = value
        .parse()
        .map_err(|_| format!("--threads expects a whole number of threads, got {value:?}"))?;
    if !(1..=MAX_NUM_THREADS).contains(&threads) {
        return Err(format!("--threads must be between 1 and {MAX_NUM_THREADS}").into());
    }
    Ok(threads)
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
