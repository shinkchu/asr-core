# asr-core

[![Crates.io](https://img.shields.io/crates/v/asr-core.svg)](https://crates.io/crates/asr-core)
[![Documentation](https://docs.rs/asr-core/badge.svg)](https://docs.rs/asr-core)
[![CI](https://github.com/appstore/asr-core/actions/workflows/ci.yml/badge.svg)](https://github.com/appstore/asr-core/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/appstore/asr-core/blob/master/LICENSE)

嵌入 Rust 应用的语音识别库。统一文件与麦克风输入、会话结束、取消、分句、实时观察和错误处理。

当前文档对应 **0.0.1**。

已实现的适配器（真实环境验证状态见[验证记录](https://github.com/appstore/asr-core/blob/master/docs/validation.md)）：

| 后端 | 工作方式 | Feature |
|---|---|---|
| Zipformer transducer | 本地流式，中间结果与端点分句 | `backend-sherpa` |
| SenseVoice / Paraformer | 本地 VAD 切段后识别 | `backend-sherpa,vad-silero` |
| Offline transducer | 本地 VAD 切段后识别，支持热词 | `backend-sherpa,vad-silero` |
| Qwen3-ASR | 本地 LLM 识别（中英等多语言），支持热词 | `backend-sherpa,vad-silero` |
| FunASR-Nano | 本地 LLM 识别（中英日 + 中文方言），支持热词 | `backend-sherpa,vad-silero` |
| FireRedASR2（AED/CTC） | 本地 VAD 切段后识别（中英） | `backend-sherpa,vad-silero` |
| DashScope | Paraformer WebSocket 实时协议 | `backend-dashscope` |
| OpenAI 兼容 HTTP | 完整 WAV 上传；可选 JSON、text、SSE | `backend-openai-http` |
| HTTP 按句上传 | VAD 切段后依次上传 | `backend-openai-http,vad-silero` |
| OpenAI Realtime | GA 纯转写 WebSocket，24kHz PCM | `backend-openai-realtime` |
| 默认麦克风 | F32/I16/U16，回调外批处理 | `capture-cpal` |

## 安装

默认 feature 为空；按实际后端显式启用，避免无意引入 CPAL、sherpa-onnx 或网络运行时：

```toml
[dependencies]
asr-core = { version = "0.0.1", features = ["backend-openai-http"] }
```

本地 sherpa-onnx + 麦克风：

```toml
[dependencies]
asr-core = { version = "0.0.1", default-features = false, features = ["backend-sherpa", "vad-silero", "capture-cpal"] }
```

DashScope：

```toml
[dependencies]
asr-core = { version = "0.0.1", default-features = false, features = ["backend-dashscope"] }
```

OpenAI Realtime：

```toml
[dependencies]
asr-core = { version = "0.0.1", default-features = false, features = ["backend-openai-realtime"] }
```

无 feature 的核心不引入 CPAL、sherpa-onnx 或网络运行时。模型资产由宿主显式准备；`punct-sherpa` 为本地后端提供可选标点恢复。

## 文件转写

```rust,no_run
use std::time::{Duration, Instant};
use asr_core::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = audio::read_wav_pcm16("speech.wav")?;
    let engine = Engine::prepare(
        EngineConfig::OpenAiHttp(OpenAiHttpConfig::new(
            "https://api.openai.com/v1", "your-transcription-model",
            Secret::new(std::env::var("ASR_API_KEY")?),
        )),
        EngineOptions::default(),
    )?;
    let outcome = engine.transcribe(
        &audio,
        SessionOptions::new(audio.spec),
        Instant::now() + Duration::from_secs(60),
    )?;
    println!("{}", outcome.transcript.text());
    Ok(())
}
```

没有自动重试、协议猜测或后端回退。HTTP API 根地址可包含 `/v1/` 和代理前缀，接口追加 `audio/transcriptions`。第三方需要实现音频协议，仅聊天接口兼容不够。配置不序列化密钥，Debug 输出也隐藏密钥。

## GPU / 硬件加速

本地流式与离线识别可通过 `provider` 选择执行设备，默认 `cpu`，旧 JSON 配置可继续使用：

```rust
use asr_core::{ExecutionProvider, StreamingConfig};
let mut config = StreamingConfig::new("/path/to/model");
config.provider = ExecutionProvider::Cuda; // NVIDIA；Apple 使用 CoreMl
```

`OfflineConfig` 使用相同字段；文件示例支持 `--provider cpu|cuda|coreml`。
CUDA 需要匹配的 GPU 原生库，启用 `backend-sherpa-shared` 并在构建前设置
`SHERPA_ONNX_LIB_DIR`。CoreML 需要含 CoreML 的 Apple 原生库。选择 provider
并不保证全部计算在 GPU 上执行，上游可能回退 CPU；VAD 与标点仍使用 CPU。
完整部署步骤、模型选择和验证方式见[后端配置](https://github.com/appstore/asr-core/blob/master/docs/providers.md#gpu--硬件加速)。

## 标点恢复（可选）

标点是否原生输出因家族而异：SenseVoice、Qwen3-ASR、FunASR-Nano 原生输出已带标点；流式 zipformer、离线 transducer、Paraformer、FireRedASR2 的原始输出没有标点。后一类家族启用 `punct-sherpa` 并在引擎配置中声明标点模型后，所有 final 自动经过标点后处理：

```rust,no_run
use asr_core::*;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut config = StreamingConfig::new(
    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
);
config.punctuation = Some(PunctConfig::new(
    "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8",
));
let engine = Engine::prepare(EngineConfig::Streaming(config), EngineOptions::default())?;
// 模型随引擎一次加载、跨会话共享；配置即启用。
let session = engine.start(SessionOptions::default())?;
# let _ = session;
# Ok(())
# }
```

模型家族按目录内容自动识别：含 `bpe.vocab` 为英文 CNN-BiLSTM（半角标点加大小写恢复），否则为中英 CT-Transformer（纯英文输入会得到全角标点）。重复 final 会在推理前判重；推理失败回退原文，不影响会话。模型下载地址与体积见[后端配置](https://github.com/appstore/asr-core/blob/master/docs/providers.md)。

## 热词（可选）

本地 transducer 引擎（流式 Zipformer、离线 transducer）与提示注入式 LLM 引擎（Qwen3-ASR、FunASR-Nano）支持语音提示。transducer 的打分语义只存在于 `TransducerBiasConfig`；通用会话提示 `SpeechHints` 只包含短语，不承诺具体分数。配置 transducer bias 后解码切换到 `modified_beam_search`（开销约为 greedy 的 2~4 倍），未配置时保持 greedy 不变。

```rust,no_run
use asr_core::*;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut config = StreamingConfig::new(
    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
);
// 引擎级词表对所有会话生效。建模单元按模型目录自动推断：
// 含 bpe.vocab 时按 tokens 是否含 CJK 取 cjkchar+bpe / bpe，否则 cjkchar。
config.bias = Some(TransducerBiasConfig::new(vec![
    BiasPhrase::new("语音识别"),
    BiasPhrase::scored("张三", 3.5),
]));
let engine = Engine::prepare(EngineConfig::Streaming(config), EngineOptions::default())?;
# let _ = engine;
# Ok(())
# }
```

会话级短语与默认 bias 短语合并，适合各会话注入不同术语；要求引擎已配置 transducer bias（空短语表可仅启用机制），否则 `start` 返回 `UnsupportedCapability`：

```rust,no_run
# use asr_core::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let mut config = StreamingConfig::new("model");
# config.bias = Some(TransducerBiasConfig::new(Vec::new()));
# let engine = Engine::prepare(EngineConfig::Streaming(config), EngineOptions::default())?;
let session = engine.start(SessionOptions {
    hints: Some(SpeechHints::new(vec!["本次会话的专有名词".into()])),
    ..Default::default()
})?;
# let _ = session;
# Ok(())
# }
```

Qwen3-ASR 与 FunASR-Nano 使用 `OfflineConfig::prompt_hints`，只在 Engine 准备时注入；它们不接受会话级 hints。SenseVoice、Paraformer 不接受 bias 或 prompt hints。transducer 短语不能包含 `/`、换行、`:`、`,`、`#`、`@` 或控制字符；每条不超过 64 字符、每组不超过 256 条。中文建模单元会在 prepare/start 时按同一份 tokens 表检查 OOV。Qwen3/FunASR 的提示总字符数上限为 64，FunASR 另拒绝其上游分隔符 `;`、`；`、`，`。模型可用性与下载地址见[后端配置](https://github.com/appstore/asr-core/blob/master/docs/providers.md)。

## 会话契约

- `try_push` 全块接收或返回原音频；队列满返回 `WouldBlock`。文件输入可用 `push_wait`。
- 输入是固定采样率、单声道、有限的 `f32`，范围 `[-1, 1]`。后端重采样由库处理。
- `finish(deadline)` 关闭输入，排空已接收音频，取得唯一终态。重复调用返回同一结果。
- `cancel` 和 Session Drop 会取消会话。Session 自身保留输入所有权，因此 `session.input().try_push(...)` 形式的临时句柄不会意外取消；`finish` 显式关闭输入。
- 原生推理无法强行中断；超时立即结束对外等待，执行槽保留到原生调用退出。
- `SessionFailure` 保留确认文本和接收/处理帧数。错误不会伪装成成功空文本。
- `subscribe()` 只允许取得一次；首次读取一定是 `Update::Reset`。Partial 按 utterance ID 合并，Segment 按输入顺序提交。
- 慢消费者不会影响识别终态；增量队列溢出时下一次读取会得到从权威状态生成的 Reset，自行恢复完整视图。
- 库核心不复制或保留输入录音；需要录音的宿主在音频生产端自行 tee，`utils::audio::Recorder` 是现成的参考实现。
- 可选标点恢复只作用于确认分句，partial 保持原文；字节上限按加标点后的最终文本计。

默认输入队列 2 秒、单块最多 1 秒、会话音频最长 20 分钟、观察增量队列 256 条。每个 Engine 默认最多 8 个活跃会话；Engine clone 共享已准备资源和该执行限额。每次 `Engine::prepare` 都明确创建一组新资源，不做隐式缓存或加载合并。

核心不包含热加载管理器。宿主需要切换模型时，应先在后台调用 `Engine::prepare`，成功后再用自身的锁或原子容器替换当前 `Engine`；既有 Session 继续持有旧 Engine 的资源，新 Session 使用新 Engine。`utils::manager::EngineManager` 按此模式提供了开箱可用的参考实现（带 400 ms 防抖与代际跟踪）：

```rust,no_run
# use std::time::{Duration, Instant};
# use asr_core::utils::manager::EngineManager;
# fn reload(manager: &EngineManager, config: asr_core::EngineConfig) -> Result<(), asr_core::AsrError> {
let generation = manager.request_reload(config);
let snapshot = manager.wait(generation, Instant::now() + Duration::from_secs(30))?;
# let _ = snapshot;
# Ok(())
# }
```

## 可选宿主工具（utils）

`utils` 模块收录多个宿主共同需要、但保持在核心调用链之外的能力，全部为可选增值：核心调用链不依赖它们，宿主也可以用等效的自有实现。按需经 feature 启用：

| 能力 | 路径 | feature |
|---|---|---|
| RMS 电平 | `utils::audio::rms` | 无 |
| PCM16 WAV 编码（与 `read_wav_pcm16` 配对） | `utils::audio::encode_wav_pcm16` | 无 |
| 录音 tee（带上限与截断标志） | `utils::audio::Recorder` | 无 |
| 增量重采样（rubato 封装，`AsrError` 错误） | `utils::audio::Resampler` | 无 |
| 引擎热切换（防抖、代际、显式回退） | `utils::manager::EngineManager` | 无 |
| 模型目录家族探测 | `utils::models::detect` | `backend-sherpa` 或 `punct-sherpa` |
| 配置预检（与 `Engine::prepare` 共用校验器） | `utils::precheck::validate` | 无（本地分支需 `backend-sherpa`/`vad-silero`） |
| 模型资产下载与校验（全程流式、原子发布） | `utils::download::{ensure, fetch}`、`utils::download::VAD_MODEL` | `model-download` |

`utils::download` 使用同步 blocking 客户端，不得在异步 runtime 线程内调用；请在专用线程（如 `spawn_blocking`）中执行。

## 示例与验证

```sh
# OpenAI-compatible HTTP 后端
cargo run --no-default-features --features backend-openai-http --example transcribe -- CONFIG.json speech.wav

# 本地 sherpa-onnx 文件转写（流式 transducer 目录）
cargo run --no-default-features --features backend-sherpa --example transcribe_file -- MODEL_DIR speech.wav

# 本地离线转写（按目录自动识别 SenseVoice / FireRedASR2-AED / Qwen3-ASR / FunASR-Nano；--vad 为 silero VAD 模型文件。
# Paraformer、FireRedASR-CTC 与无标记 SenseVoice 共享扁平布局，无法自动区分，须用 transcribe 示例的 JSON 配置显式指定家族）
cargo run --no-default-features --features backend-sherpa,vad-silero --example transcribe_file -- MODEL_DIR speech.wav --vad silero_vad.onnx

# 本地转写 + 标点恢复（第三个位置参数为标点模型目录；流式 zipformer 去掉 --vad 即可）。
# 仅原始输出无标点的家族需要——流式 zipformer、离线 transducer、Paraformer、FireRedASR2；
# SenseVoice、Qwen3-ASR、FunASR-Nano 原生输出已带标点，无需配置
cargo run --no-default-features --features backend-sherpa,vad-silero,punct-sherpa --example transcribe_file -- MODEL_DIR speech.wav PUNCT_DIR --vad silero_vad.onnx

# 本地转写 + 热词偏置（英文模型需归档自带 bpe.vocab）
cargo run --no-default-features --features backend-sherpa --example transcribe_file -- MODEL_DIR speech.wav --hotwords "语音识别,张三"

# 调整识别线程数（默认 2；单会话文件转写可按机器核数调大换吞吐/延迟，
# 引擎配置字段 num_threads 的取值范围 1–256）
cargo run --no-default-features --features backend-sherpa --example transcribe_file -- MODEL_DIR speech.wav --threads 4

# 显式流式提交；展示 try_push 遇到背压后保留原块并限期重试
cargo run --no-default-features --features backend-sherpa --example streaming -- MODEL_DIR speech.wav

# 麦克风 + HTTP 后端；实时展示 initial/overflow/final Reset、Partial、Segment 与断开
cargo run --no-default-features --features capture-cpal,backend-openai-http --example microphone -- CONFIG.json

# 全量验证
cargo test --all-features --all-targets
cargo test --no-default-features
```

通用示例从 JSON 读取 `EngineConfig`，从 `ASR_API_KEY` 注入凭据。transcribe_file 的实时进度与分段转写输出到 stderr（分段定稿即打印，流式部分结果在交互式终端上单行刷新），完整转写只写 stdout，重定向互不干扰。麦克风示例按 Enter 完整停止。云端运行会向配置的服务上传输入音频。

详细说明：[架构](https://github.com/appstore/asr-core/blob/master/docs/architecture.md)、[后端配置](https://github.com/appstore/asr-core/blob/master/docs/providers.md)、[验证记录](https://github.com/appstore/asr-core/blob/master/docs/validation.md)。真实模型测试单独标记 `ignored`；不在普通测试中下载模型或调用付费服务。未入 CI fixture 的大模型回归（Qwen3-ASR、FunASR-Nano、FireRedASR2 等）还需设 `ASR_RUN_LARGE_MODEL_TEST=1` 才实际执行，区别见[验证记录](https://github.com/appstore/asr-core/blob/master/docs/validation.md)。
