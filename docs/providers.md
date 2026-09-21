# 后端配置

所有配置通过 `EngineConfig` 显式选择协议。`Secret::new` 注入密钥；序列化时忽略密钥。`prepare` 验证本地配置，云端认证要到 Session 启动时才发生。

默认 feature 为空。HTTP、DashScope、Realtime、本地 sherpa、Silero VAD、sherpa 标点和 CPAL
分别由 `backend-openai-http`、`backend-dashscope`、`backend-openai-realtime`、
`backend-sherpa`、`vad-silero`、`punct-sherpa`、`capture-cpal` 启用。`punct-sherpa`
自动启用本地 sherpa；HTTP 按句模式需要 `backend-openai-http,vad-silero`。

## OpenAI 兼容 HTTP

```rust
use asr_core::*;
let mut config = OpenAiHttpConfig::new(
    "https://provider.example/proxy/v1/", "provider-model", Secret::new("key"));
config.response = HttpResponse::Json;
// 可选：config.language = Some("zh".into());
// 可选：config.prompt = Some("专有名词提示".into());
let backend = EngineConfig::OpenAiHttp(config);
```

接口是根路径下的 `audio/transcriptions`；不要传完整端点，也不要重复 `/v1`。默认 WAV/PCM16/16kHz、multipart file/model/response_format=json。`Text` 使用 response_format=text；`Sse` 使用 json + stream=true。服务不支持相应字段时须选择其实际支持的响应模式；本库不探测或自动重试。

`max_upload_bytes` 限制的是编码后 WAV 音频载荷的字节数（按 WAV 文件计，含文件头），不含 multipart boundary、`model`/`response_format`/`language`/`prompt` 等表单字段与 HTTP 头的开销；整段模式达到上限即报错，按句模式下每次上传前按同一上限校验。按字节数严格限制请求体的代理或网关须在该配置之外预留余量：multipart 编码与头部的固定开销为几 KB 量级，并随 `model`/`language`/`prompt` 等字段值的长度增长。需要边录边按句显示时设置 `HttpMode::Utterances(VadConfig::new("silero_vad.onnx"))`，必须启用 vad-silero、采样率为 16kHz。每句独立上传，可能改变跨句上下文与标点；并发固定为 1。

HTTP 总请求受 response 期限约束；connect 约束请求体开始发送前的连接准备，send 从传输层首次读取请求体开始计时，到完整请求体交给传输层为止。请求体按最多 16 KiB 的块发送，以便感知上传背压；保留 multipart Content-Length。上传后的服务端识别等待使用剩余 response 预算，finish 的绝对截止时间优先。错误保留 HTTP 状态及 x-request-id，不回显服务端原始错误体。重定向关闭，避免把凭据或音频发送到意外端点。

SSE 使用增量解析，只保留尚未完成的事件尾部；收到完整的 transcript.text.done 即确认结果，不等待服务端关闭响应流。标准 `:` 注释心跳会忽略；`data:` 载荷仍须符合所声明 provider 的 JSON 事件协议。done 前的超时、错误或无 done 的 EOF 都是失败，不提交临时文本。

依据：[官方文件转写协议](https://developers.openai.com/api/docs/guides/speech-to-text)。2026-09-18 已在真实服务（SiliconFlow 兼容路由）验证 multipart 上传与 JSON 转写返回；官方 OpenAI SSE 事件帧格式仍待实测。

## DashScope

配置 endpoint、model、api_key、timeouts。对应 run-task / result-generated / finish-task / task-finished 协议。16kHz PCM16 二进制帧；标准 UUID 任务 ID；`heartbeat=true` 的结果忽略，只将 sentence_end=true 视作已确认文本。final 去重只依据服务端身份字段（`sentence_id` 与完整时间区间的组合），没有身份字段时不去重——相同文本的连续句子都会提交。连接关闭不是任务成功。

## OpenAI Realtime

```rust
use asr_core::*;
let mut config = OpenAiRealtimeConfig::new(
    "wss://api.openai.com/v1/realtime?intent=transcription",
    "your-transcription-model", Secret::new("key"));
config.server_vad = true;
let backend = EngineConfig::OpenAiRealtime(config);
```

适配 GA `session.update` 的 `type=transcription` 格式，24kHz PCM16。与旧 beta schema 不混用，不请求助手生成回复。`server_vad=false` 时整段在 finish 提交；true 时服务端分句，finish 关闭自动 VAD 并等待确认，再提交最后一轮。提交前的短静音只提供协议要求的最小缓冲，不写入录音。

按 input_audio_buffer.committed 的顺序分配索引，completed 可跨轮次乱序，按 item_id 去重。未知前驱、缺失终态、失败事件都是错误。第三方必须独立实现并验证实时协议。

依据：[官方 Realtime transcription 协议](https://developers.openai.com/api/docs/guides/realtime-transcription)。2026-09-18 已在真实服务（百炼 `qwen3-asr-flash-realtime` 兼容端点）验证 Manual 模式完整事件链；`server_vad=true` 的收尾关 VAD 屏障与空尾 commit 错误码在官方端点上的表现仍待实测。

## 本地模型

Streaming 读取目录中的 encoder/decoder/joiner/tokens；encoder、joiner 优先 int8，decoder 优先非量化。同一优先级出现多个候选会返回 `InvalidModel`，调用方须只保留一个目标权重文件；不会按字典序暗选。支持单层解压子目录。Offline 读取 model*.onnx 和 tokens.txt，显式指定 SenseVoice 或 Paraformer；词表读取失败、家族不匹配、空文件会报错。Offline family 另支持 `Transducer`（encoder/decoder/joiner/tokens，布局同流式）、`Qwen3Asr`（conv_frontend + encoder/decoder + tokenizer/ 目录，无 tokens.txt）、`FunAsrNano`（encoder_adaptor/llm/embedding + tokenizer/ 目录，tokenizer 目录须同时含 vocab.json、merges.txt、tokenizer.json，无 tokens.txt）、`FireRedAsrAed`（encoder/decoder + tokens.txt，无 joiner；encoder/decoder 均优先 int8——与流式"decoder 用非量化"规则不同；兼容 FireRedASR 1.0-AED-L 与 FireRedASR2 的 AED 导出）与 `FireRedAsrCtc`（单 model*.onnx + tokens.txt，布局与 Paraformer 完全同形、无任何单向标记，因此不参与目录自动探测——`utils::models` 报告为 `Flat`，宿主须显式配置 `FireRedAsrCtc`）；这些家族都不接受语言覆盖。FunASR-Nano 固定贪心解码（temperature 1e-6）并开启 ITN——上游 Rust 绑定的默认值与 C++/Python 相反（全随机采样、ITN 关闭），显式覆盖。

SenseVoice 不再固定中文，默认 auto。Paraformer 不接受语言覆盖。VAD 参数可配置并限制有效范围；模型资产由宿主下载、校验并通过 `VadConfig` 显式传入，库不提供下载器。

### 热词

热词是 transducer 与提示注入式家族（Qwen3-ASR / FunASR-Nano）的能力，推理期按 token 匹配加分或注入提示，无需重训。transducer 配置热词后解码切换到 `modified_beam_search`（开销约为 greedy 的 2~4 倍，仅热词路径承担）；Qwen3-ASR / FunASR-Nano 热词注入生成提示，无解码开销。

| 引擎 | 热词 | 生效位置 | 要求 |
|---|---|---|---|
| 流式 Zipformer | ✅ | `TransducerBiasConfig` + 会话 `SpeechHints` | modified_beam_search；bpe 单元需模型目录含 `bpe.vocab` |
| 离线 Transducer | ✅ | `transducer_bias` + 会话 `SpeechHints` | 同上 |
| Qwen3-ASR | ✅ | Engine 级 `prompt_hints` | 会话级 hints 报 `UnsupportedCapability` |
| FunASR-Nano | ✅ | Engine 级 `prompt_hints` | 同上 |
| SenseVoice / Paraformer | ❌ | — | 配置即报 `UnsupportedCapability` |
| FireRedASR2（AED/CTC） | ❌ | — | 配置即报 `UnsupportedCapability` |

会话级提示要求 transducer 已配置 bias（`Some(TransducerBiasConfig::new(Vec::new()))` 可只启用机制），否则 `start` 报 `UnsupportedCapability`。`capabilities().supports_session_hints` 只反映真实的会话提示能力；不会把 Qwen3/FunASR 的 Engine 级 prompt 混入该能力位。

建模单元自动推断：模型目录含非空 `bpe.vocab` 时按 tokens.txt 是否含 CJK 表意字符取 `cjkchar+bpe`（双语）或 `bpe`（纯英文），不存在时取 `cjkchar`；存在但为空会返回 `InvalidModel`，不会静默切换家族。`TransducerBiasConfig::modeling_unit` 可显式覆盖。cjkchar 系短语的 CJK 字符必须在模型 tokens 表中：默认 bias 在 prepare 校验，会话 hints 在 `start` 用同一张表校验。sherpa-onnx 不读 `bpe.model`，相应英文模型需用其 `scripts/export_bpe_vocab.py` 导出 `bpe.vocab`。transducer 短语每条不超过 64 字符、每组不超过 256 条；`default_score` 默认 2.0，`BiasPhrase::scored` 可按默认短语覆盖。Qwen3/FunASR 的 Engine 级 prompt 总字符数上限为 64；FunASR 另拒绝上游分隔符 `;`、`；`、`，`。

已在官方文档与实测中验证热词可用的模型：

| 模型归档（`asr-models` release） | 家族 | 建模单元 |
|---|---|---|
| `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2` | 流式中英双语 | cjkchar+bpe（归档自带 `bpe.vocab`） |
| `sherpa-onnx-conformer-zh-stateless2-2023-05-23.tar.bz2` | 离线中文 | cjkchar |
| `sherpa-onnx-zipformer-en-2023-04-01.tar.bz2` | 离线英文 | bpe（官方热词示例从该归档取 `bpe.vocab`；HF 镜像缺该文件，取镜像时需导出） |
| `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2` | 离线 Qwen3-ASR | 模型原生（逗号分隔纯文本） |
| `sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2` | 离线 FunASR-Nano | 模型原生（逗号分隔纯文本） |

注意：离线双语 zipformer（`zipformer-zh-en-2023-11-22`）为 byte-level BPE 建模，热词机制不适用；需要中英双语离线热词时使用 Qwen3-ASR 或 FunASR-Nano（同为 LLM 型离线引擎：Qwen3-ASR int8 约 880 MB，FunASR-Nano int8 约 842 MB / 解压 948 MB，LLM 自回归解码在 CPU 上明显慢于 SenseVoice，按需取舍；FunASR-Nano 的差异化优势是中文方言、歌词与说唱识别）。提示词预算：Qwen3-ASR 的 max_total_len 默认 512、热词总字符数上限 64，长音频可能需要调大 max_new_tokens；FunASR-Nano 预算来自 llm.onnx 元数据，同样强制 64 字符上限。**FunASR-Nano 量化数值警示（2026-09-15 实测，i7-10510U / AVX2-only / 14GB 内存）**：int8 量化组件在部分 AVX2-only CPU 上数值异常，症状为成串重复文本、瞬时空输出或流畅乱码（上游 issue #3066 同症状）；上游导出仓库 `https://modelscope.cn/models/zengshuishui/FunASR-nano-onnx` 提供 `llm_int8_compat` 兼容变体（`llm.int8.onnx` 默认版、`llm.int8.u8s8.rr.onnx`、`llm.int8.u8s8.rr.pt.onnx`，按需**替换**目录中的 `llm.int8.onnx`——须替换而非并存：同为 int8 的候选并存会因布局歧义直接返回 `InvalidModel`），本机实测仅 `u8s8.rr.pt` 变体在 fp32 前端下得到与官方参考一致的转写——int8 前端（embedding/encoder_adaptor）在该机同样数值异常，任何 int8 llm 变体都无法纠正；要复现 fp32 前端，须同时删除目录中的 `embedding.int8.onnx` 与 `encoder_adaptor.int8.onnx`（文件探测只要发现 int8 变体即优先选中）；fp32 前端数值正确但三件权重约 3.9GB、fp16 约 1.6GB，加载峰值内存更高，14GB 内存实测触发 systemd-oomd 连锁杀进程——**低内存的 AVX2-only 设备请改用 Qwen3-ASR（int8 同机验证可用）或 SenseVoice**。GitHub release 归档与 HF 镜像内容为 2026-01-13 版本（llm 未替换为 compat），取兼容变体请走上述上游导出仓库。ITN 对日期/百分数等格式的恢复效果有限（上游已知限制）。URL 前缀 `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/`。模型来源及许可证应由使用方按发布方说明核对。

### 标点恢复模型

启用 `punct-sherpa` 后，本地变体可配置 `punctuation: Some(PunctConfig::new(model_dir))`；模型目录手动下载解压，库不自动下载。目录含 `bpe.vocab` 即按英文 CNN-BiLSTM 加载（半角标点加大小写恢复），否则按中英 CT-Transformer 加载；`model*.onnx` 优先 int8，支持单层解压子目录。

| 归档（`punctuation-models` release） | 家族 | 下载 | 解压后 |
|---|---|---|---|
| `sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2` | 中英 CT-Transformer | 62 MB | model.int8.onnx 72 MB |
| `sherpa-onnx-online-punct-en-2024-08-06.tar.bz2` | 英文 CNN-BiLSTM | 29 MB | model.int8.onnx 7.2 MB + bpe.vocab |

URL 前缀 `https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/`，归档与文件校验和见 [model-fixtures.json](model-fixtures.json)。CT-Transformer 对纯英文输入输出全角标点（，。），英文流式模型建议搭配 CNN-BiLSTM。

标点配置即启用，在确认提交时应用一次：partial 保持原文，ID/index 判重先于推理，`max_transcript_bytes` 按加标点后的最终文本计。推理在 Session 状态锁外执行，失败回退原文。CNN-BiLSTM 家族在送入模型前对 ASCII 大写做小写规范化，中文等非 ASCII 字符不受影响。

已在 macOS x86_64 实测的模型及测试音频校验和见 [model-fixtures.json](model-fixtures.json)。模型、音频不提交进仓库。模型来源及许可证应由使用方按发布方说明核对。

## JSON 示例

本地流式（`punctuation`、`bias` 可选）：

```json
{"Streaming":{"model_dir":"/path/to/model","punctuation":{"model":"/path/to/punctuation"},"bias":{"phrases":[{"phrase":"语音识别","score":null},{"phrase":"张三","score":3.5}],"default_score":2.0,"modeling_unit":null}}}
```

离线：

```json
{"Offline":{"model_dir":"/path/to/model","family":"SenseVoice","vad":{"model":"/path/to/silero_vad.onnx","threshold":0.5,"min_silence":0.5,"min_speech":0.25,"max_speech":15.0}}}
```

HTTP（未列出的可选字段使用构造器同款默认值；密钥始终由宿主注入）：

```json
{"OpenAiHttp":{"api_root":"https://api.openai.com/v1","model":"your-transcription-model"}}
```

Realtime 的协议采样率固定为 24 kHz，不是配置字段。未知字段以及 0.3 的遗留字段会被拒绝。
使用 `ASR_API_KEY` 向通用示例注入密钥。不要把生产密钥写进 JSON 或提交到仓库。
