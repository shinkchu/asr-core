# 验证记录

> 本文档只保留当前版本的验证记录与可执行信息。早期各轮
> 过程性验证记录（里程碑验证、各轮 review 修复叙事、真实环境
> 验证与性能快照）已完成使命并删除。

## 大模型回归分层开关（2026-09-20）

model-regression workflow 不再用不断增长的 `--skip` 列表排除未入 CI
fixture 的大模型测试。`qwen3_hotwords_and_language_rejection`、
`funasr_nano_hotwords_and_language_rejection`、`fire_red_aed/ctc_*` 与
`sensevoice_markerless_funasr_nano_variant_loads` 改为在测试体内检查
`ASR_RUN_LARGE_MODEL_TEST`（仅 `1` 或 `true` 视为启用）：未启用时打印
一行 skipped 即通过，启用后才实际执行（模型目录仍由各自 `ASR_*_MODEL`
指定）。守卫置于测试体内而非 Cargo feature，这些测试在普通 `cargo test`
下始终参与编译与类型检查，CI 的 `--all-features` 也无需排除任何 feature。

与 Cargo feature 方案对比：`large-model-test` feature 会被 CI 现有的
`--all-features` 一并打开，除非放弃 `--all-features` 手工枚举后端
feature——那只是把 skip 列表换成了 feature 列表；`#[cfg]` 关闭时测试
根本不编译，API 变动会在下次启用时才暴露。

顺带修复：`sensevoice_markerless_funasr_nano_variant_loads` 的模型目录
不是 CI fixture，此前 workflow 既未下载也未 `--skip`，一旦触发即因缺
`ASR_SENSEVOICE_NANO_MODEL` 失败；现并入同一开关，新增模型不再需要改
workflow。

评审修订（同日）：开关语义由纯函数 `large_model_tier_from` 固定并有
单元测试覆盖（仅接受 `1`/`true`，空值、`0`、`false` 等一律不启用；
测试不触碰进程级环境变量，并行安全）。下方「重跑」已拆分为常规
fixture 与大模型两个独立小节，互不误触。新增大模型回归测试只需两步：
测试体开头加 `large_model_tier()` 守卫，并在「重跑（大模型/未入 CI
fixture 层）」补充对应 `ASR_*_MODEL`，无需修改 workflow。

## FireRedASR2 离线家族（2026-09-20）

日期：2026-09-20。平台：Linux x86_64（Intel i7-10510U，AVX2-only，14GB 内存）。Rust：stable（rust-version 1.98）；sherpa-onnx 1.13.8。模型在用户目录（经 hf-mirror.com 镜像下载），仓库不含模型。

新增功能：`OfflineFamily::FireRedAsrAed`（encoder/decoder + tokens.txt，无 joiner；兼容 FireRedASR 1.0-AED-L 的 sherpa-onnx 导出）与 `FireRedAsrCtc`（单 model*.onnx + tokens.txt，布局与 Paraformer 同形，经 `ensure_family_not_contradicted` 防 SenseVoice 误配）。能力规则与 Paraformer 一致：language 覆盖（`InvalidInput`）、transducer bias 与 prompt_hints（引擎级与会话级，`UnsupportedCapability`）全部拒绝；无热词通道。`utils::models::LocalModel` 新增 `FireRedAsrAed` 探测（置于 Qwen3/FunASR 之后——encoder 标记与 Qwen3 布局重叠；CTC 报告为 `Flat`，与 Paraformer 同形一致）。模型来源：HF `csukuangfj2/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25` 与 `…-zh_en-int8-2026-02-26`（解压目录 du 占用约 742MB / 1.2GB），自 ModelScope `FireRedTeam/FireRedASR2-AED`（Apache-2.0）转换，CTC 导出仅含 encoder + CTC 分支。

独立审查修复（三个无实现上下文的 reviewer 并行复审代码/测试/文档后）：**P0** 把完整 transducer 目录（encoder/decoder/joiner/tokens 齐备，如流式 zipformer 归档）配成 `FireRedAsrAed` 时，encoder/decoder/tokens 前缀全部命中而 finder 不查 joiner，穿透到 sherpa 原生层以进程退出收场（实测 exit 255）而非返回 `InvalidModel`——`find_fire_red_aed_files` 现在指名拒绝（"contains joiner*.onnx; this is a transducer layout"）；**P1** 家族标记 `has_fire_red_aed_encoder` 从两个字面文件名改为与 `pick` 一致的前缀口径（官方 transducer 归档的 `encoder-epoch-99-avg-1.onnx` 命名此前绕过标记）；合并 `FireRedAsrCtc` 与 SenseVoice/Paraformer 在 precheck 与 loader 的重复臂；补测试缺口（validate 侧 CTC 矛盾守卫 case + 哨兵、AED prompt_hints 拒绝 case、无标记 flat 目录配 CTC 的 Ok 正向 case、AED-on-transducer 布局指名拒绝、标记的 fp32/epoch 命名分支、wrapped 目录路径归属断言）。

PR 复审跟进（shinkchu，非阻塞意见）：家族标记改名 `has_fire_red_aed_layout_marker`（原名 `has_fire_red_aed_encoder` 暗示"存在 encoder"谓词，实际是 encoder 在场且 joiner 缺席的布局标记）；抽取 `any_prefix_onnx` 前缀判定 helper 供标记与布局守卫共用（与 `pick` 同口径）；providers.md 与 `LocalModel::Flat` 文档补记 CTC 布局与 Paraformer 完全同形、无单向标记、不参与自动探测。另实测确认：为 CTC 补"垃圾文件正向 loader 单测"不可行——sherpa 原生层对垃圾权重抛出不可捕获异常直接 SIGABRT（`Rust cannot catch foreign exceptions`，与 SenseVoice/Paraformer 同为上游行为，任何家族都没有垃圾加载单测），正向 loader 路径由真模型回归测试覆盖。

| 检查 | 结果 |
|---|---|
| `cargo test --all-features --lib` | 196 项通过（含 FireRed serde/布局/预检/加载失败/目录探测/预检一致性新增用例） |
| `cargo test --all-features --doc` | 5 项通过 |
| `cargo clippy --all-features --all-targets -- -D warnings` / `cargo fmt --all -- --check` | 通过 |
| 真模型回归 `fire_red_ctc_baseline_and_capability_rejection` | 通过（4.3s，峰值 RSS 约 1.06GB） |
| 真模型回归 `fire_red_aed_baseline_and_capability_rejection` | 通过（7.5s，峰值 RSS 约 1.63GB） |
| 标点恢复管线（`punct-sherpa`，CT-Transformer 真模型 + FireRedAsrCtc） | 通过，输出见下 |

真模型转写（官方 test_wavs/0.wav，中英混说，经 VAD 切段全管道；模型卡未附官方参考转写，以下为实际输出）：

- FireRedAsrCtc：`昨天是 MAY DAY IS礼拜二THEDAY AFTER TOMORROW是星期三`
- FireRedAsrAed：`昨天是 MONDAY TODAY IS礼拜二THE DAY AFTER TOMORROW是星期三`
- FireRedAsrCtc + CT-Transformer 标点：`昨天是MAY。DAY IS礼拜二。THEDAY AFTER TOMORROW是星期三。`

两种架构对同一中英混说样本输出结构一致（英语片段的逐词选择不同属模型差异）；能力拒绝（language/bias/prompt_hints/session hints → `InvalidInput`/`UnsupportedCapability`）在真模型 prepare/start 路径由回归测试断言。值得记录：int8 量化在本机（AVX2-only，即 FunASR-Nano int8 数值异常的同型机器）表现正常，无重复文本或空输出症状。

验证限制：0.wav 参考转写缺失，转写正确性只能人工比对结构；方言样本（3-sichuan/4-tianjin/5-henan.wav）与 8k 采样样本未纳入回归断言，未单独实测；AED/CTC 的 sherpa 子配置字段接线（encoder/decoder 互换类错误）只有真模型测试能兜底（与 Qwen3/FunASR 同级的接受缺口，CI fixture 不含模型）。

## 真实模型回归重跑

模型资产清单（归档 URL、逐文件 SHA-256、体积与下载映射）保存在
[model-fixtures.json](model-fixtures.json)；已登记的常规 CI fixture 模型
（streaming en / 双语 zipformer / conformer-zh / SenseVoice / Paraformer /
silero VAD）的转写一致性、VAD 分句与纯静音行为由 `--test model_regression`
的对应用例断言。历史输出示例与逐轮真机记录见 git 历史的本文件旧版。

重跑（常规 CI fixture 层）：

```sh
export ASR_STREAMING_MODEL=/path/to/sherpa-onnx-streaming-zipformer-en-2023-06-26
export ASR_STREAMING_BILINGUAL_MODEL=/path/to/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
export ASR_OFFLINE_TRANSDUCER_MODEL=/path/to/sherpa-onnx-conformer-zh-stateless2-2023-05-23
export ASR_SENSEVOICE_MODEL=/path/to/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17
export ASR_PARAFORMER_MODEL=/path/to/sherpa-onnx-paraformer-zh-2023-09-14
export ASR_PARAFORMER_WAV="$ASR_PARAFORMER_MODEL/test_wavs/0.wav"
export ASR_VAD_MODEL=/path/to/silero_vad.onnx
cargo test --all-features --test model_regression -- --ignored --nocapture
```

重跑（大模型/未入 CI fixture 层）：仅在备齐下列 GB 级或独立归档模型后
执行。不设 `ASR_RUN_LARGE_MODEL_TEST`（或设为空值/`false`）时，这些测试
在 `--ignored` 运行中打印 skipped 即通过，不会失败。

```sh
export ASR_RUN_LARGE_MODEL_TEST=1
export ASR_QWEN3_MODEL=/path/to/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25
export ASR_FUNASR_NANO_MODEL=/path/to/sherpa-onnx-funasr-nano-int8-2025-12-30
# FireRedASR2（中英离线，约 742MB/1.2GB），来源见"FireRedASR2 离线家族"一节。
export ASR_FIRE_RED_CTC_MODEL=/path/to/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25
export ASR_FIRE_RED_AED_MODEL=/path/to/sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26
# sense-voice funasr-nano 变体；样本借用经典 sense-voice 归档。
export ASR_SENSEVOICE_NANO_MODEL=/path/to/sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17
export ASR_SENSEVOICE_NANO_WAV="$ASR_SENSEVOICE_MODEL/test_wavs/zh.wav"
# FunASR-Nano 在 AVX2-only 低内存机器上的量化数值问题见 providers.md。
cargo test --all-features --test model_regression -- --ignored --nocapture
```

## 遗留事项（截至 2026-09-18 发布决策）

重构的两份过程文档 TODO.md（执行清单）与 TOFIX.md（审查修复清单）已完成使命并删除。
代码侧工作与可执行的验证均已完成，剩余未完项全部阻塞在外部条件上，摘录如下以便后续追踪：

| 事项 | 阻塞原因 |
|---|---|
| OpenAI Realtime 官方 `server_vad=true` 的收尾关 VAD 屏障 | 需官方 OpenAI key；百炼不允许会话开始后再改 session，已实测不兼容 |
| `input_audio_buffer_commit_empty` 空尾 commit 错误码字面值确认 | 需官方 OpenAI key；百炼对纯静音 commit 的报错无 code，与 OpenAI 行为分叉，是否宽容处理待实测后统一决策 |
| 官方 HTTP SSE `transcript.text.delta/done` 事件帧格式实测 | 需官方 OpenAI key；SiliconFlow 兼容路由为 whisper 风格非流式，无法验证 SSE 假设 |
| Paraformer 官方样本回归 | 缺模型资产 |
| Offline transducer 热词真模型回归（conformer-zh） | 缺模型资产 |
| 英文标点 cnn-bilstm 真机回归 | 缺模型资产 |
| FunASR-Nano 热词/语言回归复验 | 需满足内存且非 AVX2-only 的机器（本机 int8 数值损坏已确定性复现，CI 已 skip） |
| Apple Silicon 平台验证 | 需 Apple Silicon 硬件 |
| 首字延迟单独观测、长会话队列增长测试 | 需真实音频设备 / 数小时级真实会话 |

