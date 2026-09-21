# 架构与行为

`Engine → Session → AudioInput / Subscription` 是唯一公共调用链。每次 `Engine::prepare` 都明确准备一组资源；Engine clone 共享这些资源，但不同 prepare 调用不做隐式缓存或加载合并。

依赖方向固定为：配置、错误和音频值类型 → Engine/Session 协调层 → 具体 backend；backend
不得反向成为公共模型管理 API。重采样、模型目录探测、HTTP/WAV 编码和协议状态机均为内部实现。
无 feature 的 core 只保留会话与音频基础能力；后端、VAD、标点和 CPAL 通过独立 feature 注入。

## 执行与结束

Engine 用计数许可限制自身的活跃会话。每个活跃会话有一个协调线程，本地推理也在该线程执行；云端在 Engine 内惰性创建并由 clone 共享双线程 Tokio runtime。网络等待保持固定截止时间，并同时监听取消与 finish 的期限变化。Tokio runtime 使用后台关闭，避免析构时阻塞。

输入门、FIFO、帧计数、状态、确认文本、终态结果受同一互斥锁保护。finish 在锁内关闭输入门；协调线程先取完队列，再 flush 重采样器、后端/VAD。cancel 直接发布终态并唤醒输入和网络，不需要排队。所有外部终态只发布一次。

原生 FFI 调用返回前不能释放执行许可。超时后晚到的结果被拒绝。禁止通过不断创建新线程规避许可。

会话 worker 线程是 detached 的，执行许可随线程闭包移动，只有线程真正退出时才释放。终态发布后的唤醒同时作用于 worker 条件变量与网络等待：网络驱动在 `Network::run` 的 select 中响应取消与期限变化，本地推理逐块并在每次原生解码前后执行取消检查，阻塞总能在原生调用之间的检查点解除。已知边界：若某个原生 FFI 调用永不返回，worker 无法到达检查点并退出，其执行许可被永久占用，泄漏累积至并发上限后，新会话启动将被拒绝（`Busy`）。这是有意的设计取舍——许可与真实线程退出绑定，worker 退出前其名额不会被复用；宿主需保证注入的原生调用可终止。

音频使用 rubato 0.16.2（MIT）SincFixedIn，256 帧块、256 tap、Blackman-Harris 窗。维护滤波状态、去除算法延迟，并输出 `round(input_frames × dst/src)` 帧。结束补零只冲洗滤波器；Zipformer 的额外 0.8 秒右上下文不计为输入帧。

## 文本与内存

Partial 按 ID 合并并递增 revision；确认后移除该 ID 的 Partial。已完成分句按索引归并，重复 ID 幂等，冲突索引报 Protocol。重复说相同文字仍是不同分句。最多保留 128 个乱序结果、100000 个完成 ID；HTTP 有上传/响应字节上限，WebSocket 单条消息上限 1 MiB。音频队列同时限制总 frame 数和排队 chunk 数；库核心不复制或保留录音，需要录音的宿主在音频生产端 tee。

可选标点恢复是 ResultSink 确认路径上的文本变换：Engine prepare 时按目录内容识别家族并加载一次模型（无状态对象，跨会话共享 `Arc`），配置模型即对该 Engine 的所有 final 自动生效。ResultStore 先按 ID/index 判重，再在 Session 锁外推理；改写后的文本才进入字节预算、Subscription 和终态 outcome。partial 不做标点化，推理失败回退原文。

Subscription 首次读取、增量溢出和所有终态都交付 `Update::Reset`，视图从同一锁内的权威 Session 状态现场生成。Partial 增量按 ID 合并，Phase 只保留最新值；Segment 进入 ResultStore 后才投递。队列溢出只触发自愈 Reset，不改变识别成败。

失败、超时或取消时，最终 Reset 与 outcome 按索引保留所有已确认分句，包括前面有缺口的乱序结果；保留原始索引和时间戳，不把 Partial 转成确认文本。Subscription Drop 停止维护展示队列，不取消识别；未订阅时不克隆展示字符串。

## 后端边界

- 本地 Streaming：共享 recognizer、每会话独立 stream，端点和 finish 走同一 commit。
- 本地 Offline：显式指定家族，SenseVoice 支持 auto/zh/en/ja/ko/yue；VAD 每会话创建，最长段有上限，结束 flush。另支持 Transducer（encoder/decoder/joiner 布局）、Qwen3Asr（conv_frontend + tokenizer 目录）、FunAsrNano（encoder_adaptor/llm/embedding + tokenizer 目录）与 FireRedASR2（AED：encoder/decoder + tokens；CTC：单 model + tokens，均无热词）。transducer 的 `TransducerBiasConfig` 承载打分与默认短语，并启用无打分承诺的会话 `SpeechHints`；Qwen3Asr/FunAsrNano 只接受 Engine 级 `prompt_hints`，会话提示快速拒绝。
- HTTP：默认整段缓存到配置上限；Utterances 模式依次发送 VAD 完整句，无并发乱序请求。HTTP错误保留状态和 request ID，避免回显可能包含凭据的服务端错误体。
- SSE：delta 归并为完整 Partial，收到完整的 transcript.text.done 后立即确认并结束该句请求，不等待 HTTP EOF；没有 done 就断流仍是失败。
- WebSocket：`WsConnection` 只共享 TLS、握手、有限大小的 send/receive 和错误映射；DashScope 与 Realtime 使用互不共享业务状态的具体 Driver。
- DashScope：标准 UUID，只有 sentence_end 确认分句，只有 task-finished 完成会话；final 按 sentence_id 与完整时间区间的组合幂等，无身份字段时不按文本去重。
- Realtime：GA transcription session；24kHz PCM。Driver 只保留 item 顺序与完成 ID，delta 和 final 文本直接进入 ResultStore；final 先于 committed 时暂存为受预算约束的无索引结果，随后按 committed 顺序原子归位。服务端 VAD 模式结束前关闭自动切段并等待确认，再补充最小静音、显式提交剩余缓冲，等待所有已知轮次完成。

## 采集与热加载

CPAL 回调只混合声道、归一化并写预分配 SPSC 环形缓冲，不分配音频 Vec、不等待网络。普通线程按小批读取，退出时发送不足一批的数据。finish 停止设备流并等待排空，整个等待受同一 deadline 限制。设备错误或环形缓冲溢出终止会话；无自动换设备。

核心不包含热加载管理器。宿主在后台 prepare 新 Engine，并在成功后用自己的同步原语替换当前 Engine；既有 Session 继续持有旧资源。`utils::manager::EngineManager` 按此模式提供开箱可用的参考实现（400 ms 防抖与代际跟踪）。

## 边界

`audio::read_wav_pcm16` 读取完整文件，仅接受整数 PCM16 并平均多声道。超大文件宜自行增量解析后 push。模型只来自调用方明确路径，核心不下载或管理 VAD/ASR/标点资产；`utils::download`（feature `model-download`）提供可选的大小与 SHA-256 校验下载。没有 GUI、粘贴、隐式凭据发现、自动服务回退或自动重传。
