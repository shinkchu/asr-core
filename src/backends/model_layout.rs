//! Backend-private local model layout detection and validation.

use std::path::{Path, PathBuf};

/// 本地模型文件集：encoder/decoder/joiner 供流式 transducer 使用，
/// 离线模型只用 encoder（即 model*.onnx）+ tokens。
/// `bpe_vocab` 为热词 BPE 词表（模型目录自带时存在）。
#[derive(Debug, Clone)]
pub(crate) struct ModelFiles {
    pub(crate) encoder: PathBuf,
    pub(crate) decoder: PathBuf,
    pub(crate) joiner: PathBuf,
    pub(crate) tokens: PathBuf,
    pub(crate) bpe_vocab: Option<PathBuf>,
}

/// 流式 transducer：encoder/decoder/joiner/tokens。
/// 优先选择 int8 encoder/joiner（CPU 推理更快），decoder 用非量化版本（精度更好）。
pub(crate) fn find_model_files(dir: &Path) -> Result<ModelFiles, String> {
    let dir = descend_into_single_child(dir);
    let entries = list_dir_files(&dir)?;

    let encoder = pick(&entries, "encoder", true)?;
    let decoder = pick(&entries, "decoder", false)?;
    let joiner = pick(&entries, "joiner", true)?;
    let tokens = require_tokens(&dir)?;

    Ok(ModelFiles {
        encoder,
        decoder,
        joiner,
        tokens,
        bpe_vocab: find_bpe_vocab(&dir)?,
    })
}

/// 离线模型：model*.onnx（优先 int8）+ tokens.txt（SenseVoice / offline Paraformer 等）
pub(crate) fn find_offline_model_files(dir: &Path) -> Result<ModelFiles, String> {
    let dir = descend_into_single_child(dir);
    let entries = list_dir_files(&dir)?;
    let model = pick(&entries, "model", true)?;
    let tokens = require_tokens(&dir)?;

    Ok(ModelFiles {
        encoder: model,
        decoder: PathBuf::new(),
        joiner: PathBuf::new(),
        tokens,
        bpe_vocab: find_bpe_vocab(&dir)?,
    })
}

/// Qwen3-ASR：conv_frontend.onnx + encoder/decoder（优先 int8）+ tokenizer/ 目录。
/// 该家族没有 tokens.txt，tokenizer 目录须含 merges.txt 与 vocab.json。
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "vad-silero"), allow(dead_code))]
pub(crate) struct Qwen3ModelFiles {
    pub(crate) conv_frontend: PathBuf,
    pub(crate) encoder: PathBuf,
    pub(crate) decoder: PathBuf,
    pub(crate) tokenizer: PathBuf,
}

/// 官方归档为 `conv_frontend.onnx`；容忍未来出现 int8 命名的变体。
pub(crate) fn has_conv_frontend(dir: &Path) -> bool {
    dir.join("conv_frontend.onnx").is_file() || dir.join("conv_frontend.int8.onnx").is_file()
}

pub(crate) fn find_qwen3_files(dir: &Path) -> Result<Qwen3ModelFiles, String> {
    let dir = descend_where(dir, has_conv_frontend);
    let entries = list_dir_files(&dir)?;
    let conv_frontend = pick(&entries, "conv_frontend", true)?;
    let encoder = pick(&entries, "encoder", true)?;
    let decoder = pick(&entries, "decoder", true)?;
    let tokenizer = dir.join("tokenizer");
    for required in ["merges.txt", "vocab.json"] {
        let path = tokenizer.join(required);
        if !path.is_file() || path.metadata().is_ok_and(|m| m.len() == 0) {
            return Err(format!("tokenizer directory is missing {required}"));
        }
    }
    Ok(Qwen3ModelFiles {
        conv_frontend,
        encoder,
        decoder,
        tokenizer,
    })
}

/// FunASR-Nano：encoder_adaptor/llm/embedding（优先 int8）+ tokenizer 目录。
/// 官方归档的 tokenizer 目录为 `Qwen3-0.6B/`，须同时含 vocab.json、
/// merges.txt 与 tokenizer.json（上游 `funasr-nano-tokenizer.cc` 三者缺一即退出）。
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "vad-silero"), allow(dead_code))]
pub(crate) struct FunAsrNanoModelFiles {
    pub(crate) encoder_adaptor: PathBuf,
    pub(crate) llm: PathBuf,
    pub(crate) embedding: PathBuf,
    pub(crate) tokenizer: PathBuf,
}

/// `encoder_adaptor*.onnx` 是 FunASR-Nano 目录的家族标记（其他家族不含该前缀）。
pub(crate) fn has_encoder_adaptor(dir: &Path) -> bool {
    dir.join("encoder_adaptor.onnx").is_file() || dir.join("encoder_adaptor.int8.onnx").is_file()
}

/// 扫描子目录中恰好一个含 vocab.json + merges.txt + tokenizer.json 三件套的
/// tokenizer 目录；零个或多个候选都报错（多个即歧义，无法确定性选择）。
fn find_funasr_tokenizer_dir(dir: &Path) -> Result<PathBuf, String> {
    let it = std::fs::read_dir(dir).map_err(|e| format!("failed to read model directory: {e}"))?;
    let mut candidates: Vec<PathBuf> = it
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter(|p| {
            ["vocab.json", "merges.txt", "tokenizer.json"]
                .iter()
                .all(|name| {
                    let file = p.join(name);
                    file.is_file() && file.metadata().is_ok_and(|m| m.len() > 0)
                })
        })
        .collect();
    match candidates.len() {
        1 => Ok(candidates.remove(0)),
        0 => Err(
            "model directory is missing a tokenizer directory with vocab.json, \
             merges.txt and tokenizer.json"
                .into(),
        ),
        _ => Err(
            "multiple tokenizer directories found; point the engine at a directory \
             containing exactly one"
                .into(),
        ),
    }
}

pub(crate) fn find_funasr_nano_files(dir: &Path) -> Result<FunAsrNanoModelFiles, String> {
    let dir = descend_where(dir, has_encoder_adaptor);
    let entries = list_dir_files(&dir)?;
    let encoder_adaptor = pick(&entries, "encoder_adaptor", true)?;
    let llm = pick(&entries, "llm", true)?;
    let embedding = pick(&entries, "embedding", true)?;
    let tokenizer = find_funasr_tokenizer_dir(&dir)?;
    Ok(FunAsrNanoModelFiles {
        encoder_adaptor,
        llm,
        embedding,
        tokenizer,
    })
}

/// FireRedASR-AED：encoder/decoder（均优先 int8，与流式 transducer 的
/// "decoder 用非量化"规则不同）+ tokens.txt。兼容 FireRedASR 1.0-AED-L 与
/// FireRedASR2 的 sherpa-onnx AED 导出（同一布局，同一 sherpa 子配置）。
/// 解析目录含 joiner*.onnx 即流式 transducer 布局，须指名拒绝：encoder/
/// decoder/tokens 前缀在该布局全部命中，放行会穿透到 sherpa 原生层以
/// 进程退出收场，而非返回 InvalidModel。
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "vad-silero"), allow(dead_code))]
pub(crate) struct FireRedAedModelFiles {
    pub(crate) encoder: PathBuf,
    pub(crate) decoder: PathBuf,
    pub(crate) tokens: PathBuf,
}

/// 任一 `encoder*.onnx` 且无任何 `joiner*.onnx` 是 FireRedASR-AED 目录的
/// 家族标记（transducer 目录同样有 encoder，但必然带 joiner）。前缀口径
/// 与 `pick` 一致：transducer 归档的 epoch 后缀命名同样参与判定。
/// 注意这不是"存在 encoder"谓词——joiner 在场即整体为假。
pub(crate) fn has_fire_red_aed_layout_marker(dir: &Path) -> bool {
    match list_dir_files(dir) {
        Ok(entries) => any_prefix_onnx(&entries, "encoder") && !any_prefix_onnx(&entries, "joiner"),
        // 不可读目录对任何标记都给出否定证据，与 descend_where 的语义一致。
        Err(_) => false,
    }
}

/// `pick` 前缀口径的布尔形式：是否存在以 `prefix` 开头、`.onnx` 结尾的
/// 普通文件。家族标记与布局守卫共用，避免两处口径漂移。
fn any_prefix_onnx(entries: &[PathBuf], prefix: &str) -> bool {
    entries.iter().any(|path| {
        path.is_file()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(prefix) && name.ends_with(".onnx"))
    })
}

pub(crate) fn find_fire_red_aed_files(dir: &Path) -> Result<FireRedAedModelFiles, String> {
    let dir = descend_where(dir, has_fire_red_aed_layout_marker);
    let entries = list_dir_files(&dir)?;
    if any_prefix_onnx(&entries, "joiner") {
        return Err(
            "model directory contains joiner*.onnx; this is a transducer layout, not FireRedASR-AED"
                .into(),
        );
    }
    let encoder = pick(&entries, "encoder", true)?;
    let decoder = pick(&entries, "decoder", true)?;
    let tokens = require_tokens(&dir)?;
    Ok(FireRedAedModelFiles {
        encoder,
        decoder,
        tokens,
    })
}

/// 模型目录若自带 bpe.vocab（双语/英文 transducer 归档常见）则返回其路径。
fn find_bpe_vocab(dir: &Path) -> Result<Option<PathBuf>, String> {
    let vocab = dir.join("bpe.vocab");
    if !vocab.is_file() {
        return Ok(None);
    }
    if vocab
        .metadata()
        .map_or(true, |metadata| metadata.len() == 0)
    {
        return Err(
            "bpe.vocab is empty; provide the model's non-empty vocabulary or remove the invalid file"
                .into(),
        );
    }
    Ok(Some(vocab))
}

/// 标点模型文件集：model*.onnx（优先 int8）+ 可选 bpe.vocab。
/// bpe.vocab 存在 → 英文 CNN-BiLSTM（model 与 vocab 都需要）；不存在 → 中英 CT-Transformer（仅 model）。
#[derive(Debug, Clone)]
#[cfg(feature = "punct-sherpa")]
pub(crate) struct PunctModelFiles {
    pub(crate) model: PathBuf,
    pub(crate) vocab: Option<PathBuf>,
}

#[cfg(feature = "punct-sherpa")]
pub(crate) fn find_punct_model_files(dir: &Path) -> Result<PunctModelFiles, String> {
    let dir = descend_where(dir, has_punct_model);
    let entries = list_dir_files(&dir)?;
    let model = pick(&entries, "model", true)?;
    let vocab = find_bpe_vocab(&dir)?;
    Ok(PunctModelFiles { model, vocab })
}

#[cfg(feature = "punct-sherpa")]
fn has_punct_model(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|it| {
        it.filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().ends_with(".onnx"))
    })
}

/// 检测 tokens 是否带 <|zh|> 语言标记。标记存在是经典 SenseVoice 的正向证据；
/// 缺失不是反证（markerless SenseVoice 变体与 Paraformer 的 tokens 生来无法
/// 区分），不得单独作为家族判定依据。
pub(crate) fn is_sense_voice_tokens(tokens: &Path) -> Result<bool, String> {
    std::fs::read_to_string(tokens)
        .map(|s| {
            s.lines()
                .any(|l| l.split_whitespace().next() == Some("<|zh|>"))
        })
        .map_err(|e| format!("failed to read tokens: {e}"))
}

/// 家族预检只做单向证据裁决：tokens 含 <|zh|> 标记 ⇒ 证明是经典 SenseVoice，
/// 配置为其他家族可安全拒绝；标记缺失不构成任何证明（Fun-ASR-Nano 等
/// markerless SenseVoice 变体与 Paraformer 的 tokens 生来无法区分），必须放行，
/// 由 sherpa-onnx 裁决。预检只允许拒绝"加载器必然拒绝"的配置。
///
/// 仅适用于共享"单 onnx + tokens.txt"布局、布局无法互相区分的
/// SenseVoice/Paraformer/FireRedAsrCtc 家族（当前唯一调用点在 `load_offline`）；
/// 其他家族靠目录布局发现（`find_*_files`），无此预检。
#[cfg(feature = "vad-silero")]
pub(crate) fn ensure_family_not_contradicted(
    family: crate::OfflineFamily,
    tokens: &Path,
) -> Result<(), String> {
    // 先读 tokens：读取失败（非法 UTF-8/权限）须如旧实现般以 InvalidModel 暴露，
    // 不能被 SenseVoice 分支短路跳过。
    let sense = is_sense_voice_tokens(tokens)?;
    if !sense || family == crate::OfflineFamily::SenseVoice {
        return Ok(());
    }
    Err(format!(
        "tokens contain SenseVoice language markers but family is {family:?}"
    ))
}

#[cfg(feature = "backend-sherpa")]
pub(crate) fn is_cjk_char(c: char) -> bool {
    // 建模单元推断用的"CJK 字形"判定：覆盖 Unicode 主要表意区块，
    // 含扩展 A~I 与兼容表意区（含补充区）。宽一点只会把含罕见汉字的
    // tokens 判为双语（cjkchar+bpe），用户可用 modeling_unit 显式覆盖。
    matches!(c as u32,
        0x3400..=0x4DBF   // CJK 扩展 A
        | 0x4E00..=0x9FFF // CJK 基本
        | 0xF900..=0xFAFF // CJK 兼容表意
        | 0x20000..=0x2A6DF   // 扩展 B
        | 0x2A700..=0x2B739   // 扩展 C
        | 0x2B740..=0x2B81D   // 扩展 D
        | 0x2B820..=0x2CEA1   // 扩展 E
        | 0x2CEB0..=0x2EBE0   // 扩展 F
        | 0x2EBF0..=0x2EE5D   // 扩展 I
        | 0x31350..=0x323AF   // 扩展 H
        | 0x30000..=0x3134A   // 扩展 G
        | 0x2F800..=0x2FA1F   // 兼容表意补充
    )
}

/// 热词建模单元推断：含 `bpe.vocab` 的目录按 BPE 处理（tokens 含 CJK 字符为
/// 双语 `cjkchar+bpe`，否则纯英文 `bpe`）；无 `bpe.vocab` 视为 `cjkchar`。
/// 返回 (modeling_unit, bpe_vocab 路径)。
#[cfg(feature = "backend-sherpa")]
pub(crate) fn detect_hotword_modeling_unit(files: &ModelFiles) -> (String, Option<PathBuf>) {
    let Some(vocab) = &files.bpe_vocab else {
        return ("cjkchar".into(), None);
    };
    let cjk = std::fs::read_to_string(&files.tokens)
        .map(|s| s.chars().any(is_cjk_char))
        .unwrap_or(false);
    if cjk {
        ("cjkchar+bpe".into(), Some(vocab.clone()))
    } else {
        ("bpe".into(), Some(vocab.clone()))
    }
}

/// 解压包通常带一层同名顶层目录；若传入目录没有模型文件但有唯一子目录，则进入该子目录
fn descend_into_single_child(dir: &Path) -> PathBuf {
    descend_where(dir, |d| d.join("tokens.txt").is_file())
}

/// 解压包通常带一层同名顶层目录；若传入目录没有模型文件但有唯一子目录，则进入该子目录。
/// `utils::models::detect` 也用它解析家族根目录，保证 marker 判断与 `find_*` 同目录。
pub(crate) fn descend_where(dir: &Path, has_model: impl Fn(&Path) -> bool) -> PathBuf {
    if has_model(dir) {
        return dir.to_path_buf();
    }
    if let Ok(mut it) = std::fs::read_dir(dir) {
        let dirs: Vec<PathBuf> = it
            .by_ref()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .map(|e| e.path())
            .collect();
        if dirs.len() == 1 && has_model(&dirs[0]) {
            return dirs[0].clone();
        }
    }
    dir.to_path_buf()
}

fn list_dir_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    let it = std::fs::read_dir(dir).map_err(|e| format!("failed to read model directory: {e}"))?;
    for e in it.filter_map(|e| e.ok()) {
        out.push(e.path());
    }
    Ok(out)
}

fn pick(candidates: &[PathBuf], prefix: &str, prefer_int8: bool) -> Result<PathBuf, String> {
    let matched: Vec<PathBuf> = candidates
        .iter()
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            p.is_file() && name.starts_with(prefix) && name.ends_with(".onnx")
        })
        .cloned()
        .collect();
    if matched.is_empty() {
        return Err(format!("model directory is missing {prefix}*.onnx"));
    }

    let is_preferred = |path: &Path| {
        path.file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.contains(".int8."))
            .unwrap_or(false)
            == prefer_int8
    };
    let preferred_exists = matched.iter().any(|path| is_preferred(path));
    let mut selected: Vec<PathBuf> = matched
        .into_iter()
        .filter(|path| !preferred_exists || is_preferred(path))
        .collect();
    selected.sort_by_key(|path| path.file_name().map(ToOwned::to_owned));
    if selected.len() > 1 {
        let names = selected
            .iter()
            .filter_map(|path| path.file_name())
            .map(|name| name.to_string_lossy())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "multiple equally preferred {prefix} model files found: {names}; keep exactly one candidate in the model directory"
        ));
    }
    let path = selected.remove(0);
    if path.metadata().map_or(true, |m| m.len() == 0) {
        return Err(format!("model file is empty: {}", path.display()));
    }
    Ok(path)
}

fn require_tokens(dir: &Path) -> Result<PathBuf, String> {
    let tokens = dir.join("tokens.txt");
    if tokens.is_file() && tokens.metadata().is_ok_and(|m| m.len() > 0) {
        Ok(tokens)
    } else {
        Err("model directory is missing tokens.txt".into())
    }
}

#[cfg(test)]
pub(crate) mod fixtures;
#[cfg(test)]
mod tests;
