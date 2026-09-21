use super::{model_error, model_layout::ModelFiles};
use crate::{AsrError, BiasPhrase, SpeechHints, TransducerBiasConfig};
use std::{collections::HashSet, path::Path, sync::Arc};

const MAX_PHRASE_CHARS: usize = 64;
const MAX_PHRASES: usize = 256;
#[cfg(feature = "vad-silero")]
const MAX_QWEN3_PROMPT_CHARS: usize = 64;
#[cfg(feature = "vad-silero")]
const MAX_FUNASR_NANO_PROMPT_CHARS: usize = 64;

fn validate_phrase(phrase: &str) -> Result<(), AsrError> {
    if phrase.trim().is_empty() {
        return Err(AsrError::invalid("speech hint phrase must not be empty"));
    }
    if phrase.chars().count() > MAX_PHRASE_CHARS {
        return Err(AsrError::invalid(format!(
            "speech hint phrase must not exceed {MAX_PHRASE_CHARS} characters"
        )));
    }
    if phrase.chars().any(char::is_control) {
        return Err(AsrError::invalid(
            "speech hint phrase must not contain control characters",
        ));
    }
    Ok(())
}

fn validate_count(count: usize) -> Result<(), AsrError> {
    if count > MAX_PHRASES {
        Err(AsrError::invalid(format!(
            "speech hints must not exceed {MAX_PHRASES} phrases"
        )))
    } else {
        Ok(())
    }
}

fn validate_transducer_phrase(phrase: &str) -> Result<(), AsrError> {
    validate_phrase(phrase)?;
    if phrase
        .chars()
        .any(|c| matches!(c, '/' | ':' | ',' | '#' | '@'))
    {
        return Err(AsrError::invalid(
            "transducer hint phrases must not contain '/', ':', ',', '#' or '@'",
        ));
    }
    Ok(())
}

pub(crate) fn validate_bias(config: &TransducerBiasConfig) -> Result<(), AsrError> {
    if !config.default_score.is_finite() {
        return Err(AsrError::invalid(
            "transducer default bias score must be finite",
        ));
    }
    validate_count(config.phrases.len())?;
    if let Some(unit) = &config.modeling_unit {
        if !matches!(unit.as_str(), "cjkchar" | "bpe" | "cjkchar+bpe") {
            return Err(AsrError::invalid(
                "transducer modeling_unit must be cjkchar, bpe or cjkchar+bpe",
            ));
        }
    }
    for phrase in &config.phrases {
        validate_transducer_phrase(&phrase.phrase)?;
        if phrase.score.is_some_and(|score| !score.is_finite()) {
            return Err(AsrError::invalid(
                "per-phrase transducer bias score must be finite",
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_session_hints(hints: &SpeechHints) -> Result<(), AsrError> {
    validate_count(hints.phrases.len())?;
    for phrase in &hints.phrases {
        validate_transducer_phrase(phrase)?;
    }
    Ok(())
}

fn render_bias_phrase(phrase: &BiasPhrase) -> String {
    match phrase.score {
        Some(score) => format!("{} :{score}", phrase.phrase),
        None => phrase.phrase.clone(),
    }
}

pub(crate) fn render_bias(config: &TransducerBiasConfig) -> Option<String> {
    let rendered = config
        .phrases
        .iter()
        .map(render_bias_phrase)
        .collect::<Vec<_>>()
        .join("/");
    (!rendered.is_empty()).then_some(rendered)
}

fn render_session_hints(hints: &SpeechHints) -> Option<String> {
    let rendered = hints.phrases.join("/");
    (!rendered.is_empty()).then_some(rendered)
}

pub(crate) fn merge(
    defaults: Option<&String>,
    session: Option<&SpeechHints>,
) -> Result<Option<String>, AsrError> {
    let session = session.and_then(render_session_hints);
    let merged = match (defaults, session) {
        (None, None) => None,
        (Some(defaults), None) => Some(defaults.clone()),
        (None, Some(session)) => Some(session),
        (Some(defaults), Some(session)) => {
            let total = defaults.split('/').count() + session.split('/').count();
            if total > MAX_PHRASES {
                return Err(AsrError::invalid(format!(
                    "engine and session speech hints combined must not exceed \
                     {MAX_PHRASES} phrases"
                )));
            }
            Some(format!("{defaults}/{session}"))
        }
    };
    Ok(merged)
}

#[cfg(feature = "vad-silero")]
fn validate_prompt(hints: &SpeechHints, family: &str, limit: usize) -> Result<(), AsrError> {
    validate_count(hints.phrases.len())?;
    for phrase in &hints.phrases {
        validate_phrase(phrase)?;
        if phrase.contains(',') {
            return Err(AsrError::invalid(format!(
                "{family} prompt phrase must not contain ','"
            )));
        }
    }
    let total = hints
        .phrases
        .iter()
        .map(|phrase| phrase.chars().count())
        .sum::<usize>()
        .saturating_add(hints.phrases.len().saturating_sub(1));
    if total > limit {
        return Err(AsrError::invalid(format!(
            "{family} prompt hints total {total} characters; keep them within {limit} characters"
        )));
    }
    Ok(())
}

#[cfg(feature = "vad-silero")]
pub(crate) fn qwen3_prompt(hints: &SpeechHints) -> Result<Option<String>, AsrError> {
    validate_prompt(hints, "Qwen3-ASR", MAX_QWEN3_PROMPT_CHARS)?;
    Ok(join_prompt(hints))
}

#[cfg(feature = "vad-silero")]
pub(crate) fn funasr_nano_prompt(hints: &SpeechHints) -> Result<Option<String>, AsrError> {
    validate_prompt(hints, "FunASR-Nano", MAX_FUNASR_NANO_PROMPT_CHARS)?;
    for phrase in &hints.phrases {
        if phrase.chars().any(|c| matches!(c, ';' | '；' | '，')) {
            return Err(AsrError::invalid(format!(
                "FunASR-Nano prompt phrase '{phrase}' must not contain ';', '；' or '，'"
            )));
        }
    }
    Ok(join_prompt(hints))
}

#[cfg(feature = "vad-silero")]
fn join_prompt(hints: &SpeechHints) -> Option<String> {
    let joined = hints.phrases.join(",");
    (!joined.is_empty()).then_some(joined)
}

#[derive(Clone, Debug)]
struct TokenVocabulary {
    unit: String,
    known: Arc<HashSet<String>>,
}

impl TokenVocabulary {
    fn load(unit: &str, tokens_path: &Path) -> Result<Option<Self>, AsrError> {
        if unit != "cjkchar" && unit != "cjkchar+bpe" {
            return Ok(None);
        }
        let tokens =
            std::fs::read_to_string(tokens_path).map_err(|error| model_error(error.to_string()))?;
        let known = tokens
            .lines()
            .filter_map(|line| line.split_whitespace().next())
            .map(str::to_owned)
            .collect();
        Ok(Some(Self {
            unit: unit.to_owned(),
            known: Arc::new(known),
        }))
    }
}

fn validate_model_phrases<'a>(
    vocabulary: &TokenVocabulary,
    phrases: impl IntoIterator<Item = &'a str>,
) -> Result<(), AsrError> {
    let cjkchar_bpe = vocabulary.unit == "cjkchar+bpe";
    for phrase in phrases {
        let missing: Vec<char> = phrase
            .chars()
            .filter(|c| !c.is_whitespace())
            .filter(|c| !(cjkchar_bpe && c.is_ascii_alphabetic()))
            .filter(|c| !vocabulary.known.contains(c.to_string().as_str()))
            .collect();
        if !missing.is_empty() {
            let listed: String = missing.iter().take(8).collect();
            return Err(model_error(format!(
                "speech hint '{phrase}' uses characters missing from the model tokens: {listed}"
            )));
        }
    }
    Ok(())
}

/// Token vocabulary retained by a prepared transducer for validating session hints.
#[derive(Clone, Debug)]
pub(crate) struct HotwordVocabulary {
    vocabulary: Option<TokenVocabulary>,
}

impl HotwordVocabulary {
    pub(crate) fn prepare(
        bias: &TransducerBiasConfig,
        unit: &str,
        tokens_path: &Path,
    ) -> Result<Self, AsrError> {
        let vocabulary = TokenVocabulary::load(unit, tokens_path)?;
        if let Some(vocabulary) = &vocabulary {
            validate_model_phrases(
                vocabulary,
                bias.phrases.iter().map(|phrase| phrase.phrase.as_str()),
            )?;
        }
        Ok(Self { vocabulary })
    }

    pub(crate) fn empty() -> Self {
        Self { vocabulary: None }
    }

    pub(crate) fn validate_session(&self, hints: &SpeechHints) -> Result<(), AsrError> {
        match &self.vocabulary {
            Some(vocabulary) => {
                validate_model_phrases(vocabulary, hints.phrases.iter().map(String::as_str))
            }
            None => Ok(()),
        }
    }
}

/// Bias preparation shared by the streaming/offline loaders and
/// `backends::precheck`: syntax validation, modeling-unit resolution
/// (including the `bpe.vocab` requirement for bpe units) and vocabulary
/// preparation against the model tokens. One home for the sequence keeps
/// the precheck from diverging from what the loaders enforce.
pub(crate) fn prepare_bias(
    bias: &TransducerBiasConfig,
    files: &ModelFiles,
) -> Result<(String, Option<String>, HotwordVocabulary), AsrError> {
    validate_bias(bias)?;
    let (unit, bpe_vocab) = resolve_modeling_unit(bias, files)?;
    let vocabulary = HotwordVocabulary::prepare(bias, &unit, &files.tokens)?;
    Ok((unit, bpe_vocab, vocabulary))
}

pub(crate) fn resolve_modeling_unit(
    bias: &TransducerBiasConfig,
    files: &ModelFiles,
) -> Result<(String, Option<String>), AsrError> {
    let unit = match bias.modeling_unit.as_deref() {
        Some(unit) => unit.to_owned(),
        None => super::model_layout::detect_hotword_modeling_unit(files).0,
    };
    let bpe_vocab = if unit == "cjkchar" {
        None
    } else {
        Some(
            files
                .bpe_vocab
                .as_ref()
                .ok_or_else(|| {
                    model_error(format!(
                        "modeling unit '{unit}' requires bpe.vocab in the model directory; \
                         export it with sherpa-onnx scripts/export_bpe_vocab.py or pick a \
                         model archive that ships it"
                    ))
                })?
                .to_string_lossy()
                .into_owned(),
        )
    };
    Ok((unit, bpe_vocab))
}

#[cfg(test)]
mod tests;
