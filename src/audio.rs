//! File input and audio format conversion. Live capture is available with `capture-cpal`.
pub(crate) mod resample;

use crate::{AsrError, ErrorKind};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioSpec {
    pub sample_rate: u32,
}

impl AudioSpec {
    pub fn mono(sample_rate: u32) -> Result<Self, AsrError> {
        if !(8000..=192000).contains(&sample_rate) {
            return Err(AsrError::invalid("sample rate must be in 8000..=192000 Hz"));
        }
        Ok(Self { sample_rate })
    }
}

#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub spec: AudioSpec,
}

impl AudioChunk {
    pub fn mono(samples: Vec<f32>, sample_rate: u32) -> Result<Self, AsrError> {
        Ok(Self {
            samples,
            spec: AudioSpec::mono(sample_rate)?,
        })
    }
}

/// A complete mono recording used by [`crate::Engine::transcribe`].
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub spec: AudioSpec,
}

impl AudioBuffer {
    pub fn mono(samples: Vec<f32>, sample_rate: u32) -> Result<Self, AsrError> {
        Ok(Self {
            samples,
            spec: AudioSpec::mono(sample_rate)?,
        })
    }
}

#[cfg(feature = "backend-openai-http")]
pub(crate) const PCM16_WAV_HEADER_BYTES: usize = 44;
#[cfg(feature = "backend-openai-http")]
pub(crate) const PCM16_BYTES_PER_SAMPLE: usize = 2;
#[cfg(feature = "backend-openai-http")]
pub(crate) fn pcm16_wav_sample_capacity(bytes: usize) -> Option<usize> {
    bytes
        .checked_sub(PCM16_WAV_HEADER_BYTES)
        .map(|payload| payload / PCM16_BYTES_PER_SAMPLE)
}

/// Read an entire 16-bit integer PCM WAV into a mono floating-point buffer.
///
/// All channels are averaged per frame, and the complete file is retained in memory.
/// Float WAVs and integer depths other than 16 bits are rejected.
pub fn read_wav_pcm16(path: impl AsRef<Path>) -> Result<AudioBuffer, AsrError> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| AsrError::new(ErrorKind::Io, "wav", e.to_string()))?;
    read_wav(reader)
}
#[cfg(test)]
pub(crate) fn decode_wav_pcm16(bytes: &[u8]) -> Result<AudioBuffer, AsrError> {
    let reader = hound::WavReader::new(Cursor::new(bytes))
        .map_err(|e| AsrError::new(ErrorKind::InvalidInput, "wav", e.to_string()))?;
    read_wav(reader)
}
fn read_wav<R: std::io::Read>(mut wav: hound::WavReader<R>) -> Result<AudioBuffer, AsrError> {
    let spec = wav.spec();
    if spec.sample_format != hound::SampleFormat::Int
        || spec.bits_per_sample != 16
        || spec.channels == 0
    {
        return Err(AsrError::invalid(
            "expected a 16-bit integer PCM WAV with at least one channel",
        ));
    }
    AudioSpec::mono(spec.sample_rate)?;
    let input = wav
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| AsrError::invalid(e.to_string()))?;
    if input.len() % spec.channels as usize != 0 {
        return Err(AsrError::invalid("incomplete WAV frame"));
    }
    let samples = input
        .chunks_exact(spec.channels as usize)
        .map(|frame| frame.iter().map(|s| *s as f32 / 32768.0).sum::<f32>() / spec.channels as f32)
        .collect();
    AudioBuffer::mono(samples, spec.sample_rate)
}
pub(crate) fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, AsrError> {
    AudioSpec::mono(sample_rate)?;
    let mut bytes = Cursor::new(Vec::new());
    {
        let mut wav = hound::WavWriter::new(
            &mut bytes,
            hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .map_err(|e| AsrError::invalid(e.to_string()))?;
        for s in samples {
            if !s.is_finite() {
                return Err(AsrError::invalid("non-finite WAV sample"));
            }
            wav.write_sample(pcm16(*s))
                .map_err(|e| AsrError::invalid(e.to_string()))?;
        }
        wav.finalize()
            .map_err(|e| AsrError::invalid(e.to_string()))?;
    }
    Ok(bytes.into_inner())
}
pub(crate) fn pcm16(value: f32) -> i16 {
    (value.clamp(-1.0, 1.0) * 32768.0)
        .round()
        .clamp(-32768.0, 32767.0) as i16
}
#[cfg(any(feature = "backend-dashscope", feature = "backend-openai-realtime"))]
pub(crate) fn pcm_bytes(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|s| pcm16(*s).to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests;
