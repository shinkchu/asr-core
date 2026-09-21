//! Host-side audio conveniences: level metering, WAV encoding, a raw-input
//! recorder tee, and resampling. None of this is on the engine's critical
//! path; the engine core never retains input audio, so a host that wants a
//! copy of the utterance tees it here before pushing it into a session.

use std::time::Duration;

use crate::{AsrError, AudioBuffer, AudioChunk, ErrorKind};

/// Root-mean-square amplitude of mono samples. Empty or silent input is 0.0;
/// non-finite samples propagate (NaN in, NaN out).
///
/// This replaces the 0.3 `Event::Level`: hosts compute the level at the
/// capture or push site instead of receiving it as a session event.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Encode a mono buffer as a complete 16-bit integer PCM WAV byte stream —
/// the inverse of [`crate::audio::read_wav_pcm16`]. Typical uses are
/// persisting a recorded utterance or uploading it to a cloud transcription
/// backend.
pub fn encode_wav_pcm16(buffer: &AudioBuffer) -> Result<Vec<u8>, AsrError> {
    crate::audio::encode_wav_pcm16(&buffer.samples, buffer.spec.sample_rate)
}

/// Collects the raw mono audio handed to a session so the host keeps an
/// archival copy — for example to hand the finished utterance to a second
/// engine or to persist it.
///
/// Tee at the push site: call [`Recorder::push`] with every chunk before it
/// enters [`crate::AudioInput`]. The recorder caps the retained duration and
/// flags overflow through [`Recorder::is_truncated`] instead of failing the
/// session.
pub struct Recorder {
    spec: crate::AudioSpec,
    max_samples: usize,
    samples: Vec<f32>,
    truncated: bool,
}

impl Recorder {
    /// Records at most `limit` worth of audio at `spec`.
    pub fn new(spec: crate::AudioSpec, limit: Duration) -> Self {
        let max_samples = (limit.as_secs_f64() * f64::from(spec.sample_rate))
            .round()
            .clamp(0.0, usize::MAX as f64) as usize;
        Self {
            spec,
            max_samples,
            samples: Vec::new(),
            truncated: false,
        }
    }

    /// Records one chunk. Chunks must match the recorder's sample rate;
    /// input beyond the configured limit is dropped and marked truncated.
    pub fn push(&mut self, chunk: &AudioChunk) -> Result<(), AsrError> {
        if chunk.spec != self.spec {
            return Err(AsrError::new(
                ErrorKind::InvalidInput,
                "recorder",
                format!(
                    "chunk sample rate {} does not match recorder sample rate {}",
                    chunk.spec.sample_rate, self.spec.sample_rate
                ),
            ));
        }
        let remaining = self.max_samples.saturating_sub(self.samples.len());
        if chunk.samples.len() <= remaining {
            self.samples.extend_from_slice(&chunk.samples);
        } else {
            self.samples.extend_from_slice(&chunk.samples[..remaining]);
            self.truncated = true;
        }
        Ok(())
    }

    /// Whether any input was dropped because the limit was reached.
    pub fn is_truncated(&self) -> bool {
        self.truncated
    }

    /// Takes the recording. The buffer may be empty when nothing was pushed.
    pub fn finish(self) -> AudioBuffer {
        AudioBuffer {
            samples: self.samples,
            spec: self.spec,
        }
    }
}

pub use crate::audio::resample::Resampler;

#[cfg(test)]
mod tests;
