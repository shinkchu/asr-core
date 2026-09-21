//! Incremental band-limited mono resampling with explicit end-of-input flushing.
use crate::{AsrError, ErrorKind};
use rubato::{
    Resampler as _, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

const BLOCK: usize = 256;

fn resample_error(kind: ErrorKind, message: impl Into<String>) -> AsrError {
    AsrError::new(kind, "resample", message)
}

/// Incremental mono resampler between two fixed rates in 8000..=192000 Hz.
/// Feed input with [`Resampler::try_push`], then call [`Resampler::finish`]
/// exactly once; output is appended to the caller's buffer. Equal rates pass
/// audio through unchanged.
pub struct Resampler {
    src: u32,
    dst: u32,
    filter: Option<SincFixedIn<f32>>,
    in_buf: Vec<f32>,
    out_buf: Vec<f32>,
    pending: Vec<f32>,
    skip: usize,
    input_frames: u64,
    output_frames: u64,
    finished: bool,
}
impl Resampler {
    pub fn try_new(src: u32, dst: u32) -> Result<Self, AsrError> {
        if !(8000..=192000).contains(&src) || !(8000..=192000).contains(&dst) {
            return Err(resample_error(
                ErrorKind::InvalidInput,
                "sample rates must be in 8000..=192000 Hz",
            ));
        }
        let filter = if src == dst {
            None
        } else {
            Some(
                SincFixedIn::<f32>::new(
                    dst as f64 / src as f64,
                    1.0,
                    SincInterpolationParameters {
                        sinc_len: 256,
                        f_cutoff: 0.95,
                        interpolation: SincInterpolationType::Cubic,
                        oversampling_factor: 256,
                        window: WindowFunction::BlackmanHarris2,
                    },
                    BLOCK,
                    1,
                )
                .map_err(|e| {
                    resample_error(
                        ErrorKind::Backend,
                        format!("resampler construction failed: {e}"),
                    )
                })?,
            )
        };
        let skip = filter.as_ref().map(|r| r.output_delay()).unwrap_or(0);
        let in_buf = Vec::with_capacity(BLOCK);
        let mut out_buf = Vec::new();
        if let Some(r) = filter.as_ref() {
            out_buf.resize(r.output_frames_max(), 0.0);
        }
        Ok(Self {
            src,
            dst,
            filter,
            in_buf,
            out_buf,
            pending: Vec::with_capacity(BLOCK * 2),
            skip,
            input_frames: 0,
            output_frames: 0,
            finished: false,
        })
    }
    pub fn try_push(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<(), AsrError> {
        if self.finished {
            return Err(resample_error(
                ErrorKind::InvalidInput,
                "resampler input is closed",
            ));
        }
        if input.iter().any(|s| !s.is_finite()) {
            return Err(resample_error(
                ErrorKind::InvalidInput,
                "non-finite audio sample",
            ));
        }
        self.input_frames += input.len() as u64;
        if self.filter.is_none() {
            out.extend_from_slice(input);
            self.output_frames += input.len() as u64;
            return Ok(());
        }
        // Do not retain a caller's arbitrarily large block in the internal buffer.
        for chunk in input.chunks(BLOCK) {
            self.pending.extend_from_slice(chunk);
            while self.pending.len() >= BLOCK {
                self.in_buf.clear();
                self.in_buf.extend_from_slice(&self.pending[..BLOCK]);
                self.pending.drain(..BLOCK);
                self.process(out)?;
            }
        }
        Ok(())
    }
    fn wanted(&self) -> u64 {
        ((self.input_frames as u128 * self.dst as u128 + self.src as u128 / 2) / self.src as u128)
            as u64
    }
    /// Resample exactly BLOCK frames staged in `in_buf` into `out` through the
    /// pre-allocated output buffer, so per-block processing never allocates.
    fn process(&mut self, out: &mut Vec<f32>) -> Result<(), AsrError> {
        let limit = self.wanted();
        let Self {
            filter,
            in_buf,
            out_buf,
            skip,
            output_frames,
            ..
        } = self;
        let (_, written) = filter
            .as_mut()
            .unwrap()
            .process_into_buffer(&[&in_buf[..]], &mut [&mut out_buf[..]], None)
            .map_err(|e| resample_error(ErrorKind::Backend, format!("resampling failed: {e}")))?;
        let skipped = (*skip).min(written);
        *skip -= skipped;
        let count = (written - skipped).min(limit.saturating_sub(*output_frames) as usize);
        out.extend_from_slice(&out_buf[skipped..skipped + count]);
        *output_frames += count as u64;
        Ok(())
    }
    /// Emit delayed real audio, trim the filter delay, and produce round(N * dst/src) frames.
    /// Padding used internally to flush the filter is never exposed as extra input duration.
    pub fn finish(&mut self, out: &mut Vec<f32>) -> Result<(), AsrError> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        if self.filter.is_none() || self.input_frames == 0 {
            return Ok(());
        }
        // Zero-pad the partial tail block and reuse the staging buffer for the flush.
        self.in_buf.clear();
        self.in_buf.extend_from_slice(&self.pending);
        self.in_buf.resize(BLOCK, 0.0);
        self.pending.clear();
        self.process(out)?;
        self.in_buf.fill(0.0);
        let delay = self.filter.as_ref().unwrap().output_delay();
        let delay_in_input_frames =
            (delay as u128 * self.src as u128).div_ceil(self.dst as u128) as usize;
        // One block covers rounding and one covers the already-padded tail.
        let flush_blocks = delay_in_input_frames.div_ceil(BLOCK) + 2;
        for _ in 0..flush_blocks {
            if self.output_frames == self.wanted() {
                return Ok(());
            }
            self.process(out)?;
        }
        Err(resample_error(
            ErrorKind::Backend,
            "resampler did not flush within the bounded filter tail",
        ))
    }
}

#[cfg(test)]
mod tests;
