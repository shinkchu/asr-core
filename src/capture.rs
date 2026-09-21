use crate::*;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{traits::*, HeapProd, HeapRb};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

#[derive(Clone, Default)]
struct CaptureFaults {
    overrun: Arc<AtomicBool>,
    invalid_sample: Arc<AtomicBool>,
    device_error: Arc<AtomicBool>,
}

/// Owns microphone capture and its recognition session. Stopping drains the capture ring first.
pub struct CaptureSession {
    session: Session,
    stream: Option<cpal::Stream>,
    stop: Arc<AtomicBool>,
    finished: std::sync::mpsc::Receiver<()>,
}
impl CaptureSession {
    pub fn start(engine: &Engine, mut options: SessionOptions) -> Result<Self, AsrError> {
        let device = cpal::default_host()
            .default_input_device()
            .ok_or_else(|| capture_error("no default input device"))?;
        let supported = device
            .default_input_config()
            .map_err(|_| capture_error("cannot read input configuration"))?;
        options.input = AudioSpec::mono(supported.sample_rate())?;
        let rate = options.input.sample_rate;
        let channels = supported.channels() as usize;
        if channels == 0 {
            return Err(capture_error("input device has zero channels"));
        }
        let geometry = crate::coordinator::session_geometry(&options)?;
        let capacity = geometry.capacity;
        let max_chunk_frames = geometry.max_chunk_frames;
        let batch = max_chunk_frames.min((rate as usize / 50).max(1)).max(1);
        let session = engine.start(options)?;
        let input = session.input();
        let worker_input = input.clone();
        let (producer, mut consumer) = HeapRb::<f32>::new(capacity.max(batch)).split();
        let faults = CaptureFaults::default();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let worker_overrun = faults.overrun.clone();
        let worker_invalid_sample = faults.invalid_sample.clone();
        let worker_error = faults.device_error.clone();
        let config: cpal::StreamConfig = supported.into();
        let stream = match supported.sample_format() {
            cpal::SampleFormat::F32 => {
                build(&device, config, producer, channels, faults, |v: f32| v)
            }
            cpal::SampleFormat::I16 => {
                build(&device, config, producer, channels, faults, normalize_i16)
            }
            cpal::SampleFormat::U16 => {
                build(&device, config, producer, channels, faults, normalize_u16)
            }
            _ => {
                return Err(AsrError::new(
                    ErrorKind::UnsupportedCapability,
                    "capture",
                    "input sample format is unsupported",
                ))
            }
        }?;
        let (tx, finished) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("asr-core-capture-drain".into())
            .spawn(move || {
                'drain: loop {
                    if worker_invalid_sample.load(Ordering::Acquire) {
                        worker_input.fail(AsrError::new(
                            ErrorKind::InvalidInput,
                            "capture",
                            "input device produced a non-finite sample",
                        ));
                        break;
                    }
                    if worker_overrun.load(Ordering::Acquire) {
                        worker_input.fail(AsrError::new(
                            ErrorKind::AudioOverrun,
                            "capture",
                            "capture ring overflowed",
                        ));
                        break;
                    }
                    if worker_error.load(Ordering::Acquire) {
                        worker_input.fail(capture_error("input device stream failed"));
                        break;
                    }
                    let mut samples = Vec::with_capacity(batch);
                    while samples.len() < batch {
                        match consumer.try_pop() {
                            Some(v) => samples.push(v),
                            None => break,
                        }
                    }
                    if !samples.is_empty() {
                        let mut chunk = AudioChunk {
                            samples,
                            spec: AudioSpec { sample_rate: rate },
                        };
                        loop {
                            match worker_input
                                .push_wait(chunk, Instant::now() + Duration::from_millis(20))
                            {
                                Ok(()) => break,
                                Err(error) if error.error.kind == ErrorKind::Timeout => {
                                    chunk = error.chunk;
                                    if worker_overrun.load(Ordering::Acquire) {
                                        worker_input.fail(AsrError::new(
                                            ErrorKind::AudioOverrun,
                                            "capture",
                                            "capture ring overflowed",
                                        ));
                                        break 'drain;
                                    }
                                    if worker_error.load(Ordering::Acquire) {
                                        worker_input
                                            .fail(capture_error("input device stream failed"));
                                        break 'drain;
                                    }
                                }
                                Err(error) => {
                                    worker_input.fail(error.error);
                                    break 'drain;
                                }
                            }
                        }
                    } else if worker_stop.load(Ordering::Acquire) {
                        break;
                    } else {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                }
                let _ = tx.send(());
            })
            .map_err(|_| capture_error("cannot start capture drain worker"))?;
        let result = Self {
            session,
            stream: Some(stream),
            stop,
            finished,
        };
        result
            .stream
            .as_ref()
            .unwrap()
            .play()
            .map_err(|_| capture_error("cannot start microphone"))?;
        Ok(result)
    }
    /// Subscribes to the underlying session; see [`Session::subscribe`] for
    /// the one-shot guarantees (at most one success per session, and dropping
    /// the `Subscription` does not free the slot for re-subscription).
    pub fn subscribe(&self) -> Option<Subscription> {
        self.session.subscribe()
    }
    pub fn cancel(&mut self) {
        self.stream.take();
        self.stop.store(true, Ordering::Release);
        self.session.cancel();
    }
    pub fn finish(&mut self, deadline: Instant) -> SessionResult {
        self.stream.take();
        self.stop.store(true, Ordering::Release);
        if self
            .finished
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .is_err()
        {
            self.session.control().fail(AsrError::new(
                ErrorKind::Timeout,
                "capture drain",
                "capture drain deadline exceeded",
            ));
        }
        self.session.finish(deadline)
    }
}
impl Drop for CaptureSession {
    fn drop(&mut self) {
        self.cancel();
    }
}
fn capture_error(message: &str) -> AsrError {
    AsrError::new(ErrorKind::Io, "capture", message)
}
fn build<T: cpal::SizedSample>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    mut producer: HeapProd<f32>,
    channels: usize,
    faults: CaptureFaults,
    convert: impl Fn(T) -> f32 + Send + 'static,
) -> Result<cpal::Stream, AsrError> {
    let callback_faults = faults.clone();
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                for frame in data.chunks_exact(channels) {
                    let sample = frame.iter().map(|v| convert(*v)).sum::<f32>() / channels as f32;
                    if !sample.is_finite() {
                        callback_faults
                            .invalid_sample
                            .store(true, Ordering::Release);
                        break;
                    }
                    if producer.try_push(sample.clamp(-1.0, 1.0)).is_err() {
                        callback_faults.overrun.store(true, Ordering::Release);
                        break;
                    }
                }
            },
            move |_| {
                faults.device_error.store(true, Ordering::Release);
            },
            None,
        )
        .map_err(|_| capture_error("cannot open input stream"))
}

fn normalize_i16(value: i16) -> f32 {
    value as f32 / 32768.0
}
fn normalize_u16(value: u16) -> f32 {
    (value as f32 - 32768.0) / 32768.0
}

#[cfg(test)]
mod tests;
