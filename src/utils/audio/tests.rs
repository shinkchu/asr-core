use super::*;
use crate::audio::read_wav_pcm16;

#[test]
fn rms_is_quiet_for_empty_and_loud_for_constant_input() {
    assert_eq!(rms(&[]), 0.0);
    assert_eq!(rms(&[0.0; 16]), 0.0);
    let constant = vec![0.5; 100];
    assert!((rms(&constant) - 0.5).abs() < 1e-6);
    let mixed = vec![-1.0, 1.0, -1.0, 1.0];
    assert!((rms(&mixed) - 1.0).abs() < 1e-6);
}

#[test]
fn wav_encoding_roundtrips_through_read_wav_pcm16() {
    // Stay inside [-1, 1]: PCM16 encoding clamps out-of-range input, so
    // the roundtrip can only be lossless for in-range samples.
    let buffer =
        AudioBuffer::mono((0..160).map(|i| (i as f32 * 0.05).sin()).collect(), 16000).unwrap();
    let bytes = encode_wav_pcm16(&buffer).unwrap();
    assert_eq!(&bytes[..4], b"RIFF");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("utterance.wav");
    std::fs::write(&path, &bytes).unwrap();
    let decoded = read_wav_pcm16(&path).unwrap();
    assert_eq!(decoded.spec, buffer.spec);
    assert_eq!(decoded.samples.len(), buffer.samples.len());
    for (a, b) in decoded.samples.iter().zip(&buffer.samples) {
        assert!((a - b).abs() < 1e-4);
    }
}

#[test]
fn wav_encoding_rejects_out_of_range_rates() {
    let buffer = AudioBuffer::mono(vec![0.0], 4000).unwrap_err();
    assert_eq!(buffer.kind, ErrorKind::InvalidInput);
}

#[test]
fn recorder_caps_duration_and_flags_truncation() {
    let spec = crate::AudioSpec::mono(16000).unwrap();
    let mut recorder = Recorder::new(spec, Duration::from_millis(10));
    assert_eq!(recorder.max_samples, 160);
    let full = AudioChunk::mono(vec![0.5; 160], 16000).unwrap();
    let extra = AudioChunk::mono(vec![0.5; 40], 16000).unwrap();
    recorder.push(&full).unwrap();
    assert!(!recorder.is_truncated());
    recorder.push(&extra).unwrap();
    assert!(recorder.is_truncated());
    let recorded = recorder.finish();
    assert_eq!(recorded.samples.len(), 160);
    assert_eq!(recorded.spec.sample_rate, 16000);
}

#[test]
fn recorder_rejects_mismatched_sample_rates_and_accepts_empty() {
    let spec = crate::AudioSpec::mono(16000).unwrap();
    let mut recorder = Recorder::new(spec, Duration::from_secs(1));
    let chunk = AudioChunk::mono(vec![0.0; 8], 48000).unwrap();
    let error = recorder.push(&chunk).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    let empty = recorder.finish();
    assert!(empty.samples.is_empty());
    assert_eq!(empty.spec.sample_rate, 16000);
}

#[test]
fn zero_limit_recorder_truncates_everything() {
    let spec = crate::AudioSpec::mono(16000).unwrap();
    let mut recorder = Recorder::new(spec, Duration::ZERO);
    let chunk = AudioChunk::mono(vec![0.5; 8], 16000).unwrap();
    recorder.push(&chunk).unwrap();
    assert!(recorder.is_truncated());
    assert!(recorder.finish().samples.is_empty());
}

#[test]
fn resampler_reexports_publicly_with_typed_errors() {
    let error = Resampler::try_new(0, 16000)
        .err()
        .expect("invalid rates must fail");
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    let mut resampler = Resampler::try_new(8000, 16000).unwrap();
    let mut out = Vec::new();
    resampler.try_push(&[0.5; 100], &mut out).unwrap();
    resampler.finish(&mut out).unwrap();
    assert_eq!(out.len(), 200);
}
