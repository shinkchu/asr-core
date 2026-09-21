use super::*;
fn convert(input: &[f32], src: u32, dst: u32, chunk: usize) -> Vec<f32> {
    let mut r = Resampler::try_new(src, dst).unwrap();
    let mut out = Vec::new();
    for c in input.chunks(chunk) {
        r.try_push(c, &mut out).unwrap();
    }
    r.finish(&mut out).unwrap();
    out
}
#[test]
fn length_and_chunk_invariance() {
    for src in [8000, 16000, 24000, 44100, 48000] {
        let input: Vec<_> = (0..src + 17).map(|i| (i as f32 * 0.02).sin()).collect();
        let whole = convert(&input, src, 16000, input.len());
        let split = convert(&input, src, 16000, 137);
        assert_eq!(
            whole.len(),
            ((input.len() as f64 * 16000.0 / src as f64).round()) as usize
        );
        assert_eq!(whole, split);
    }
}
#[test]
fn short_and_empty_inputs_flush() {
    for n in [0, 1, 7, 255, 256, 257] {
        let input = vec![0.5; n];
        assert_eq!(convert(&input, 8000, 16000, 7).len(), n * 2);
    }
}
#[test]
fn passband_and_alias_rejection() {
    let amplitude = |hz: f32| {
        let input: Vec<_> = (0..48000)
            .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / 48000.0).sin())
            .collect();
        let output = convert(&input, 48000, 16000, 239);
        let interior = &output[1000..15000];
        (interior.iter().map(|x| x * x).sum::<f32>() / interior.len() as f32).sqrt()
    };
    assert!((20.0 * (amplitude(6000.0) / std::f32::consts::FRAC_1_SQRT_2).log10()).abs() < 0.5);
    assert!(
        amplitude(12000.0) < 0.007,
        "12 kHz must not alias at full amplitude"
    );
    assert!(amplitude(9000.0) < 0.007);
}
#[test]
fn invalid_rates_and_nonfinite_audio_are_rejected() {
    assert!(Resampler::try_new(0, 16000).is_err());
    let mut r = Resampler::try_new(16000, 16000).unwrap();
    assert!(r.try_push(&[f32::NAN], &mut Vec::new()).is_err());
}
