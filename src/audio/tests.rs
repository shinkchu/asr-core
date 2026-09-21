use super::*;
#[test]
fn invalid_files_return_errors() {
    for bytes in [vec![], b"RIFF".to_vec(), vec![0; 12]] {
        assert!(decode_wav_pcm16(&bytes).is_err());
    }
}
#[test]
fn pcm16_roundtrip_and_endpoints() {
    let data = [-1.0, -0.5, 0.0, 0.5, 1.0];
    let encoded = encode_wav_pcm16(&data, 24000).unwrap();
    let decoded = decode_wav_pcm16(&encoded).unwrap();
    assert_eq!(decoded.spec.sample_rate, 24000);
    assert_eq!(decoded.samples.len(), data.len());
    for (a, b) in data.iter().zip(decoded.samples) {
        assert!((a - b).abs() <= 1.0 / 32768.0);
    }
}
#[test]
fn float_wav_is_rejected() {
    let mut bytes = Cursor::new(Vec::new());
    {
        let mut wav = hound::WavWriter::new(
            &mut bytes,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        wav.write_sample(0.5f32).unwrap();
        wav.finalize().unwrap();
    }
    assert!(decode_wav_pcm16(&bytes.into_inner()).is_err());
}
#[test]
fn unsupported_integer_depths_and_truncated_data_are_rejected() {
    for bits in [8, 24, 32] {
        let mut bytes = Cursor::new(Vec::new());
        {
            let mut wav = hound::WavWriter::new(
                &mut bytes,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 16000,
                    bits_per_sample: bits,
                    sample_format: hound::SampleFormat::Int,
                },
            )
            .unwrap();
            if bits == 8 {
                wav.write_sample(0i8).unwrap();
            } else {
                wav.write_sample(0i32).unwrap();
            }
            wav.finalize().unwrap();
        }
        assert!(decode_wav_pcm16(&bytes.into_inner()).is_err());
    }

    let mut truncated = encode_wav_pcm16(&[0.25], 16000).unwrap();
    truncated.pop();
    assert!(decode_wav_pcm16(&truncated).is_err());
}
#[test]
fn stereo_pcm16_is_downmixed_by_frame() {
    let mut bytes = Cursor::new(Vec::new());
    {
        let mut wav = hound::WavWriter::new(
            &mut bytes,
            hound::WavSpec {
                channels: 2,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for sample in [16384i16, 16384, -32768, 0] {
            wav.write_sample(sample).unwrap();
        }
        wav.finalize().unwrap();
    }
    let decoded = decode_wav_pcm16(&bytes.into_inner()).unwrap();
    assert_eq!(decoded.samples, [0.5, -0.5]);
}
