use super::*;
#[test]
fn integer_input_formats_have_matching_zero_and_full_scale() {
    for (signed, unsigned) in [
        (-32768, 0),
        (-16384, 16384),
        (0, 32768),
        (16384, 49152),
        (32767, 65535),
    ] {
        assert_eq!(normalize_i16(signed), normalize_u16(unsigned));
    }
    assert_eq!(normalize_i16(0), 0.0);
    assert_eq!(normalize_u16(32768), 0.0);
    assert_eq!(normalize_i16(-32768), -1.0);
    assert!(normalize_u16(65535) < 1.0);
}
