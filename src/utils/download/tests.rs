use super::*;

fn asset<'a>(size_bytes: u64, sha256: &'a str) -> ModelAsset<'a> {
    ModelAsset {
        url: "unused://fixture",
        file_name: "fixture.onnx",
        size_bytes,
        sha256,
    }
}

fn fixture_sha(bytes: &[u8]) -> String {
    use sha2::Digest;
    format!("{:x}", sha2::Sha256::digest(bytes))
}

#[test]
fn streaming_verifies_size_and_checksum() {
    let dir = tempfile::tempdir().unwrap();
    let part = dir.path().join("fixture.onnx.part");
    let content = vec![7u8; 1024];
    let sha = fixture_sha(&content);
    let good = asset(1024, &sha);
    let mut file = std::fs::File::create(&part).unwrap();
    stream_and_verify(&mut content.as_slice(), &mut file, &good).unwrap();

    // Checksum mismatch.
    let error = stream_and_verify(
        &mut content.as_slice(),
        &mut file,
        &asset(1024, &fixture_sha(b"other")),
    )
    .unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(error.message.contains("checksum"), "{error}");

    // Oversized input bails out with a resource limit.
    let error = stream_and_verify(
        &mut content.as_slice(),
        &mut file,
        &asset(16, &fixture_sha(&content)),
    )
    .unwrap_err();
    assert_eq!(error.kind, ErrorKind::ResourceLimit);

    // Truncated input is an invalid model, not a partial success.
    let error = stream_and_verify(&mut &content[..32], &mut file, &good).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidModel);
    assert!(error.message.contains("size mismatch"), "{error}");
}

#[test]
fn publish_prefers_a_valid_concurrent_result_and_cleans_up() {
    let dir = tempfile::tempdir().unwrap();
    let content = vec![3u8; 64];
    let sha = fixture_sha(&content);
    let asset = asset(64, &sha);
    let part = dir.path().join("fixture.onnx.part");
    std::fs::write(&part, &content).unwrap();
    let path = dir.path().join("fixture.onnx");

    // Fresh publish moves the part into place.
    publish(&part, &path, &asset).unwrap();
    assert!(path.is_file());
    assert!(!part.exists());

    // A valid published target makes a second publisher discard its part.
    std::fs::write(&part, &content).unwrap();
    publish(&part, &path, &asset).unwrap();
    assert!(path.is_file());
    assert!(!part.exists());

    // An invalid target is replaced by the new part.
    std::fs::write(&path, b"corrupt").unwrap();
    std::fs::write(&part, &content).unwrap();
    publish(&part, &path, &asset).unwrap();
    assert!(file_matches(&path, &asset));
    assert!(!part.exists());
}

#[test]
fn file_matches_enforces_size_and_checksum() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.onnx");
    let content = vec![5u8; 32];
    let sha = fixture_sha(&content);
    let good = asset(32, &sha);
    std::fs::write(&path, &content).unwrap();
    assert!(file_matches(&path, &good));
    assert!(!file_matches(&path, &asset(64, &sha)));
    assert!(!file_matches(&path, &asset(32, &fixture_sha(b"x"))));
    assert!(!file_matches(&dir.path().join("missing.onnx"), &good));
}

/// Regression: a symlink at the final path is never accepted
/// as a verified asset, even when its target matches size and SHA-256.
#[test]
#[cfg(unix)]
fn file_matches_and_ensure_reject_symlinks_with_matching_targets() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let content = vec![5u8; 32];
    let sha = fixture_sha(&content);
    let good = asset(32, &sha);
    let outside_file = outside.path().join("real.onnx");
    std::fs::write(&outside_file, &content).unwrap();
    let path = dir.path().join("fixture.onnx");
    symlink(&outside_file, &path).unwrap();

    assert!(!file_matches(&path, &good));

    // `ensure` must not hand the link back as a verified asset: it falls
    // through to a download, which fails offline, and the file linked
    // outside dest_dir stays untouched.
    let error = ensure(&good, dir.path()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Http, "{error}");
    assert_eq!(std::fs::read(&outside_file).unwrap(), content);
}

/// Regression: publishing over a planted symlink replaces the
/// link itself without following it or touching the linked file.
#[test]
#[cfg(unix)]
fn publish_replaces_a_symlink_at_the_final_path_without_following_it() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let content = vec![3u8; 64];
    let sha = fixture_sha(&content);
    let asset = asset(64, &sha);
    let outside_file = outside.path().join("real.onnx");
    std::fs::write(&outside_file, b"precious").unwrap();
    let path = dir.path().join("fixture.onnx");
    symlink(&outside_file, &path).unwrap();

    let part = dir.path().join("fixture.onnx.1.2.part");
    std::fs::write(&part, &content).unwrap();
    publish(&part, &path, &asset).unwrap();

    assert!(is_plain_file(&path));
    assert!(file_matches(&path, &asset));
    assert_eq!(std::fs::read(&outside_file).unwrap(), b"precious");
}

#[test]
fn file_matches_verifies_megabyte_files_without_loading_them_into_memory() {
    let dir = tempfile::tempdir().unwrap();
    // A few MB — large enough that whole-file buffering would dominate
    // the memory cost, small enough to keep the test fast. Verification
    // hashes in 16 KiB chunks, so memory stays flat regardless.
    let content: Vec<u8> = (0..4 * 1024 * 1024).map(|i| (i % 251) as u8).collect();
    let path = dir.path().join("fixture.onnx");
    std::fs::write(&path, &content).unwrap();
    let sha = fixture_sha(&content);
    let good = asset(content.len() as u64, &sha);
    assert!(file_matches(&path, &good));

    // One flipped byte fails the checksum even though the size matches.
    let mut tampered = content.clone();
    tampered[1024 * 1024] ^= 0xff;
    std::fs::write(&path, &tampered).unwrap();
    assert!(!file_matches(&path, &good));
}

#[test]
fn vad_asset_keeps_the_official_checksum() {
    assert_eq!(VAD_MODEL.size_bytes, 643_854);
    assert_eq!(VAD_MODEL.file_name, "silero_vad.onnx");
    assert_eq!(
        VAD_MODEL.sha256,
        "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"
    );
}

#[test]
fn ensure_reuses_an_already_valid_file_without_touching_the_network() {
    let dir = tempfile::tempdir().unwrap();
    let content = vec![9u8; 128];
    let sha = fixture_sha(&content);
    let good = asset(128, &sha);
    let path = dir.path().join("fixture.onnx");
    std::fs::write(&path, &content).unwrap();
    let returned = ensure(&good, dir.path()).unwrap();
    assert_eq!(returned, path);
    assert_eq!(std::fs::read(&path).unwrap(), content);
}

#[test]
fn path_like_file_names_are_rejected_before_any_io() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["../evil.onnx", "/abs/evil.onnx", "..", ".", "a\\b.onnx", ""] {
        let bad = ModelAsset {
            url: "unused://fixture",
            file_name: name,
            size_bytes: 1,
            sha256: "0",
        };
        let error = ensure(&bad, dir.path()).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidInput, "{name}: {error}");
        let error = fetch(&bad, dir.path()).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidInput, "{name}: {error}");
    }
    let outside = dir.path().join("../evil.onnx");
    assert!(!outside.exists(), "nothing may be written outside dest_dir");
}

#[test]
fn fetch_failure_reports_http_and_leaves_no_residue() {
    let dir = tempfile::tempdir().unwrap();
    let error = fetch(&asset(64, "0"), dir.path()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Http);
    let residue: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(residue.is_empty(), "unexpected residue: {residue:?}");
}

/// A failing sink is a local disk problem (`Write`), not a network one —
/// `stream_and_verify` maps it to `ErrorKind::Io`, not `Http`.
#[test]
fn stream_hash_reports_sink_failures_as_write_errors() {
    let mut input = &b"abc"[..];
    let error = stream_hash(&mut input, None, &mut |_| {
        Err(std::io::Error::other("disk full"))
    })
    .unwrap_err();
    assert!(matches!(error, HashError::Write(_)), "{error:?}");
}

struct BrokenReader;

impl std::io::Read for BrokenReader {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
        Err(std::io::Error::other("connection reset"))
    }
}

#[test]
fn stream_hash_reports_reader_failures_as_read_errors() {
    let error = stream_hash(&mut BrokenReader, None, &mut |_| Ok(())).unwrap_err();
    assert!(matches!(error, HashError::Read(_)), "{error:?}");
}
