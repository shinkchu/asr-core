//! Verified model asset downloads.
//!
//! The engine takes explicit model paths and never touches the network. For
//! hosts that want to fetch assets on demand, this module streams downloads
//! to a `.part` file while hashing, enforces the expected size, verifies the
//! SHA-256, and only then publishes the file atomically. The `.part` name is
//! unpredictable and opened with `create_new`, so a hostile writer in the
//! destination directory cannot pre-place a symlink for it to follow; the
//! final path is likewise never followed through a symlink. The synchronous
//! blocking client must not be called from an async runtime thread — run it
//! on a dedicated thread (`std::thread::spawn`, `spawn_blocking`, …).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::{AsrError, ErrorKind};

/// A downloadable model asset: source URL, target file name, exact expected
/// size, and expected SHA-256 as lowercase hex. The fields are borrowed so
/// hosts can build assets from configuration at runtime; a `const` asset is
/// simply `ModelAsset<'static>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelAsset<'a> {
    pub url: &'a str,
    pub file_name: &'a str,
    /// Exact expected size in bytes; oversized downloads bail out early.
    pub size_bytes: u64,
    pub sha256: &'a str,
}

/// Silero VAD model required by offline and VAD-gated streaming sessions
/// (about 644 KB).
pub const VAD_MODEL: ModelAsset<'static> = ModelAsset {
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
    file_name: "silero_vad.onnx",
    size_bytes: 643_854,
    sha256: "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
};

static PART_SEQ: AtomicU64 = AtomicU64::new(0);

/// A fresh unpredictable value seeded from the OS entropy behind
/// `RandomState`, so a hostile writer in `dest_dir` cannot guess the `.part`
/// name ahead of time. Std-only on purpose: no dedicated RNG dependency.
fn random_suffix() -> u128 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let a = RandomState::new().build_hasher().finish();
    let b = RandomState::new().build_hasher().finish();
    ((a as u128) << 64) | (b as u128)
}

/// How many unpredictable `.part` candidates `fetch` tries before giving
/// up: every retry draws a fresh suffix, so exhausting these means a hostile
/// or full directory, not bad luck.
const PART_NAME_ATTEMPTS: usize = 8;

/// Fixed streaming chunk size for every SHA-256 path in this module: memory
/// stays bounded whatever the asset size.
const CHUNK: usize = 16 * 1024;

fn http_error(message: impl Into<String>) -> AsrError {
    AsrError::new(ErrorKind::Http, "download", message)
}

fn io_error(message: impl Into<String>) -> AsrError {
    AsrError::new(ErrorKind::Io, "download", message)
}

fn invalid_model(message: impl Into<String>) -> AsrError {
    AsrError::new(ErrorKind::InvalidModel, "download", message)
}

fn resource_limit(message: impl Into<String>) -> AsrError {
    AsrError::new(ErrorKind::ResourceLimit, "download", message)
}

/// Rejects anything that is not a plain file name: `dest_dir.join` would
/// otherwise let `../x` or an absolute path land outside the destination.
fn validate_file_name(name: &str) -> Result<(), AsrError> {
    let plain = !name.is_empty()
        && !name.contains('/')
        && !name.contains('\\')
        && Path::new(name).file_name() == Some(std::ffi::OsStr::new(name));
    if plain {
        Ok(())
    } else {
        Err(AsrError::new(
            ErrorKind::InvalidInput,
            "download",
            "asset file_name must be a plain file name without path separators",
        ))
    }
}

#[derive(Debug)]
enum HashError {
    /// The download/response reader failed (network side).
    Read(std::io::Error),
    /// The sink failed (local disk side).
    Write(std::io::Error),
    /// The running total exceeded `cap`.
    Oversized,
}

/// Streams `reader` in fixed-size chunks, feeding every chunk to `sink` and
/// a SHA-256 hasher, enforcing `cap` on the running total when given.
/// Returns the total bytes read and the hex digest. Shared by the download
/// stream and on-disk file verification so neither path ever holds a whole
/// asset in memory. Read and write failures are reported separately so
/// callers can tell a broken connection from a full disk.
fn stream_hash<R: std::io::Read>(
    reader: &mut R,
    cap: Option<u64>,
    sink: &mut dyn FnMut(&[u8]) -> std::io::Result<()>,
) -> Result<(u64, String), HashError> {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; CHUNK];
    loop {
        let count = reader.read(&mut buffer).map_err(HashError::Read)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .ok_or(HashError::Oversized)?;
        if let Some(cap) = cap {
            if total > cap {
                return Err(HashError::Oversized);
            }
        }
        hasher.update(&buffer[..count]);
        sink(&buffer[..count]).map_err(HashError::Write)?;
    }
    Ok((total, format!("{:x}", hasher.finalize())))
}

/// Whether `path` names a plain regular file, without following a final
/// symlink. `symlink_metadata` inspects the entry itself and `is_file` is
/// false for the link, so a link pointing at a well-formed file outside the
/// destination directory is rejected here; on Windows the same holds for
/// junctions and mount points (the symlink-like reparse points, though not
/// arbitrary ones).
fn is_plain_file(path: &Path) -> bool {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata.is_file(),
        Err(_) => false,
    }
}

/// Whether `path` is a plain file (never a symlink) with the asset's exact
/// size and checksum. The checksum is streamed, so verifying an asset of any
/// size costs a constant amount of memory.
fn file_matches(path: &Path, asset: &ModelAsset<'_>) -> bool {
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return false;
    };
    // No-follow: `is_file` is false for the link itself, so a link pointing
    // at a well-formed asset outside dest_dir must never count as verified,
    // whatever its target contains.
    if !metadata.is_file() {
        return false;
    }
    if metadata.len() != asset.size_bytes {
        return false;
    }
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    stream_hash(&mut std::io::BufReader::new(file), None, &mut |_| Ok(()))
        .is_ok_and(|(_, digest)| digest.eq_ignore_ascii_case(asset.sha256))
}

/// Returns `dest_dir/<file_name>`, downloading the asset first when it is
/// missing or fails its size/checksum verification.
pub fn ensure(asset: &ModelAsset<'_>, dest_dir: &Path) -> Result<PathBuf, AsrError> {
    validate_file_name(asset.file_name)?;
    let path = dest_dir.join(asset.file_name);
    if file_matches(&path, asset) {
        return Ok(path);
    }
    fetch(asset, dest_dir)
}

/// Always downloads the asset to `dest_dir/<file_name>`:
/// streamed to a unique, unpredictable `.part` file opened with `create_new`
/// (so a pre-placed symlink at that path can never be followed), size-capped
/// while streaming, verified against the expected size and SHA-256, then
/// published with an atomic rename. Every failure path removes the `.part`
/// file when it can (the removal is best-effort). A `.part` left by a
/// crashed process is never swept: the library cannot tell it apart from
/// another live process's in-flight download, so hosts that care can clean
/// the directory between runs.
pub fn fetch(asset: &ModelAsset<'_>, dest_dir: &Path) -> Result<PathBuf, AsrError> {
    validate_file_name(asset.file_name)?;
    std::fs::create_dir_all(dest_dir)
        .map_err(|e| io_error(format!("failed to create model directory: {e}")))?;
    // `create_new` fails if the name is taken (by a leftover `.part`, a
    // symlink, anything), so collision retry is the whole defense — there is
    // no remove-then-create window for a symlink to slip into.
    let mut part = None;
    let mut file = None;
    for _ in 0..PART_NAME_ATTEMPTS {
        let candidate = dest_dir.join(format!(
            "{}.{}.{}.{}.part",
            asset.file_name,
            std::process::id(),
            PART_SEQ.fetch_add(1, Ordering::Relaxed),
            random_suffix()
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(handle) => {
                part = Some(candidate);
                file = Some(handle);
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(io_error(format!("failed to create temporary file: {e}"))),
        }
    }
    let (Some(part), Some(mut file)) = (part, file) else {
        return Err(io_error(format!(
            "failed to create a unique temporary file name after {PART_NAME_ATTEMPTS} attempts"
        )));
    };
    let path = dest_dir.join(asset.file_name);

    let result = (|| -> Result<PathBuf, AsrError> {
        let client = reqwest::blocking::Client::builder()
            // The blocking timeout bounds each connect/read/write operation,
            // not the whole transfer, so large assets survive on slow links
            // while a stalled connection fails instead of hanging.
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| http_error(format!("failed to build HTTP client: {e}")))?;
        let mut response = client
            .get(asset.url)
            .send()
            .map_err(|e| http_error(format!("failed to download model: {e}")))?;
        if !response.status().is_success() {
            return Err(http_error(format!(
                "failed to download model: HTTP {}",
                response.status()
            )));
        }
        if let Some(length) = response.content_length() {
            if length != asset.size_bytes {
                return Err(invalid_model(format!(
                    "model download reported {length} bytes but {} were expected",
                    asset.size_bytes
                )));
            }
        }
        stream_and_verify(&mut response, &mut file, asset)?;
        // The part handle stays open across publish: writes already reached
        // the inode being renamed, and std opens files with full sharing
        // flags on Windows, so neither the rename nor cleanup is blocked.
        publish(&part, &path, asset)?;
        Ok(path)
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&part);
    }
    result
}

/// Streaming body: writes the already-opened `.part` file while hashing,
/// then enforces the exact size and checksum. The handle is opened by the
/// caller with `create_new`, so this never re-resolves the path. Generic
/// over the reader for offline tests.
fn stream_and_verify<R: std::io::Read>(
    reader: &mut R,
    file: &mut std::fs::File,
    asset: &ModelAsset<'_>,
) -> Result<(), AsrError> {
    use std::io::Write;

    let (total, actual) = stream_hash(reader, Some(asset.size_bytes), &mut |chunk| {
        file.write_all(chunk)
    })
    .map_err(|error| match error {
        HashError::Oversized => resource_limit(format!(
            "model download exceeded the expected size of {} bytes",
            asset.size_bytes
        )),
        HashError::Read(e) => http_error(format!("model download was interrupted: {e}")),
        HashError::Write(e) => io_error(format!("failed to write model: {e}")),
    })?;
    file.flush()
        .map_err(|e| io_error(format!("failed to flush model: {e}")))?;
    file.sync_all()
        .map_err(|e| io_error(format!("failed to sync model: {e}")))?;
    if total != asset.size_bytes {
        return Err(invalid_model(format!(
            "model download size mismatch: expected {} bytes, got {total}",
            asset.size_bytes
        )));
    }
    if !actual.eq_ignore_ascii_case(asset.sha256) {
        return Err(invalid_model(format!(
            "model checksum mismatch: expected {}, got {actual}",
            asset.sha256
        )));
    }
    Ok(())
}

/// Publishes the verified part file. A valid published target means a
/// concurrent downloader won the race — reuse it and discard our part.
/// Every check goes through no-follow `symlink_metadata`, so a symlink
/// planted at the final path is never accepted and never followed. Both
/// POSIX and Windows rename replace an existing plain file atomically; a
/// symlink is removed first (on POSIX rename replaces the link itself, but
/// dropping it up front keeps the two platforms on one path), and Windows
/// rename would fail on a directory-sized target anyway.
fn publish(part: &Path, path: &Path, asset: &ModelAsset<'_>) -> Result<(), AsrError> {
    if is_plain_file(path) && file_matches(path, asset) {
        let _ = std::fs::remove_file(part);
        return Ok(());
    }
    #[cfg(windows)]
    if std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink()) {
        std::fs::remove_file(path)
            .map_err(|e| io_error(format!("failed to remove invalid model: {e}")))?;
    }
    match std::fs::rename(part, path) {
        Ok(()) => {
            // The rename just moved our own verified regular file into place;
            // anything else means the entry was swapped during publish.
            if is_plain_file(path) {
                Ok(())
            } else {
                let _ = std::fs::remove_file(path);
                Err(io_error("published model path is not a plain file"))
            }
        }
        Err(error) => {
            if file_matches(path, asset) {
                let _ = std::fs::remove_file(part);
                return Ok(());
            }
            let _ = std::fs::remove_file(part);
            Err(io_error(format!("failed to save model: {error}")))
        }
    }
}

#[cfg(test)]
mod tests;
