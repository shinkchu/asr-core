use super::*;
use crate::{DashScopeConfig, Secret};
use std::sync::mpsc;

fn config(model: &str) -> EngineConfig {
    EngineConfig::DashScope(DashScopeConfig::new(
        "ws://localhost/v1",
        model,
        Secret::new("test-key"),
    ))
}

#[test]
fn stale_load_cannot_publish_and_failure_requires_explicit_fallback() {
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let manager = EngineManager::with_loader(move |config| {
        if let EngineConfig::DashScope(c) = &config {
            if c.model == "old" {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            }
            if c.model == "bad" {
                return Err(AsrError::backend("fixture failure"));
            }
        }
        Engine::prepare(config, EngineOptions::default())
    });
    let old = manager.request_reload(config("old"));
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let new = manager.request_reload(config("new"));
    release_tx.send(()).unwrap();
    assert_eq!(
        manager
            .wait(old, Instant::now() + Duration::from_secs(2))
            .unwrap_err()
            .kind,
        ErrorKind::Cancelled
    );
    assert_eq!(
        manager
            .wait(new, Instant::now() + Duration::from_secs(2))
            .unwrap()
            .generation,
        new
    );
    let bad = manager.request_reload(config("bad"));
    assert!(manager
        .wait(bad, Instant::now() + Duration::from_secs(2))
        .is_err());
    assert!(manager.get().is_none());
    assert_eq!(manager.last_available().unwrap().generation, new);
    let status = manager.status();
    assert_eq!(status.available_generation, Some(new));
    assert!(!status.ready);
}

#[test]
fn wait_times_out_without_a_completed_load() {
    // The debounce window (400 ms) keeps the first request pending past
    // the (already passed) deadline, so wait must report Timeout.
    let manager =
        EngineManager::with_loader(|config| Engine::prepare(config, EngineOptions::default()));
    let generation = manager.request_reload(config("slow"));
    assert_eq!(
        manager.wait(generation, Instant::now()).unwrap_err().kind,
        ErrorKind::Timeout
    );
}

#[test]
fn rapid_requests_collapse_into_one_background_load() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static LOADS: AtomicUsize = AtomicUsize::new(0);
    let manager = EngineManager::with_loader(|config| {
        LOADS.fetch_add(1, Ordering::SeqCst);
        Engine::prepare(config, EngineOptions::default())
    });
    // Both requests land inside the debounce window, so the first target
    // is superseded before the loader ever picks it up.
    let first = manager.request_reload(config("a"));
    let second = manager.request_reload(config("b"));
    assert_ne!(first, second);
    let snapshot = manager
        .wait(second, Instant::now() + Duration::from_secs(2))
        .unwrap();
    assert_eq!(snapshot.generation, second);
    assert_eq!(LOADS.load(Ordering::SeqCst), 1);
    assert!(manager.get().is_some());
    assert_eq!(
        manager.wait(first, Instant::now()).unwrap_err().kind,
        ErrorKind::Cancelled
    );
}
