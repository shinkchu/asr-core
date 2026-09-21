use super::*;

fn manager() -> EngineManager {
    EngineManager::with_loader(|config| Engine::prepare(config, EngineOptions::default()))
}

/// Polls until the loader thread has released its `Arc`, i.e. `observed`
/// holds the only remaining reference. Polling instead of sleeping keeps
/// the test independent of scheduling.
fn assert_loader_exited(observed: &Arc<Inner>) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Arc::strong_count(observed) != 1 {
        assert!(Instant::now() < deadline, "loader thread did not exit");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Two clones dropping concurrently must still shut the loader down.
/// This is exactly the race `Arc::strong_count` cannot handle: each
/// dropping clone observes the other's reference before it is released,
/// so both skip the shutdown path and the worker sleeps forever.
#[test]
fn concurrent_clone_drops_shut_the_loader_down() {
    let manager = manager();
    // Keeps `Inner` alive for observation after both handles are gone.
    let observed = Arc::clone(&manager.inner);
    let a = manager.clone();
    let b = manager;
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let threads = [
        {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                drop(a);
            })
        },
        {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                drop(b);
            })
        },
    ];
    for thread in threads {
        thread.join().unwrap();
    }
    assert_loader_exited(&observed);
}

#[test]
fn dropping_the_last_clone_shuts_the_loader_down() {
    let manager = manager();
    let observed = Arc::clone(&manager.inner);
    drop(manager);
    assert_loader_exited(&observed);
}
