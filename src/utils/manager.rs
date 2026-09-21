//! Background engine hot-swapping.
//!
//! The engine core intentionally has no reload manager: a host that wants
//! hot-swapping prepares a new [`Engine`] in the background and swaps the
//! shared handle itself. This manager is the reference implementation of
//! that pattern — debounced requests, generation tracking, and an
//! explicitly-fallback `get`/`last_available` split.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Condvar, Mutex,
    },
    time::{Duration, Instant},
};

use crate::{AsrError, Engine, EngineConfig, EngineOptions, ErrorKind};

/// A prepared engine tagged with the generation that produced it.
#[derive(Debug, Clone)]
pub struct EngineSnapshot {
    pub generation: u64,
    pub engine: Engine,
}

/// Current reload bookkeeping: the latest requested generation, whether it
/// is still loading, and which generation is currently available.
#[derive(Debug, Clone)]
pub struct ReloadStatus {
    pub generation: u64,
    pub loading: bool,
    /// True when the available snapshot matches the requested generation and
    /// the last load did not fail.
    pub ready: bool,
    pub error: Option<AsrError>,
    pub available_generation: Option<u64>,
}

/// Shared reload state guarded by `state`.
///
/// Invariant: every access goes through `lock().unwrap()`, which is only
/// sound because the critical sections are panic-free — flag writes,
/// `Option` swaps and clones — while the fallible loader runs outside any
/// lock under `catch_unwind`. If a panic ever happens while the lock is
/// held, the mutex poisons and every later `lock().unwrap()` panics in
/// cascade; do not introduce panicking operations into a critical section
/// (the session coordinator pins the same rule for `Shared`).
struct State {
    generation: u64,
    request: Option<(EngineConfig, Instant)>,
    loading: bool,
    available: Option<EngineSnapshot>,
    error: Option<AsrError>,
    worker_error: Option<AsrError>,
    /// Set by the last [`EngineManager`] clone on drop; the loader exits on
    /// sight. Only read and written under the lock, so the wakeup cannot be
    /// lost.
    shutdown: bool,
}

struct Inner {
    /// Live [`EngineManager`] handle count. The worker holds an `Arc` for its
    /// whole life but is not counted here: the clone that takes the counter
    /// to zero is the last manager, even when several clones drop
    /// concurrently. `Arc::strong_count` cannot express this — two clones
    /// dropping on different threads can each observe the other's reference
    /// before it is released, and both would skip the shutdown path.
    managers: AtomicUsize,
    state: Mutex<State>,
    changed: Condvar,
}

/// Keeps one prepared [`Engine`] hot in a background loader thread.
///
/// `request_reload` debounces by 400 ms and bumps a generation; only the
/// latest request is loaded. `get` returns strictly the requested
/// generation; use `last_available` for an explicit fallback to an older
/// engine. Clones share the same manager; the loader thread is event-driven
/// (no polling) and exits when the last clone is dropped.
pub struct EngineManager {
    inner: Arc<Inner>,
}

impl Clone for EngineManager {
    fn clone(&self) -> Self {
        self.inner.managers.fetch_add(1, Ordering::Relaxed);
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Drop for EngineManager {
    fn drop(&mut self) {
        // The worker holds one `Arc` for its whole life and is not part of
        // the handle count, so the clone that takes the count to zero is the
        // last manager — race-free even when several clones drop
        // concurrently. If the spawn itself failed there is no thread and
        // flagging shutdown is harmless.
        if self.inner.managers.fetch_sub(1, Ordering::AcqRel) == 1 {
            let mut s = self.inner.state.lock().unwrap();
            s.shutdown = true;
            self.inner.changed.notify_all();
        }
    }
}

impl EngineManager {
    /// Manages engines prepared with `options` via [`Engine::prepare`].
    pub fn new(options: EngineOptions) -> Self {
        Self::with_loader(move |config| Engine::prepare(config, options.clone()))
    }

    fn with_loader(
        loader: impl Fn(EngineConfig) -> Result<Engine, AsrError> + Send + 'static,
    ) -> Self {
        let inner = Arc::new(Inner {
            managers: AtomicUsize::new(1),
            state: Mutex::new(State {
                generation: 0,
                request: None,
                loading: false,
                available: None,
                error: None,
                worker_error: None,
                shutdown: false,
            }),
            changed: Condvar::new(),
        });
        let spawn = std::thread::Builder::new()
            .name("asr-core-loader".into())
            .spawn({
                let inner = Arc::clone(&inner);
                move || loop {
                    // The thread holds this Arc for its whole life, so
                    // `Inner` outlives every wait below. Exit is signaled by
                    // the last manager handle's `Drop` through `shutdown` —
                    // never detected by polling reference counts.
                    let mut s = inner.state.lock().unwrap();
                    loop {
                        if s.shutdown {
                            return;
                        }
                        match s.request.as_ref() {
                            // A due request falls through to the load below.
                            Some((_, due)) if *due <= Instant::now() => break,
                            // Nothing pending: sleep until the next
                            // `request_reload` or `Drop` notifies.
                            None => s = inner.changed.wait(s).unwrap(),
                            Some((_, due)) => {
                                let delay = due.saturating_duration_since(Instant::now());
                                s = inner.changed.wait_timeout(s, delay).unwrap().0;
                            }
                        }
                    }
                    let (config, _) = s.request.take().unwrap();
                    let generation = s.generation;
                    drop(s);
                    let result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loader(config)))
                            .unwrap_or_else(|_| {
                                Err(AsrError::new(
                                    ErrorKind::Backend,
                                    "load",
                                    "model loader panicked",
                                ))
                            });
                    let mut s = inner.state.lock().unwrap();
                    let mut retired = None;
                    // Generation check and publication share this single state lock.
                    if generation == s.generation {
                        s.loading = false;
                        match result {
                            Ok(engine) => {
                                retired =
                                    s.available.replace(EngineSnapshot { generation, engine });
                                s.error = None;
                            }
                            Err(e) => s.error = Some(e),
                        }
                    }
                    drop(s);
                    inner.changed.notify_all();
                    drop(retired);
                }
            });
        if let Err(e) = spawn {
            let mut s = inner.state.lock().unwrap();
            let error = AsrError::new(ErrorKind::Io, "load", e.to_string());
            s.error = Some(error.clone());
            s.worker_error = Some(error);
        }
        Self { inner }
    }

    /// Replaces the target configuration and returns the new generation.
    /// Rapid successive calls collapse into one background load.
    pub fn request_reload(&self, config: EngineConfig) -> u64 {
        let mut s = self.inner.state.lock().unwrap();
        s.generation += 1;
        s.loading = s.worker_error.is_none();
        s.error = s.worker_error.clone();
        s.request = Some((config, Instant::now() + Duration::from_millis(400)));
        let generation = s.generation;
        drop(s);
        self.inner.changed.notify_all();
        generation
    }

    /// Only the requested generation is returned. Use `last_available` for
    /// an explicit fallback to an older engine.
    pub fn get(&self) -> Option<EngineSnapshot> {
        let s = self.inner.state.lock().unwrap();
        s.available
            .as_ref()
            .filter(|a| a.generation == s.generation && s.error.is_none())
            .cloned()
    }

    /// The newest successfully loaded engine, whatever its generation.
    pub fn last_available(&self) -> Option<EngineSnapshot> {
        self.inner.state.lock().unwrap().available.clone()
    }

    pub fn status(&self) -> ReloadStatus {
        let s = self.inner.state.lock().unwrap();
        ReloadStatus {
            generation: s.generation,
            loading: s.loading,
            ready: s
                .available
                .as_ref()
                .is_some_and(|a| a.generation == s.generation)
                && s.error.is_none(),
            error: s.error.clone(),
            available_generation: s.available.as_ref().map(|a| a.generation),
        }
    }

    /// Blocks until `generation` is loaded, fails, or is superseded by a
    /// newer request (`ErrorKind::Cancelled`), or `deadline` passes
    /// (`ErrorKind::Timeout`).
    pub fn wait(&self, generation: u64, deadline: Instant) -> Result<EngineSnapshot, AsrError> {
        let mut s = self.inner.state.lock().unwrap();
        loop {
            if generation != s.generation {
                return Err(AsrError::new(
                    ErrorKind::Cancelled,
                    "load",
                    "reload was superseded",
                ));
            }
            if let Some(error) = &s.error {
                return Err(error.clone());
            }
            if let Some(snapshot) = &s.available {
                if snapshot.generation == generation {
                    return Ok(snapshot.clone());
                }
            }
            if Instant::now() >= deadline {
                return Err(AsrError::new(
                    ErrorKind::Timeout,
                    "load",
                    "waiting for reload timed out",
                ));
            }
            s = self
                .inner
                .changed
                .wait_timeout(s, deadline.saturating_duration_since(Instant::now()))
                .unwrap()
                .0;
        }
    }
}

#[cfg(all(test, feature = "backend-dashscope"))]
mod tests;

#[cfg(test)]
mod lifecycle_tests;
