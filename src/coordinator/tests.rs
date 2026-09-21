use super::*;
use crate::session::driver::ResultSink as Control;
use std::sync::mpsc;
type FinishMock = Box<dyn Fn(&Control) -> Result<(), AsrError> + Send>;
type PushFn = Box<dyn FnMut(&Control) -> Result<(), AsrError> + Send>;
struct Mock {
    finish: FinishMock,
}
impl Driver for Mock {
    fn push(&mut self, _: &[f32], c: &Control) -> Result<(), AsrError> {
        c.check()
    }
    fn finish(&mut self, c: &Control) -> Result<(), AsrError> {
        (self.finish)(c)
    }
}
fn session(
    f: impl Fn(&Control) -> Result<(), AsrError> + Send + 'static,
    options: SessionOptions,
) -> Session {
    spawn(
        Box::new(Mock {
            finish: Box::new(f),
        }),
        AudioSpec { sample_rate: 16000 },
        options,
        None,
        (),
    )
    .unwrap()
}
fn punctuated_session(
    f: impl Fn(&Control) -> Result<(), AsrError> + Send + 'static,
    options: SessionOptions,
    punctuator: FinalTextProcessor,
) -> Session {
    spawn(
        Box::new(Mock {
            finish: Box::new(f),
        }),
        AudioSpec { sample_rate: 16000 },
        options,
        Some(punctuator),
        (),
    )
    .unwrap()
}
struct PushMock {
    push: PushFn,
}
impl Driver for PushMock {
    fn push(&mut self, _: &[f32], sink: &Control) -> Result<(), AsrError> {
        (self.push)(sink)
    }
    fn finish(&mut self, sink: &Control) -> Result<(), AsrError> {
        sink.check()
    }
}

/// Blocks in `start` until its channel receives, holding the worker
/// thread so deadline and cancel behavior can be exercised.
struct Block(mpsc::Receiver<()>);
impl Driver for Block {
    fn start(&mut self, _: &Control) -> Result<(), AsrError> {
        self.0.recv().unwrap();
        Ok(())
    }
    fn push(&mut self, _: &[f32], _: &Control) -> Result<(), AsrError> {
        Ok(())
    }
    fn finish(&mut self, _: &Control) -> Result<(), AsrError> {
        Ok(())
    }
}

/// Signals `ready` from `start` and then blocks until released.
struct Blocked {
    ready: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
}
impl Driver for Blocked {
    fn start(&mut self, c: &Control) -> Result<(), AsrError> {
        self.ready.send(()).unwrap();
        self.release.recv().unwrap();
        c.check()
    }
    fn push(&mut self, _: &[f32], c: &Control) -> Result<(), AsrError> {
        c.check()
    }
    fn finish(&mut self, c: &Control) -> Result<(), AsrError> {
        c.check()
    }
}

/// Signals `entered` from `finish` and blocks until released.
struct FinishBarrier {
    entered: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
}
impl Driver for FinishBarrier {
    fn push(&mut self, _: &[f32], c: &Control) -> Result<(), AsrError> {
        c.check()
    }
    fn finish(&mut self, c: &Control) -> Result<(), AsrError> {
        self.entered.send(()).unwrap();
        self.release.recv().unwrap();
        c.check()
    }
}

fn push_session(
    push: impl FnMut(&Control) -> Result<(), AsrError> + Send + 'static,
    options: SessionOptions,
) -> Session {
    spawn(
        Box::new(PushMock {
            push: Box::new(push),
        }),
        AudioSpec { sample_rate: 16000 },
        options,
        None,
        (),
    )
    .unwrap()
}
fn wait_until_running(subscription: &mut Subscription) {
    match subscription.recv_timeout(Duration::from_secs(1)).unwrap() {
        Update::Reset(view) if view.phase == SessionPhase::Running => return,
        Update::Reset(view) => assert_eq!(view.phase, SessionPhase::Starting),
        update => panic!("first subscription update was not Reset: {update:?}"),
    }
    loop {
        match subscription.recv_timeout(Duration::from_secs(1)).unwrap() {
            Update::Phase(SessionPhase::Running) => return,
            Update::Reset(view) if view.phase == SessionPhase::Running => return,
            _ => {}
        }
    }
}
#[test]
fn subscription_recv_timeout_does_not_panic_on_duration_max() {
    let s = session(|_| Ok(()), SessionOptions::default());
    let mut subscription = s.subscribe().unwrap();
    s.finish(Instant::now() + Duration::from_secs(1)).unwrap();
    // Most importantly this must not overflow Instant.
    let _ = subscription.recv_timeout(Duration::MAX);
}
#[test]
fn driver_without_poll_interval_sleeps_until_session_work_arrives() {
    struct NoPoll {
        ready: mpsc::Sender<()>,
        polled: mpsc::Sender<()>,
    }
    impl Driver for NoPoll {
        fn start(&mut self, c: &Control) -> Result<(), AsrError> {
            self.ready.send(()).unwrap();
            c.check()
        }
        fn push(&mut self, _: &[f32], c: &Control) -> Result<(), AsrError> {
            c.check()
        }
        fn poll(&mut self, c: &Control) -> Result<(), AsrError> {
            self.polled.send(()).unwrap();
            c.check()
        }
        fn finish(&mut self, c: &Control) -> Result<(), AsrError> {
            c.check()
        }
    }

    let (ready_tx, ready_rx) = mpsc::channel();
    let (poll_tx, poll_rx) = mpsc::channel();
    let s = spawn(
        Box::new(NoPoll {
            ready: ready_tx,
            polled: poll_tx,
        }),
        AudioSpec { sample_rate: 16000 },
        Default::default(),
        None,
        (),
    )
    .unwrap();
    ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(
        poll_rx.recv_timeout(Duration::from_millis(40)),
        Err(mpsc::RecvTimeoutError::Timeout)
    );
    s.finish(Instant::now() + Duration::from_secs(1)).unwrap();
    assert_eq!(poll_rx.try_recv(), Err(mpsc::TryRecvError::Disconnected));
}
#[test]
fn ordered_deduplicated_segments_and_stable_finish() {
    let s = session(
        |c| {
            c.partial("a", "temporary")?;
            c.commit(1, "b", "world", None, None)?;
            c.commit(0, "a", "hello", None, None)?;
            c.commit(0, "a", "hello", None, None)
        },
        Default::default(),
    );
    let result = s.finish(Instant::now() + Duration::from_secs(2)).unwrap();
    assert_eq!(result.transcript.text(), "hello world");
    assert_eq!(
        s.finish(Instant::now()).unwrap().transcript,
        result.transcript
    );
    s.cancel();
    assert_eq!(
        s.finish(Instant::now()).unwrap().transcript,
        result.transcript
    );
}
#[test]
fn result_sink_delta_and_id_registration_use_the_authoritative_store() {
    let s = session(
        |sink| {
            sink.register_id("item-0")?;
            sink.append_delta("item-0", "hel")?;
            sink.append_delta("item-0", "lo")?;
            sink.commit(0, "item-0", "hello", None, None)
        },
        Default::default(),
    );

    let outcome = s.finish(Instant::now() + Duration::from_secs(2)).unwrap();
    assert_eq!(outcome.transcript.text(), "hello");
    assert_eq!(outcome.transcript.segments[0].id, "item-0");
}
#[test]
fn subscription_starts_with_reset_and_is_single_consumer() {
    let s = session(|_| Ok(()), Default::default());
    let mut subscription = s.subscribe().unwrap();
    assert!(s.subscribe().is_none());
    assert!(matches!(subscription.recv(), Some(Update::Reset(_))));
    s.cancel();
}
#[test]
fn slow_subscription_overflow_recovers_without_failing_recognition() {
    let s = session(
        |sink| {
            for index in 0..3 {
                sink.commit(index, index.to_string(), "word", None, None)?;
            }
            Ok(())
        },
        SessionOptions {
            event_capacity: 1,
            ..Default::default()
        },
    );
    let mut subscription = s.subscribe().unwrap();
    wait_until_running(&mut subscription);

    let outcome = s.finish(Instant::now() + Duration::from_secs(1)).unwrap();
    assert_eq!(outcome.transcript.segments.len(), 3);
    let deadline = Instant::now() + Duration::from_secs(1);
    let view = loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match subscription.recv_timeout(remaining).unwrap() {
            Update::Reset(view) => break view,
            Update::Segment(_) | Update::Partial { .. } | Update::Phase(_) => {}
        }
    };
    assert_eq!(view.phase, SessionPhase::Completed);
    assert_eq!(view.transcript, outcome.transcript);
    assert!(view.partials.is_empty());
    assert_eq!(view.queued_frames, 0);
    assert_eq!(
        subscription.recv_timeout(Duration::ZERO),
        Err(mpsc::RecvTimeoutError::Disconnected)
    );
}
#[test]
fn reset_atomically_covers_old_deltas_and_only_new_changes_follow() {
    let mut push_count = 0u64;
    let (overflowed_tx, overflowed_rx) = mpsc::channel();
    let s = push_session(
        move |sink| {
            if push_count == 0 {
                sink.commit(0, "a", "one", None, None)?;
                sink.commit(1, "b", "two", None, None)?;
                overflowed_tx.send(()).unwrap();
            } else {
                sink.commit(2, "c", "three", None, None)?;
            }
            push_count += 1;
            Ok(())
        },
        SessionOptions {
            event_capacity: 1,
            ..Default::default()
        },
    );
    let mut subscription = s.subscribe().unwrap();
    wait_until_running(&mut subscription);
    let input = s.input();
    input
        .push_wait(
            AudioChunk::mono(vec![0.1; 160], 16000).unwrap(),
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
    overflowed_rx.recv_timeout(Duration::from_secs(1)).unwrap();

    let deadline = Instant::now() + Duration::from_secs(1);
    let view = loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match subscription.recv_timeout(remaining).unwrap() {
            Update::Reset(view) => break view,
            Update::Segment(_) | Update::Partial { .. } | Update::Phase(_) => {}
        }
    };
    assert_eq!(view.transcript.text(), "one two");

    input
        .push_wait(
            AudioChunk::mono(vec![0.1; 160], 16000).unwrap(),
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
    let Update::Segment(segment) = subscription.recv_timeout(Duration::from_secs(1)).unwrap()
    else {
        panic!("post-Reset segment was not delivered incrementally");
    };
    assert_eq!(segment.id, "c");
    s.cancel();
}
#[test]
fn completed_finals_remove_partials_from_subscription_views() {
    #[derive(Clone, Copy)]
    enum FinalKind {
        Segment,
        Empty,
        Pending,
        Unindexed,
    }
    for kind in [
        FinalKind::Segment,
        FinalKind::Empty,
        FinalKind::Pending,
        FinalKind::Unindexed,
    ] {
        let s = push_session(
            move |sink| {
                sink.partial("item", "temporary")?;
                match kind {
                    FinalKind::Segment => sink.commit(0, "item", "final", None, None),
                    FinalKind::Empty => sink.commit(0, "item", "", None, None),
                    FinalKind::Pending => sink.commit(1, "item", "final", None, None),
                    FinalKind::Unindexed => sink.complete_unindexed("item", "final"),
                }
            },
            Default::default(),
        );
        let mut subscription = s.subscribe().unwrap();
        wait_until_running(&mut subscription);
        s.input()
            .push_wait(
                AudioChunk::mono(vec![0.1; 160], 16000).unwrap(),
                Instant::now() + Duration::from_secs(1),
            )
            .unwrap();

        loop {
            match subscription.recv_timeout(Duration::from_secs(1)).unwrap() {
                Update::Segment(segment) => {
                    assert!(matches!(kind, FinalKind::Segment));
                    assert_eq!(segment.id, "item");
                    break;
                }
                Update::Reset(view) => {
                    assert!(!matches!(kind, FinalKind::Segment));
                    assert!(view.partials.is_empty());
                    break;
                }
                Update::Partial { .. } | Update::Phase(_) => {}
            }
        }
        s.cancel();
    }
}
#[test]
fn indexing_unindexed_final_with_pending_index_does_not_reset_subscription() {
    let s = push_session(
        |sink| {
            sink.partial("item", "temporary")?;
            sink.complete_unindexed("item", "final")?;
            // Committing at a pending index drains no segments, so the
            // already-removed partial must not trigger a second Reset.
            sink.commit(1, "item", "", None, None)
        },
        Default::default(),
    );
    let mut subscription = s.subscribe().unwrap();
    wait_until_running(&mut subscription);
    s.input()
        .push_wait(
            AudioChunk::mono(vec![0.1; 160], 16000).unwrap(),
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();

    let mut reset = None;
    while reset.is_none() {
        match subscription.recv_timeout(Duration::from_secs(1)).unwrap() {
            // Stale updates may still be in flight ahead of the Reset.
            Update::Partial { .. } | Update::Phase(_) => {}
            Update::Segment(_) => panic!("pending-index commit must not emit segments"),
            Update::Reset(view) => reset = Some(view),
        }
    }
    assert!(reset.unwrap().partials.is_empty());
    assert!(matches!(
        subscription.recv_timeout(Duration::from_millis(200)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    s.cancel();
}
#[test]
fn indexing_unindexed_final_forwards_commit_timestamps() {
    let s = session(
        |sink| {
            sink.complete_unindexed("a", "final")?;
            sink.commit(0, "a", "", Some(1.0), Some(2.0))
        },
        Default::default(),
    );
    let result = s.finish(Instant::now() + Duration::from_secs(2)).unwrap();
    let segment = &result.transcript.segments[0];
    assert_eq!(segment.text, "final");
    assert_eq!(segment.start_seconds, Some(1.0));
    assert_eq!(segment.end_seconds, Some(2.0));
}
#[test]
fn dropped_or_absent_subscription_never_retains_updates_or_fails_session() {
    let s = session(
        |sink| {
            for index in 0..150 {
                sink.commit(index, index.to_string(), "word", None, None)?;
            }
            Ok(())
        },
        SessionOptions {
            event_capacity: 1,
            ..Default::default()
        },
    );
    let mut subscription = s.subscribe().unwrap();
    wait_until_running(&mut subscription);
    drop(subscription);
    let outcome = s.finish(Instant::now() + Duration::from_secs(1)).unwrap();
    assert_eq!(outcome.transcript.segments.len(), 150);
    let state = s.control.0.state.lock().unwrap();
    assert!(state.observer.updates.is_empty());
    assert!(!state.observer.subscription_enabled);
}
#[test]
fn terminal_subscription_reset_is_delivered_before_disconnect() {
    let s = session(
        |sink| sink.commit(0, "a", "confirmed", None, None),
        Default::default(),
    );
    let outcome = s.finish(Instant::now() + Duration::from_secs(1)).unwrap();
    let mut subscription = s.subscribe().unwrap();
    let Update::Reset(view) = subscription.recv().unwrap() else {
        panic!("terminal subscription did not start with Reset");
    };
    assert_eq!(view.phase, SessionPhase::Completed);
    assert_eq!(view.transcript, outcome.transcript);
    assert!(subscription.recv().is_none());
}
#[test]
fn subscription_timeout_does_not_change_session_result() {
    let s = session(|_| Ok(()), Default::default());
    let mut subscription = s.subscribe().unwrap();
    wait_until_running(&mut subscription);
    assert_eq!(
        subscription.recv_timeout(Duration::from_millis(1)),
        Err(mpsc::RecvTimeoutError::Timeout)
    );
    assert!(s.finish(Instant::now() + Duration::from_secs(1)).is_ok());
    assert!(matches!(subscription.recv(), Some(Update::Reset(_))));
    assert!(subscription.recv().is_none());
}
#[test]
fn missing_predecessor_is_failure() {
    let s = session(
        |c| c.commit(1, "b", "later", None, None),
        Default::default(),
    );
    let failure = s
        .finish(Instant::now() + Duration::from_secs(2))
        .unwrap_err();
    assert_eq!(failure.error.kind, ErrorKind::Protocol);
    assert_eq!(failure.outcome.transcript.text(), "later");
    assert_eq!(failure.outcome.transcript.segments[0].index, 1);
    assert_eq!(
        s.finish(Instant::now()).unwrap_err().outcome.transcript,
        failure.outcome.transcript
    );
}
#[test]
fn failure_retains_all_confirmed_segments_in_input_order() {
    let s = session(
        |c| {
            c.commit(0, "first", "first", None, None)?;
            c.commit(4, "empty", " ", None, None)?;
            c.commit(3, "fourth", "fourth", Some(3.0), Some(4.0))?;
            c.commit(2, "third", "third", Some(2.0), Some(3.0))?;
            c.partial("second", "unconfirmed")?;
            Err(AsrError::backend("second utterance failed"))
        },
        Default::default(),
    );
    let mut subscription = s.subscribe().unwrap();
    wait_until_running(&mut subscription);
    let failure = s
        .finish(Instant::now() + Duration::from_secs(2))
        .unwrap_err();
    assert_eq!(failure.error.kind, ErrorKind::Backend);
    let transcript = &failure.outcome.transcript;
    assert_eq!(transcript.text(), "first third fourth");
    assert_eq!(
        transcript
            .segments
            .iter()
            .map(|s| s.index)
            .collect::<Vec<_>>(),
        [0, 2, 3]
    );
    assert_eq!(transcript.segments[1].id, "third");
    assert_eq!(transcript.segments[1].start_seconds, Some(2.0));
    assert_eq!(transcript.segments[1].end_seconds, Some(3.0));
    let Update::Reset(view) = subscription.recv().unwrap() else {
        panic!("backend failure did not produce final Reset");
    };
    assert_eq!(view.phase, SessionPhase::Failed);
    assert_eq!(view.error, Some(failure.error.clone()));
    assert_eq!(view.transcript, failure.outcome.transcript);
    assert_eq!(view.received_frames, failure.outcome.received_frames);
    assert_eq!(view.processed_frames, failure.outcome.processed_frames);
    assert_eq!(view.queued_frames, 0);
    assert!(view.partials.is_empty());
    assert_eq!(
        s.finish(Instant::now()).unwrap_err().outcome.transcript,
        *transcript
    );
}
#[test]
fn timeout_and_cancel_retain_out_of_order_confirmed_segments() {
    for cancel in [false, true] {
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        struct Blocked {
            ready: mpsc::Sender<()>,
            release: mpsc::Receiver<()>,
        }
        impl Driver for Blocked {
            fn start(&mut self, c: &Control) -> Result<(), AsrError> {
                c.commit(1, "later", "confirmed", None, None)?;
                self.ready.send(()).unwrap();
                self.release.recv().unwrap();
                c.check()
            }
            fn push(&mut self, _: &[f32], c: &Control) -> Result<(), AsrError> {
                c.check()
            }
            fn finish(&mut self, c: &Control) -> Result<(), AsrError> {
                c.check()
            }
        }
        let s = spawn(
            Box::new(Blocked {
                ready: ready_tx,
                release: release_rx,
            }),
            AudioSpec { sample_rate: 16000 },
            Default::default(),
            None,
            (),
        )
        .unwrap();
        ready_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        // Initial Reset exposes only the contiguous transcript prefix.
        let mut subscription = s.subscribe().unwrap();
        let Update::Reset(view) = subscription.recv().unwrap() else {
            panic!("subscription did not start with Reset");
        };
        assert!(view.transcript.segments.is_empty());
        if cancel {
            s.cancel();
        }
        let failure = s
            .finish(Instant::now() + Duration::from_millis(40))
            .unwrap_err();
        release_tx.send(()).unwrap();
        assert_eq!(
            failure.error.kind,
            if cancel {
                ErrorKind::Cancelled
            } else {
                ErrorKind::Timeout
            }
        );
        assert_eq!(failure.outcome.transcript.text(), "confirmed");
        let Update::Reset(view) = subscription.recv().unwrap() else {
            panic!("terminal state did not produce Reset");
        };
        assert_eq!(view.error, Some(failure.error.clone()));
        assert_eq!(view.transcript, failure.outcome.transcript);
        assert_eq!(view.received_frames, failure.outcome.received_frames);
        assert_eq!(view.processed_frames, failure.outcome.processed_frames);
        assert_eq!(view.queued_frames, 0);
        assert!(view.partials.is_empty());
    }
}
#[test]
fn slow_subscription_consumer_retains_transcript() {
    let s = session(
        |c| {
            for i in 0..3 {
                c.commit(i, i.to_string(), "word", None, None)?;
            }
            Ok(())
        },
        SessionOptions {
            event_capacity: 1,
            ..Default::default()
        },
    );
    let mut subscription = s.subscribe().unwrap();
    wait_until_running(&mut subscription);
    let outcome = s.finish(Instant::now() + Duration::from_secs(2)).unwrap();
    assert_eq!(outcome.transcript.segments.len(), 3);
    let Update::Reset(view) = subscription.recv().unwrap() else {
        panic!("slow consumer did not self-heal with Reset");
    };
    assert_eq!(view.transcript, outcome.transcript);
}
#[test]
fn deadline_resolves_even_when_native_worker_is_blocked() {
    let (tx, rx) = mpsc::channel();
    let s = session(
        move |_| {
            rx.recv().unwrap();
            Ok(())
        },
        Default::default(),
    );
    let began = Instant::now();
    assert_eq!(
        s.finish(began + Duration::from_millis(40))
            .unwrap_err()
            .error
            .kind,
        ErrorKind::Timeout
    );
    assert!(began.elapsed() < Duration::from_secs(1));
    tx.send(()).unwrap();
}
#[test]
fn worker_permit_is_held_until_the_blocked_worker_really_exits() {
    struct PermitProbe(mpsc::Sender<()>);
    impl Drop for PermitProbe {
        fn drop(&mut self) {
            self.0.send(()).unwrap();
        }
    }

    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (permit_tx, permit_rx) = mpsc::channel();
    let s = spawn(
        Box::new(Blocked {
            ready: ready_tx,
            release: release_rx,
        }),
        AudioSpec { sample_rate: 16000 },
        Default::default(),
        None,
        PermitProbe(permit_tx),
    )
    .unwrap();
    ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();

    let failure = s
        .finish(Instant::now() + Duration::from_millis(40))
        .unwrap_err();
    assert_eq!(failure.error.kind, ErrorKind::Timeout);
    assert_eq!(permit_rx.try_recv(), Err(mpsc::TryRecvError::Empty));

    release_tx.send(()).unwrap();
    permit_rx.recv_timeout(Duration::from_secs(1)).unwrap();
}
#[test]
fn concurrent_finish_uses_the_earliest_deadline() {
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let s = spawn(
        Box::new(Blocked {
            ready: ready_tx,
            release: release_rx,
        }),
        AudioSpec { sample_rate: 16000 },
        Default::default(),
        None,
        (),
    )
    .unwrap();
    ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();

    std::thread::scope(|scope| {
        let long_deadline = Instant::now() + Duration::from_secs(5);
        let session = &s;
        let first = scope.spawn(move || session.finish(long_deadline));
        let wait_limit = Instant::now() + Duration::from_secs(1);
        loop {
            if s.control.0.state.lock().unwrap().lifecycle.deadline == Some(long_deadline) {
                break;
            }
            assert!(Instant::now() < wait_limit, "first finish did not start");
            std::thread::yield_now();
        }

        let short_deadline = Instant::now() + Duration::from_millis(40);
        let began = Instant::now();
        let second_failure = s.finish(short_deadline).unwrap_err();
        assert_eq!(second_failure.error.kind, ErrorKind::Timeout);
        assert!(began.elapsed() < Duration::from_secs(1));
        assert_eq!(
            s.control.0.state.lock().unwrap().lifecycle.deadline,
            Some(short_deadline)
        );

        let first_failure = first.join().unwrap().unwrap_err();
        assert_eq!(first_failure.error.kind, ErrorKind::Timeout);
        assert_eq!(
            first_failure.outcome.transcript,
            second_failure.outcome.transcript
        );
        assert_eq!(
            first_failure.outcome.received_frames,
            second_failure.outcome.received_frames
        );
        assert_eq!(
            first_failure.outcome.processed_frames,
            second_failure.outcome.processed_frames
        );
    });
    release_tx.send(()).unwrap();
}
#[test]
fn finish_without_recorded_deadline_times_out_instead_of_panicking() {
    // `request_finish` normally guarantees `lifecycle.deadline` is Some
    // whenever `outcome` is still None. Clear the deadline by hand
    // mid-finish to pin down that the public `finish` falls back to the
    // caller's deadline (timeout) instead of unwrapping a None.

    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let s = spawn(
        Box::new(FinishBarrier {
            entered: entered_tx,
            release: release_rx,
        }),
        AudioSpec { sample_rate: 16000 },
        Default::default(),
        None,
        (),
    )
    .unwrap();
    let fallback_deadline = Instant::now() + Duration::from_millis(300);
    std::thread::scope(|scope| {
        let finisher = scope.spawn(|| s.finish(fallback_deadline));
        // The driver's finish running implies `request_finish` already
        // recorded the deadline under the lock.
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        s.control.0.state.lock().unwrap().lifecycle.deadline = None;
        s.control.signal();

        let failure = finisher.join().unwrap().unwrap_err();
        assert_eq!(failure.error.kind, ErrorKind::Timeout);
        assert!(s
            .control
            .0
            .state
            .lock()
            .unwrap()
            .lifecycle
            .outcome
            .is_some());
    });
    release_tx.send(()).unwrap();
}
#[test]
fn cancel_and_finish_preserve_whichever_terminal_result_wins() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let s = spawn(
        Box::new(FinishBarrier {
            entered: entered_tx,
            release: release_rx,
        }),
        AudioSpec { sample_rate: 16000 },
        Default::default(),
        None,
        (),
    )
    .unwrap();
    std::thread::scope(|scope| {
        let finisher = scope.spawn(|| s.finish(Instant::now() + Duration::from_secs(2)));
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        s.cancel();
        release_tx.send(()).unwrap();
        let failure = finisher.join().unwrap().unwrap_err();
        assert_eq!(failure.error.kind, ErrorKind::Cancelled);
        assert_eq!(
            s.finish(Instant::now()).unwrap_err().error.kind,
            ErrorKind::Cancelled
        );
    });

    let completed = session(|_| Ok(()), Default::default());
    let outcome = completed
        .finish(Instant::now() + Duration::from_secs(1))
        .unwrap();
    completed.cancel();
    let repeated = completed.finish(Instant::now()).unwrap();
    assert_eq!(repeated.transcript, outcome.transcript);
    assert_eq!(repeated.received_frames, outcome.received_frames);
    assert_eq!(repeated.processed_frames, outcome.processed_frames);
}
#[test]
fn dropping_audio_input_handles_does_not_control_session_lifetime() {
    let s = session(|_| Ok(()), Default::default());

    drop(s.input());
    let external = s.input();
    external
        .try_push(AudioChunk::mono(vec![0.1; 160], 16000).unwrap())
        .unwrap();
    drop(external);

    s.input()
        .try_push(AudioChunk::mono(vec![0.2; 160], 16000).unwrap())
        .unwrap();
    let outcome = s.finish(Instant::now() + Duration::from_secs(1)).unwrap();
    assert_eq!(outcome.received_frames, 320);
    assert_eq!(outcome.processed_frames, 320);
}
#[test]
fn temporary_input_is_safe() {
    let s = session(|_| Ok(()), Default::default());
    s.input()
        .try_push(AudioChunk::mono(vec![0.1; 160], 16000).unwrap())
        .unwrap();
    assert_eq!(
        s.finish(Instant::now() + Duration::from_secs(1))
            .unwrap()
            .processed_frames,
        160
    );
}
#[test]
fn backpressure_returns_original_chunk_and_cancel_wakes_input() {
    let (tx, rx) = mpsc::channel();
    let s = spawn(
        Box::new(Block(rx)),
        AudioSpec { sample_rate: 16000 },
        SessionOptions {
            queue_duration: Duration::from_millis(10),
            max_chunk_duration: Duration::from_millis(10),
            ..Default::default()
        },
        None,
        (),
    )
    .unwrap();
    let input = s.input();
    input
        .try_push(AudioChunk::mono(vec![0.0; 160], 16000).unwrap())
        .unwrap();
    let error = input
        .try_push(AudioChunk::mono(vec![0.5; 80], 16000).unwrap())
        .unwrap_err();
    assert_eq!(error.error.kind, ErrorKind::WouldBlock);
    assert_eq!(error.chunk.samples, vec![0.5; 80]);
    s.cancel();
    assert_eq!(
        input.try_push(error.chunk).unwrap_err().error.kind,
        ErrorKind::InputClosed
    );
    tx.send(()).unwrap();
}
#[test]
fn push_rejects_runtime_input_errors_with_input_stage() {
    let s = session(|_| Ok(()), Default::default());
    let input = s.input();

    let mismatched = input
        .try_push(AudioChunk::mono(vec![0.0; 160], 8000).unwrap())
        .unwrap_err();
    assert_eq!(mismatched.error.kind, ErrorKind::InvalidInput);
    assert_eq!(mismatched.error.stage, "input");

    let oversized = input
        .try_push(AudioChunk::mono(vec![0.0; 16_001], 16000).unwrap())
        .unwrap_err();
    assert_eq!(oversized.error.kind, ErrorKind::InvalidInput);
    assert_eq!(oversized.error.stage, "input");

    let nonfinite = input
        .try_push(AudioChunk::mono(vec![f32::NAN], 16000).unwrap())
        .unwrap_err();
    assert_eq!(nonfinite.error.kind, ErrorKind::InvalidInput);
    assert_eq!(nonfinite.error.stage, "input");

    s.cancel();
}
#[test]
fn validated_push_wait_skips_sample_scan_but_keeps_other_checks() {
    let s = session(|_| Ok(()), Default::default());
    let input = s.input();
    let deadline = Instant::now() + Duration::from_secs(1);

    // Internal feed (`Engine::transcribe`): the engine pre-scan already
    // rejected non-finite samples, so the per-chunk re-scan is skipped.
    input
        .push_wait_validated(AudioChunk::mono(vec![f32::NAN], 16000).unwrap(), deadline)
        .unwrap();

    // Every other check still applies: input spec ...
    let mismatched = input
        .push_wait_validated(AudioChunk::mono(vec![0.1; 160], 8000).unwrap(), deadline)
        .unwrap_err();
    assert_eq!(mismatched.error.kind, ErrorKind::InvalidInput);
    assert_eq!(mismatched.error.message, "input sample rate changed");

    // ... chunk duration ...
    let oversized = input
        .push_wait_validated(
            AudioChunk::mono(vec![0.0; 16_001], 16000).unwrap(),
            deadline,
        )
        .unwrap_err();
    assert_eq!(oversized.error.kind, ErrorKind::InvalidInput);
    assert_eq!(
        oversized.error.message,
        "audio chunk exceeds maximum duration"
    );

    // ... and terminal session state.
    s.cancel();
    let closed = input
        .push_wait_validated(AudioChunk::mono(vec![0.1; 160], 16000).unwrap(), deadline)
        .unwrap_err();
    assert_eq!(closed.error.kind, ErrorKind::InputClosed);
}
#[test]
fn cancel_wakes_a_producer_blocked_in_push_wait() {
    let (release_tx, release_rx) = mpsc::channel();
    let s = spawn(
        Box::new(Block(release_rx)),
        AudioSpec { sample_rate: 16000 },
        SessionOptions {
            queue_duration: Duration::from_millis(10),
            max_chunk_duration: Duration::from_millis(10),
            ..Default::default()
        },
        None,
        (),
    )
    .unwrap();
    let input = s.input();
    input
        .try_push(AudioChunk::mono(vec![0.0; 160], 16000).unwrap())
        .unwrap();

    let (result_tx, result_rx) = mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let result = input.push_wait(
                AudioChunk::mono(vec![0.5; 80], 16000).unwrap(),
                Instant::now() + Duration::from_secs(5),
            );
            result_tx.send(result).unwrap();
        });
        assert!(matches!(
            result_rx.recv_timeout(Duration::from_millis(40)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        s.cancel();
        let error = result_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap_err();
        assert_eq!(error.error.kind, ErrorKind::InputClosed);
        assert_eq!(error.chunk.samples, vec![0.5; 80]);
    });
    release_tx.send(()).unwrap();
}
#[test]
fn push_wait_rejects_an_expired_deadline_even_with_free_capacity() {
    let s = session(|_| Ok(()), Default::default());
    let input = s.input();
    let chunk = AudioChunk::mono(vec![0.25; 160], 16000).unwrap();

    let error = input.push_wait(chunk, Instant::now()).unwrap_err();

    assert_eq!(error.error.kind, ErrorKind::Timeout);
    assert_eq!(error.chunk.samples, vec![0.25; 160]);
    let mut subscription = s.subscribe().unwrap();
    let Update::Reset(view) = subscription.recv().unwrap() else {
        panic!("subscription did not start with Reset");
    };
    assert_eq!(view.received_frames, 0);

    s.cancel();
    let closed = input
        .push_wait(
            AudioChunk::mono(vec![0.5; 80], 16000).unwrap(),
            Instant::now(),
        )
        .unwrap_err();
    assert_eq!(closed.error.kind, ErrorKind::InputClosed);
    assert_eq!(
        s.finish(Instant::now())
            .unwrap_err()
            .outcome
            .received_frames,
        0
    );
}
#[test]
fn start_failure_resolves_with_queued_audio() {
    struct Failed;
    impl Driver for Failed {
        fn start(&mut self, _: &Control) -> Result<(), AsrError> {
            Err(AsrError::backend("fixture start failure"))
        }
        fn push(&mut self, _: &[f32], _: &Control) -> Result<(), AsrError> {
            panic!("must never decode");
        }
        fn finish(&mut self, _: &Control) -> Result<(), AsrError> {
            panic!("must never finish failed driver");
        }
    }
    let s = spawn(
        Box::new(Failed),
        AudioSpec { sample_rate: 16000 },
        Default::default(),
        None,
        (),
    )
    .unwrap();
    let input = s.input();
    let _ = input.try_push(AudioChunk::mono(vec![0.1; 100], 16000).unwrap());
    let _ = input.try_push(AudioChunk::mono(vec![0.1; 100], 16000).unwrap());
    assert_eq!(
        s.finish(Instant::now() + Duration::from_secs(1))
            .unwrap_err()
            .error
            .kind,
        ErrorKind::Backend
    );
}
#[test]
fn dropping_subscription_keeps_session_live() {
    let s = session(
        |c| {
            for i in 0..10 {
                c.commit(i, i.to_string(), "word", None, None)?;
            }
            Ok(())
        },
        SessionOptions {
            event_capacity: 1,
            ..Default::default()
        },
    );
    drop(s.subscribe());
    assert_eq!(
        s.finish(Instant::now() + Duration::from_secs(1))
            .unwrap()
            .transcript
            .segments
            .len(),
        10
    );
}
#[test]
fn transcript_limit_fails_without_losing_confirmed_text() {
    let s = session(
        |c| {
            c.commit(0, "a", "hello", None, None)?;
            c.commit(1, "b", "world!", None, None)
        },
        SessionOptions {
            max_transcript_bytes: 10,
            ..Default::default()
        },
    );
    let failure = s
        .finish(Instant::now() + Duration::from_secs(1))
        .unwrap_err();
    assert_eq!(failure.error.kind, ErrorKind::ResourceLimit);
    assert_eq!(failure.outcome.transcript.text(), "hello");
}
#[test]
fn punctuated_segments_reach_subscription_reset_and_transcript() {
    let punctuator: FinalTextProcessor = Arc::new(|text: &str| format!("{text}。"));
    let s = punctuated_session(
        |c| {
            c.partial("a", "temporary")?;
            c.commit(0, "a", "hello", None, None)?;
            // Whitespace-only text is never punctuated, so it stays an
            // empty segment instead of degrading into punctuation junk.
            c.commit(1, "b", "  ", None, None)
        },
        SessionOptions::default(),
        punctuator,
    );
    let mut subscription = s.subscribe().unwrap();
    wait_until_running(&mut subscription);
    let result = s.finish(Instant::now() + Duration::from_secs(2)).unwrap();
    assert_eq!(result.transcript.segments.len(), 1);
    assert_eq!(result.transcript.text(), "hello。");
    let Update::Reset(view) = subscription.recv().unwrap() else {
        panic!("terminal punctuation result did not produce Reset");
    };
    assert_eq!(view.transcript, result.transcript);
    assert!(view.partials.is_empty());
}
#[test]
fn duplicate_final_is_rejected_before_expensive_text_processing() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let processor_calls = calls.clone();
    let processor: FinalTextProcessor = Arc::new(move |text: &str| {
        processor_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        format!("{text}。")
    });
    let s = punctuated_session(
        |sink| {
            sink.commit(0, "a", "hello", None, None)?;
            sink.commit(0, "a", "hello", None, None)
        },
        SessionOptions::default(),
        processor,
    );
    let result = s.finish(Instant::now() + Duration::from_secs(2)).unwrap();
    assert_eq!(result.transcript.text(), "hello。");
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}
#[test]
fn duplicate_unindexed_final_is_rejected_before_text_processing() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let processor_calls = calls.clone();
    let processor: FinalTextProcessor = Arc::new(move |text: &str| {
        processor_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        format!("{text}。")
    });
    let s = punctuated_session(
        |sink| {
            sink.complete_unindexed("a", "hello")?;
            sink.complete_unindexed("a", "hello")?;
            sink.index_unindexed(0, "a", None, None)
        },
        SessionOptions::default(),
        processor,
    );
    let result = s.finish(Instant::now() + Duration::from_secs(2)).unwrap();
    assert_eq!(result.transcript.text(), "hello。");
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}
#[test]
fn no_final_text_processor_returns_raw_text() {
    let s = session(
        |c| c.commit(0, "a", "hello", None, None),
        SessionOptions::default(),
    );
    let result = s.finish(Instant::now() + Duration::from_secs(2)).unwrap();
    assert_eq!(result.transcript.text(), "hello");
}
#[test]
fn transcript_limit_accounts_for_punctuated_text() {
    let punctuator: FinalTextProcessor = Arc::new(|text: &str| format!("{text}!!!!"));
    let s = punctuated_session(
        |c| {
            c.commit(0, "a", "hello", None, None)?;
            c.commit(1, "b", "world", None, None)
        },
        SessionOptions {
            max_transcript_bytes: 10,
            ..Default::default()
        },
        punctuator,
    );
    let failure = s
        .finish(Instant::now() + Duration::from_secs(1))
        .unwrap_err();
    assert_eq!(failure.error.kind, ErrorKind::ResourceLimit);
    assert_eq!(failure.outcome.transcript.text(), "hello!!!!");
}
#[test]
fn pending_flush_on_failure_keeps_punctuated_text() {
    let punctuator: FinalTextProcessor = Arc::new(|text: &str| format!("{text}。"));
    let s = punctuated_session(
        |c| {
            // Index 1 stays pending (missing predecessor) until the
            // terminal failure flushes it through settle_once_locked.
            c.commit(1, "b", "later", None, None)?;
            Err(AsrError::backend("fixture failure"))
        },
        SessionOptions::default(),
        punctuator,
    );
    let failure = s
        .finish(Instant::now() + Duration::from_secs(2))
        .unwrap_err();
    assert_eq!(failure.error.kind, ErrorKind::Backend);
    assert_eq!(failure.outcome.transcript.text(), "later。");
    assert_eq!(
        s.finish(Instant::now()).unwrap_err().outcome.transcript,
        failure.outcome.transcript
    );
}
#[test]
fn punctuation_inference_does_not_hold_the_session_lock() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    // Receiver is Send but not Sync; the Mutex makes the closure Sync.
    let release_rx = Mutex::new(release_rx);
    let punctuator: FinalTextProcessor = Arc::new(move |text: &str| {
        entered_tx.send(()).unwrap();
        // Hold the inference open while the main thread probes the session.
        release_rx.lock().unwrap().recv().unwrap();
        format!("{text}。")
    });
    let s = punctuated_session(
        |c| c.commit(0, "a", "hello", None, None),
        SessionOptions::default(),
        punctuator,
    );
    let (push_tx, push_rx) = mpsc::channel();
    let input = s.input();
    let (view_tx, view_rx) = mpsc::channel();
    std::thread::scope(|scope| {
        let finisher = scope.spawn(|| s.finish(Instant::now() + Duration::from_secs(5)));
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        // The driver thread is inside the punctuator here. Both probes
        // must complete without waiting for the inference to return; on
        // a regression the release below lets the scope unwind instead of
        // deadlocking.
        scope.spawn(|| {
            let _ = push_tx.send(input.try_push(AudioChunk::mono(vec![0.0; 160], 16000).unwrap()));
        });
        let pushed = match push_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(pushed) => pushed,
            Err(_) => {
                release_tx.send(()).unwrap();
                panic!("try_push blocked on punctuation inference");
            }
        };
        // finish() already closed the input; reaching that rejection
        // proves the push made it through the state lock during inference.
        assert_eq!(pushed.unwrap_err().error.kind, ErrorKind::InputClosed);
        scope.spawn(|| {
            let mut subscription = s.subscribe().unwrap();
            let _ = view_tx.send(subscription.recv().unwrap());
        });
        let update = match view_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(update) => update,
            Err(_) => {
                release_tx.send(()).unwrap();
                panic!("subscription blocked on punctuation inference");
            }
        };
        let Update::Reset(view) = update else {
            panic!("subscription did not start with Reset");
        };
        assert!(view.transcript.segments.is_empty());
        release_tx.send(()).unwrap();
        let result = finisher.join().unwrap().unwrap();
        assert_eq!(result.transcript.text(), "hello。");
    });
}
#[test]
fn sub_frame_durations_are_rejected() {
    let options = SessionOptions {
        queue_duration: Duration::from_nanos(1),
        max_chunk_duration: Duration::from_nanos(1),
        ..Default::default()
    };

    let result = spawn(
        Box::new(Mock {
            finish: Box::new(|_| Ok(())),
        }),
        AudioSpec { sample_rate: 16000 },
        options,
        None,
        (),
    );

    let error = result.err().expect("must reject sub-frame limits");
    assert_eq!(error.kind, ErrorKind::InvalidInput);
}
#[test]
fn push_error_exposes_the_structured_error_as_its_source() {
    let error = PushError {
        error: AsrError::new(ErrorKind::WouldBlock, "input", "full"),
        chunk: AudioChunk::mono(Vec::new(), 16_000).unwrap(),
    };
    let source = std::error::Error::source(&error).unwrap();
    assert!(source.downcast_ref::<AsrError>().is_some());
}
#[test]
fn oversized_queue_is_rejected() {
    let options = SessionOptions {
        queue_duration: Duration::from_secs(1200),
        ..Default::default()
    };

    let result = spawn(
        Box::new(Mock {
            finish: Box::new(|_| Ok(())),
        }),
        AudioSpec { sample_rate: 16000 },
        options,
        None,
        (),
    );

    let error = result.err().expect("oversized queue must fail");
    assert_eq!(error.kind, ErrorKind::InvalidInput);
}
#[test]
fn queue_is_bounded_by_chunk_count() {
    let (tx, rx) = mpsc::channel();
    let s = spawn(
        Box::new(Block(rx)),
        AudioSpec { sample_rate: 16000 },
        SessionOptions {
            queue_duration: Duration::from_secs(1),
            max_chunk_duration: Duration::from_secs(1),
            ..Default::default()
        },
        None,
        (),
    )
    .unwrap();
    let input = s.input();
    for _ in 0..MAX_QUEUE_CHUNKS {
        input
            .try_push(AudioChunk::mono(vec![0.0], 16000).unwrap())
            .unwrap();
    }
    let error = input
        .try_push(AudioChunk::mono(vec![0.0], 16000).unwrap())
        .unwrap_err();
    assert_eq!(error.error.kind, ErrorKind::WouldBlock);
    s.cancel();
    tx.send(()).unwrap();
}
#[test]
fn empty_chunk_is_a_no_op_even_when_the_queue_is_full() {
    let (tx, rx) = mpsc::channel();
    let s = spawn(
        Box::new(Block(rx)),
        AudioSpec { sample_rate: 16000 },
        SessionOptions {
            queue_duration: Duration::from_secs(1),
            max_chunk_duration: Duration::from_secs(1),
            ..Default::default()
        },
        None,
        (),
    )
    .unwrap();
    let input = s.input();
    for _ in 0..MAX_QUEUE_CHUNKS {
        input
            .try_push(AudioChunk::mono(vec![0.0], 16000).unwrap())
            .unwrap();
    }

    let error = input
        .try_push(AudioChunk::mono(vec![0.5], 16000).unwrap())
        .unwrap_err();
    assert_eq!(error.error.kind, ErrorKind::WouldBlock);

    let start = Instant::now();
    input
        .try_push(AudioChunk::mono(Vec::new(), 16000).unwrap())
        .unwrap();
    input
        .push_wait(
            AudioChunk::mono(Vec::new(), 16000).unwrap(),
            start + Duration::from_secs(5),
        )
        .unwrap();
    assert!(start.elapsed() < Duration::from_secs(1));

    s.cancel();
    tx.send(()).unwrap();
}
