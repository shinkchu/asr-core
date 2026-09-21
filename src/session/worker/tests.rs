use super::*;
use crate::ErrorKind;
use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

struct ExitProbe(mpsc::Sender<()>);
impl Drop for ExitProbe {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

/// Parks the worker inside the check-to-lock window so the test can
/// complete a full `cancel()` (settle + signal, with nobody listening)
/// before the worker reaches the state lock.
struct CancelInCheckWindow {
    entered: mpsc::Sender<()>,
    resume: Option<mpsc::Receiver<()>>,
}
impl Driver for CancelInCheckWindow {
    fn start(&mut self, _: &ResultSink) -> Result<(), AsrError> {
        let entered = self.entered.clone();
        let resume = self.resume.take().expect("start runs once");
        AFTER_CHECK.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                entered.send(()).expect("test channel is open");
                resume.recv().expect("test channel is open");
            }));
        });
        Ok(())
    }
    fn push(&mut self, _: &[f32], sink: &ResultSink) -> Result<(), AsrError> {
        sink.check()
    }
    fn finish(&mut self, sink: &ResultSink) -> Result<(), AsrError> {
        sink.check()
    }
}

#[test]
fn cancel_between_check_and_state_lock_still_exits_the_worker() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let (exited_tx, exited_rx) = mpsc::channel();
    let session = crate::coordinator::spawn(
        Box::new(CancelInCheckWindow {
            entered: entered_tx,
            resume: Some(resume_rx),
        }),
        AudioSpec { sample_rate: 16000 },
        crate::SessionOptions::default(),
        None,
        ExitProbe(exited_tx),
    )
    .unwrap();

    // The worker is parked after its unlocked terminal check: a cancel
    // settling now delivers its wake-up to nobody.
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    session.cancel();
    resume_tx.send(()).unwrap();

    // Without the in-lock terminal recheck the worker sleeps here until
    // an unrelated signal arrives, holding its permit the whole time.
    exited_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("cancelled worker must exit without a further signal");
    assert_eq!(
        session.finish(Instant::now()).unwrap_err().error.kind,
        ErrorKind::Cancelled
    );
}
