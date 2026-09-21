use super::{
    driver::{Driver, FinalTextProcessor, ResultSink},
    observe,
};
use crate::{audio::resample::Resampler, coordinator::Control, AsrError, AudioSpec, SessionPhase};

#[cfg(test)]
thread_local! {
    /// One-shot test barrier installed between the worker's unlocked
    /// terminal check and its state re-lock, so a test can land a cancel
    /// inside that window deterministically instead of racing a scheduler.
    static AFTER_CHECK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

pub(crate) fn run(
    mut driver: Box<dyn Driver>,
    target: AudioSpec,
    control: &Control,
    final_text_processor: Option<FinalTextProcessor>,
) -> Result<(), AsrError> {
    let options = &control.0.options;
    let mut resampler = Resampler::try_new(options.input.sample_rate, target.sample_rate)?;
    let sink = ResultSink::new(control.clone(), final_text_processor);
    driver.start(&sink)?;
    {
        let mut state = control.0.state.lock().unwrap();
        if state.lifecycle.phase == SessionPhase::Starting {
            state.lifecycle.phase = SessionPhase::Running;
            observe::phase(&mut state, control.0.options.event_capacity);
        }
    }
    control.signal();
    loop {
        control.check()?;
        #[cfg(test)]
        AFTER_CHECK.with(|slot| {
            if let Some(hook) = slot.borrow_mut().take() {
                hook();
            }
        });
        let mut state = control.0.state.lock().unwrap();
        // A terminal settle (cancel, fail_input, deadline) can land between
        // the unlocked `check()` above and this lock: it drains the queue and
        // leaves `finishing` false, so the re-locked state no longer implies
        // progress and the branch below would wait for a wake-up that was
        // already delivered. Drivers without a poll interval never retry on
        // their own, so re-derive the exit from the locked state instead of
        // trusting the pre-lock check.
        if state.lifecycle.outcome.is_some() {
            drop(state);
            control.check()?;
        } else if let Some(chunk) = state.input.queue.pop_front() {
            state.input.queued -= chunk.samples.len();
            drop(state);
            control.signal();
            let mut output = Vec::new();
            resampler.try_push(&chunk.samples, &mut output)?;
            if !output.is_empty() {
                driver.push(&output, &sink)?;
            }
            let mut state = control.0.state.lock().unwrap();
            if state.lifecycle.outcome.is_none() {
                state.input.processed += chunk.samples.len() as u64;
            }
            drop(state);
            if driver.poll_interval().is_some() {
                driver.poll(&sink)?;
            }
        } else if state.lifecycle.finishing {
            drop(state);
            break;
        } else if let Some(interval) = driver.poll_interval() {
            let _ = control.0.changed.wait_timeout(state, interval).unwrap();
            driver.poll(&sink)?;
        } else {
            drop(control.0.changed.wait(state).unwrap());
        }
    }
    let mut tail = Vec::new();
    resampler.finish(&mut tail)?;
    if !tail.is_empty() {
        driver.push(&tail, &sink)?;
    }
    driver.finish(&sink)?;
    control.check()
}

#[cfg(test)]
mod tests;
