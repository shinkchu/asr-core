use super::state::SessionState;
use crate::{PartialView, Segment, SessionView, Update};

pub(crate) enum Delivery {
    Update(Update),
    Disconnected,
    Pending,
}

pub(crate) fn current_view(state: &SessionState) -> SessionView {
    SessionView {
        phase: state.lifecycle.phase,
        transcript: state.results.transcript().clone(),
        partials: state.results.partial_views(),
        received_frames: state.input.received,
        processed_frames: state.input.processed,
        queued_frames: state.input.queued,
        error: state
            .lifecycle
            .outcome
            .as_ref()
            .and_then(|result| result.as_ref().err().map(|failure| failure.error.clone())),
    }
}

pub(crate) fn take_next(state: &mut SessionState) -> Delivery {
    if state.observer.needs_reset {
        state.observer.updates.clear();
        state.observer.needs_reset = false;
        return Delivery::Update(Update::Reset(current_view(state)));
    }
    if let Some(update) = state.observer.updates.pop_front() {
        return Delivery::Update(update);
    }
    if state.lifecycle.outcome.is_some() {
        Delivery::Disconnected
    } else {
        Delivery::Pending
    }
}

pub(crate) fn enable(state: &mut SessionState) {
    state.observer.subscription_enabled = true;
    state.observer.updates.clear();
    state.observer.needs_reset = true;
}

pub(crate) fn disable(state: &mut SessionState) {
    state.observer.subscription_enabled = false;
    state.observer.updates.clear();
    state.observer.needs_reset = false;
}

pub(crate) fn terminal(state: &mut SessionState) {
    if state.observer.subscription_enabled {
        require_reset(state);
    }
}

pub(crate) fn partial(state: &mut SessionState, update: &PartialView, capacity: usize) {
    if !state.observer.subscription_enabled || state.observer.needs_reset {
        return;
    }
    state.observer.updates.retain(
        |queued| !matches!(queued, Update::Partial { utterance_id, .. } if utterance_id == &update.utterance_id),
    );
    push(
        state,
        Update::Partial {
            utterance_id: update.utterance_id.clone(),
            revision: update.revision,
            text: update.text.clone(),
        },
        capacity,
    );
}

pub(crate) fn phase(state: &mut SessionState, capacity: usize) {
    if !state.observer.subscription_enabled || state.observer.needs_reset {
        return;
    }
    state
        .observer
        .updates
        .retain(|queued| !matches!(queued, Update::Phase(_)));
    push(state, Update::Phase(state.lifecycle.phase), capacity);
}

pub(crate) fn final_segments(
    state: &mut SessionState,
    completed_id: &str,
    removed_partial: bool,
    segments: &[Segment],
    capacity: usize,
) {
    if !state.observer.subscription_enabled {
        return;
    }
    if state.observer.needs_reset {
        return;
    }

    remove_queued_partial(state, completed_id);
    if removed_partial && segments.is_empty() {
        require_reset(state);
        return;
    }

    for segment in segments {
        remove_queued_partial(state, &segment.id);
        if !push(state, Update::Segment(segment.clone()), capacity) {
            return;
        }
    }
}

#[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
pub(crate) fn partial_removed_without_segment(state: &mut SessionState, id: &str) {
    if !state.observer.subscription_enabled || state.observer.needs_reset {
        return;
    }
    remove_queued_partial(state, id);
    require_reset(state);
}

fn remove_queued_partial(state: &mut SessionState, id: &str) {
    state.observer.updates.retain(
        |queued| !matches!(queued, Update::Partial { utterance_id, .. } if utterance_id == id),
    );
}

fn push(state: &mut SessionState, update: Update, capacity: usize) -> bool {
    if state.observer.updates.len() >= capacity {
        require_reset(state);
        false
    } else {
        state.observer.updates.push_back(update);
        true
    }
}

fn require_reset(state: &mut SessionState) {
    state.observer.updates.clear();
    state.observer.needs_reset = true;
}

#[cfg(test)]
mod tests;
