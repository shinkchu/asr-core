use super::*;
use crate::SessionPhase;

#[test]
fn partial_and_phase_updates_keep_only_the_latest_value() {
    let mut state = SessionState::new(1024);
    enable(&mut state);
    assert!(matches!(
        take_next(&mut state),
        Delivery::Update(Update::Reset(_))
    ));

    partial(
        &mut state,
        &PartialView {
            utterance_id: "item".into(),
            revision: 1,
            text: "old".into(),
        },
        4,
    );
    partial(
        &mut state,
        &PartialView {
            utterance_id: "item".into(),
            revision: 2,
            text: "new".into(),
        },
        4,
    );
    state.lifecycle.phase = SessionPhase::Running;
    phase(&mut state, 4);
    state.lifecycle.phase = SessionPhase::Finishing;
    phase(&mut state, 4);

    assert!(matches!(
        take_next(&mut state),
        Delivery::Update(Update::Partial {
            revision: 2,
            ref text,
            ..
        }) if text == "new"
    ));
    assert!(matches!(
        take_next(&mut state),
        Delivery::Update(Update::Phase(SessionPhase::Finishing))
    ));
    assert!(matches!(take_next(&mut state), Delivery::Pending));
}
