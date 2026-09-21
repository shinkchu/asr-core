use super::*;

fn small_store(max_text_bytes: usize) -> ResultStore {
    ResultStore::with_limits(
        max_text_bytes,
        Limits {
            max_id_bytes: 8,
            max_total_id_bytes: 12,
            max_registered_ids: 4,
            max_outstanding_items: 3,
            max_partials: 3,
            max_pending_finals: 3,
            max_unindexed_finals: 3,
        },
    )
}

#[test]
fn final_budget_includes_other_utterance_partials() {
    let mut store = small_store(10);
    store
        .set_partial("a".into(), "123456".into(), true)
        .unwrap();

    let error = store
        .commit(0, "b".into(), "12345".into(), None, None)
        .unwrap_err();

    assert_eq!(error.kind, ErrorKind::ResourceLimit);
    assert_eq!(store.retained_text_bytes, 6);
    assert_eq!(store.partials["a"].1, "123456");
    assert!(!store.registered_ids.contains("b"));
    assert!(store.transcript.segments.is_empty());
}

#[test]
fn replacing_a_partial_only_charges_the_latest_text() {
    let mut store = small_store(6);
    assert_eq!(
        store
            .set_partial("a".into(), "123456".into(), true)
            .unwrap()
            .unwrap()
            .revision,
        1
    );
    assert_eq!(
        store
            .set_partial("a".into(), "12".into(), true)
            .unwrap()
            .unwrap()
            .revision,
        2
    );
    assert_eq!(store.retained_text_bytes, 2);
    assert!(store
        .set_partial("a".into(), "1234567".into(), true)
        .is_err());
    assert_eq!(store.retained_text_bytes, 2);
    assert_eq!(store.partials["a"], (2, "12".into()));
}

#[test]
fn promoting_a_partial_to_final_does_not_double_charge() {
    let mut store = small_store(5);
    store.set_partial("a".into(), "hello".into(), true).unwrap();

    let update = store
        .commit(0, "a".into(), "hello".into(), None, None)
        .unwrap();

    assert!(update.removed_partial);
    assert_eq!(update.segments.len(), 1);
    assert_eq!(store.retained_text_bytes, 5);
    assert!(store.partials.is_empty());
    assert_eq!(store.transcript.text(), "hello");
}

#[test]
fn pending_and_contiguous_finals_share_one_budget() {
    let mut store = small_store(10);
    assert!(store
        .commit(1, "b".into(), "world".into(), None, None)
        .unwrap()
        .segments
        .is_empty());
    let update = store
        .commit(0, "a".into(), "hello".into(), None, None)
        .unwrap();

    assert_eq!(update.segments.len(), 2);
    assert_eq!(store.retained_text_bytes, 10);
    assert_eq!(store.transcript.text(), "hello world");
}

#[test]
fn unindexed_final_is_budgeted_and_requires_an_index_to_finish() {
    let mut store = small_store(5);
    store
        .complete_unindexed("a".into(), "hello".into())
        .unwrap();
    assert_eq!(store.retained_text_bytes, 5);
    assert_eq!(store.completion_error().unwrap().kind, ErrorKind::Protocol);

    let update = store.index_unindexed(0, "a", None, None).unwrap();
    assert_eq!(update.segments.len(), 1);
    assert!(!update.removed_partial);
    assert!(store.completion_error().is_none());
    assert_eq!(store.retained_text_bytes, 5);
    assert_eq!(store.transcript.text(), "hello");
}

#[test]
fn index_unindexed_passes_caller_timestamps_through() {
    let mut store = small_store(20);
    store
        .complete_unindexed("a".into(), "hello".into())
        .unwrap();

    let update = store.index_unindexed(0, "a", Some(1.5), Some(2.5)).unwrap();
    assert_eq!(update.segments.len(), 1);
    assert_eq!(store.transcript.segments[0].start_seconds, Some(1.5));
    assert_eq!(store.transcript.segments[0].end_seconds, Some(2.5));

    // The commit path over an unindexed completion must forward the
    // caller's timestamps instead of hardcoding None.
    store
        .complete_unindexed("b".into(), "world".into())
        .unwrap();
    store
        .commit(1, "b".into(), String::new(), Some(3.0), Some(4.0))
        .unwrap();
    assert_eq!(store.transcript.segments[1].start_seconds, Some(3.0));
    assert_eq!(store.transcript.segments[1].end_seconds, Some(4.0));
}

#[test]
fn id_limits_are_atomic_and_count_each_id_once() {
    let mut store = small_store(100);
    store.register_id("12345678").unwrap();
    store.register_id("12345678").unwrap();
    assert_eq!(store.registered_ids.len(), 1);
    assert_eq!(store.registered_id_bytes, 8);

    assert!(store.register_id("123456789").is_err());
    assert_eq!(store.registered_ids.len(), 1);
    assert!(store.register_id("abcd").is_ok());
    assert_eq!(store.registered_id_bytes, 12);
    assert!(store.register_id("z").is_err());
    assert!(!store.registered_ids.contains("z"));
}

#[test]
fn collection_count_limits_reject_only_the_new_item() {
    let limits = Limits {
        max_id_bytes: 16,
        max_total_id_bytes: 100,
        max_registered_ids: 2,
        max_outstanding_items: 1,
        max_partials: 2,
        max_pending_finals: 1,
        max_unindexed_finals: 1,
    };

    let mut ids = ResultStore::with_limits(100, limits);
    ids.register_id("a").unwrap();
    assert!(ids.register_id("b").is_err());
    assert_eq!(ids.registered_ids.len(), 1);
    ids.complete_unindexed("a".into(), "done".into()).unwrap();
    ids.register_id("b").unwrap();
    assert!(ids.register_id("c").is_err());
    assert_eq!(ids.registered_ids.len(), 2);
    assert!(!ids.registered_ids.contains("c"));

    let mut pending = ResultStore::with_limits(100, limits);
    pending
        .commit(2, "a".into(), "one".into(), None, None)
        .unwrap();
    assert!(pending
        .commit(3, "b".into(), "two".into(), None, None)
        .is_err());
    assert_eq!(pending.pending_by_index.len(), 1);
    assert!(!pending.registered_ids.contains("b"));

    let mut unindexed = ResultStore::with_limits(100, limits);
    unindexed
        .complete_unindexed("a".into(), "one".into())
        .unwrap();
    assert!(unindexed
        .complete_unindexed("b".into(), "two".into())
        .is_err());
    assert_eq!(unindexed.unindexed_final.len(), 1);
    assert!(!unindexed.registered_ids.contains("b"));
}

#[test]
fn delta_failure_leaves_text_revision_bytes_and_id_registration_unchanged() {
    let mut store = small_store(4);
    assert!(store.append_delta("a".into(), "hello", true).is_err());
    assert_eq!(store.retained_text_bytes, 0);
    assert!(store.partials.is_empty());
    assert!(!store.registered_ids.contains("a"));

    let first = store.append_delta("a".into(), "he", true).unwrap().unwrap();
    assert_eq!(first.revision, 1);
    assert!(store.append_delta("a".into(), "llo", true).is_err());
    assert_eq!(store.retained_text_bytes, 2);
    assert_eq!(store.partials["a"], (1, "he".into()));
}

#[test]
fn completed_before_committed_ignores_late_delta_and_duplicate_completion() {
    let mut store = small_store(10);
    store.append_delta("a".into(), "temp", true).unwrap();
    assert!(store
        .complete_unindexed("a".into(), "final".into())
        .unwrap());
    assert!(store
        .append_delta("a".into(), "late", true)
        .unwrap()
        .is_none());
    assert!(!store
        .complete_unindexed("a".into(), "different".into())
        .unwrap());
    assert_eq!(store.retained_text_bytes, 5);

    store.index_unindexed(0, "a", None, None).unwrap();
    assert_eq!(store.transcript.text(), "final");
    assert_eq!(store.retained_text_bytes, 5);
}

#[test]
fn failure_transcript_keeps_indexed_pending_but_not_unindexed_final() {
    let mut store = small_store(20);
    store
        .commit(1, "indexed".into(), "kept".into(), None, None)
        .unwrap();
    store
        .complete_unindexed("loose".into(), "hidden".into())
        .unwrap();

    store.settle();

    assert_eq!(store.transcript.text(), "kept");
    assert_eq!(store.transcript.segments[0].index, 1);
    assert_eq!(store.retained_text_bytes, 4);
}
