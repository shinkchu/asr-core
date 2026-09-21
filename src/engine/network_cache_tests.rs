use super::*;

#[test]
fn failed_builds_are_not_cached_and_later_attempts_retry() {
    let slot: Mutex<Option<Arc<u8>>> = Mutex::new(None);

    let error = init_on_success(&slot, || Err(AsrError::backend("transient failure"))).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Backend);
    assert!(slot.lock().unwrap().is_none(), "failure must not be cached");

    let built = init_on_success(&slot, || Ok(7_u8)).unwrap();
    assert_eq!(*built, 7);

    // 成功后命中缓存,不再调用构建函数。
    let cached = init_on_success(&slot, || panic!("cached value must not rebuild")).unwrap();
    assert_eq!(*cached, 7);
}
