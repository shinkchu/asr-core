use crate::AsrError;
use std::time::{Duration, Instant};

/// Upper bound for network timeouts so `Instant` deadline arithmetic can never
/// overflow, regardless of how the platform clock represents instants.
pub(crate) const MAX_NETWORK_TIMEOUT: Duration = Duration::from_secs(24 * 60 * 60);

pub(crate) fn after(timeout: Duration, name: &str) -> Result<Instant, AsrError> {
    Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| AsrError::invalid(format!("{name} timeout is too large")))
}

/// The error every deadline-bounded network operation reports when its
/// window expires.
pub(crate) fn exceeded(stage: &str) -> AsrError {
    AsrError::new(
        crate::ErrorKind::Timeout,
        stage,
        "operation deadline exceeded",
    )
}
