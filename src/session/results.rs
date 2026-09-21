use crate::{AsrError, ErrorKind, PartialView, Segment, Transcript};
use std::collections::{BTreeMap, HashMap, HashSet};

const MAX_ID_BYTES: usize = 1024;
const MAX_TOTAL_ID_BYTES: usize = 1024 * 1024;
const MAX_REGISTERED_IDS: usize = 100_000;
const MAX_OUTSTANDING_ITEMS: usize = 128;
const MAX_PARTIALS: usize = 128;
const MAX_PENDING_FINALS: usize = 128;
const MAX_UNINDEXED_FINALS: usize = 128;

#[derive(Clone, Copy)]
struct Limits {
    max_id_bytes: usize,
    max_total_id_bytes: usize,
    max_registered_ids: usize,
    max_outstanding_items: usize,
    max_partials: usize,
    max_pending_finals: usize,
    #[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
    max_unindexed_finals: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_id_bytes: MAX_ID_BYTES,
            max_total_id_bytes: MAX_TOTAL_ID_BYTES,
            max_registered_ids: MAX_REGISTERED_IDS,
            max_outstanding_items: MAX_OUTSTANDING_ITEMS,
            max_partials: MAX_PARTIALS,
            max_pending_finals: MAX_PENDING_FINALS,
            max_unindexed_finals: MAX_UNINDEXED_FINALS,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Completion {
    Indexed(u64),
    #[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
    Unindexed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CommitPreflight {
    Process,
    Duplicate,
    IndexUnindexed,
}

#[derive(Debug, Default)]
pub(crate) struct CommitUpdate {
    pub(crate) segments: Vec<Segment>,
    pub(crate) removed_partial: bool,
}

pub(crate) struct ResultStore {
    transcript: Transcript,
    pending_by_index: BTreeMap<u64, Segment>,
    completed_ids: HashMap<String, Completion>,
    partials: BTreeMap<String, (u64, String)>,
    unindexed_final: HashMap<String, String>,
    registered_ids: HashSet<String>,
    outstanding_items: HashSet<String>,
    next_segment: u64,
    retained_text_bytes: usize,
    registered_id_bytes: usize,
    max_text_bytes: usize,
    limits: Limits,
}

impl ResultStore {
    pub(crate) fn new(max_text_bytes: usize) -> Self {
        Self::with_limits(max_text_bytes, Limits::default())
    }

    fn with_limits(max_text_bytes: usize, limits: Limits) -> Self {
        Self {
            transcript: Transcript::default(),
            pending_by_index: BTreeMap::new(),
            completed_ids: HashMap::new(),
            partials: BTreeMap::new(),
            unindexed_final: HashMap::new(),
            registered_ids: HashSet::new(),
            outstanding_items: HashSet::new(),
            next_segment: 0,
            retained_text_bytes: 0,
            registered_id_bytes: 0,
            max_text_bytes,
            limits,
        }
    }

    pub(crate) fn transcript(&self) -> &Transcript {
        &self.transcript
    }

    pub(crate) fn preflight_commit(
        &self,
        index: u64,
        id: &str,
    ) -> Result<CommitPreflight, AsrError> {
        if let Some(completion) = self.completed_ids.get(id).copied() {
            return match completion {
                Completion::Indexed(previous) if previous == index => {
                    Ok(CommitPreflight::Duplicate)
                }
                Completion::Indexed(_) => Err(protocol(
                    "utterance ID was reused for another segment index",
                )),
                Completion::Unindexed => Ok(CommitPreflight::IndexUnindexed),
            };
        }
        self.validate_index(index)?;
        if index > self.next_segment
            && self.pending_by_index.len() >= self.limits.max_pending_finals
        {
            return Err(resource("too many pending final segments"));
        }
        Ok(CommitPreflight::Process)
    }

    #[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
    pub(crate) fn preflight_unindexed(&self, id: &str) -> Result<bool, AsrError> {
        if self.completed_ids.contains_key(id) {
            return Ok(false);
        }
        if self.unindexed_final.len() >= self.limits.max_unindexed_finals {
            return Err(resource("too many unindexed final transcripts"));
        }
        self.validate_new_id(id, false)?;
        Ok(true)
    }

    pub(crate) fn partial_views(&self) -> Vec<PartialView> {
        self.partials
            .iter()
            .map(|(utterance_id, (revision, text))| PartialView {
                utterance_id: utterance_id.clone(),
                revision: *revision,
                text: text.clone(),
            })
            .collect()
    }

    #[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
    pub(crate) fn has_partial(&self, id: &str) -> bool {
        self.partials.contains_key(id)
    }

    #[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
    pub(crate) fn register_id(&mut self, id: &str) -> Result<(), AsrError> {
        let is_new = self.validate_new_id(id, true)?;
        if is_new {
            self.insert_new_id(id, true);
        }
        Ok(())
    }

    #[cfg_attr(
        not(any(
            feature = "backend-sherpa",
            feature = "backend-dashscope",
            feature = "backend-openai-http"
        )),
        allow(dead_code)
    )]
    pub(crate) fn set_partial(
        &mut self,
        id: String,
        text: String,
        emit_update: bool,
    ) -> Result<Option<PartialView>, AsrError> {
        if self.completed_ids.contains_key(&id) {
            return Ok(None);
        }
        if self
            .partials
            .get(&id)
            .is_some_and(|(_, previous)| previous == &text)
        {
            return Ok(None);
        }

        let existing = self.partials.get(&id);
        if existing.is_none() && self.partials.len() >= self.limits.max_partials {
            return Err(resource("too many outstanding partial utterances"));
        }
        let previous_bytes = existing.map_or(0, |(_, previous)| previous.len());
        let retained_text_bytes = self.prospective_text_bytes(previous_bytes, text.len())?;
        let is_new_id = self.validate_new_id(&id, true)?;
        let revision = existing
            .map_or(Some(1), |(revision, _)| revision.checked_add(1))
            .ok_or_else(|| resource("partial revision limit exceeded"))?;

        if is_new_id {
            self.insert_new_id(&id, true);
        }
        self.retained_text_bytes = retained_text_bytes;
        if emit_update {
            self.partials.insert(id.clone(), (revision, text.clone()));
            Ok(Some(PartialView {
                utterance_id: id,
                revision,
                text,
            }))
        } else {
            self.partials.insert(id, (revision, text));
            Ok(None)
        }
    }

    #[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
    pub(crate) fn append_delta(
        &mut self,
        id: String,
        delta: &str,
        emit_update: bool,
    ) -> Result<Option<PartialView>, AsrError> {
        if self.completed_ids.contains_key(&id) {
            return Ok(None);
        }

        let existing = self.partials.get(&id);
        if existing.is_none() && self.partials.len() >= self.limits.max_partials {
            return Err(resource("too many outstanding partial utterances"));
        }
        let previous_bytes = existing.map_or(0, |(_, text)| text.len());
        let new_bytes = previous_bytes
            .checked_add(delta.len())
            .ok_or_else(|| resource("transcript byte limit exceeded"))?;
        if new_bytes == previous_bytes {
            return Ok(None);
        }
        let retained_text_bytes = self.prospective_text_bytes(previous_bytes, new_bytes)?;
        let is_new_id = self.validate_new_id(&id, true)?;
        let revision = existing
            .map_or(Some(1), |(revision, _)| revision.checked_add(1))
            .ok_or_else(|| resource("partial revision limit exceeded"))?;

        let mut text = existing.map_or_else(String::new, |(_, text)| text.clone());
        text.try_reserve(delta.len())
            .map_err(|_| resource("failed to reserve partial transcript"))?;
        text.push_str(delta);

        if is_new_id {
            self.insert_new_id(&id, true);
        }
        self.retained_text_bytes = retained_text_bytes;
        if emit_update {
            self.partials.insert(id.clone(), (revision, text.clone()));
            Ok(Some(PartialView {
                utterance_id: id,
                revision,
                text,
            }))
        } else {
            self.partials.insert(id, (revision, text));
            Ok(None)
        }
    }

    pub(crate) fn commit(
        &mut self,
        index: u64,
        id: String,
        text: String,
        start_seconds: Option<f64>,
        end_seconds: Option<f64>,
    ) -> Result<CommitUpdate, AsrError> {
        if let Some(completion) = self.completed_ids.get(&id).copied() {
            return match completion {
                Completion::Indexed(previous) if previous == index => Ok(CommitUpdate::default()),
                Completion::Indexed(_) => Err(protocol(
                    "utterance ID was reused for another segment index",
                )),
                Completion::Unindexed => {
                    self.index_unindexed(index, &id, start_seconds, end_seconds)
                }
            };
        }
        self.validate_index(index)?;
        if index > self.next_segment
            && self.pending_by_index.len() >= self.limits.max_pending_finals
        {
            return Err(resource("too many pending final segments"));
        }

        let previous_bytes = self.partials.get(&id).map_or(0, |(_, text)| text.len());
        let retained_text_bytes = self.prospective_text_bytes(previous_bytes, text.len())?;
        let is_new_id = self.validate_new_id(&id, false)?;

        if is_new_id {
            self.insert_new_id(&id, false);
        }
        self.outstanding_items.remove(&id);
        let removed_partial = self.partials.remove(&id).is_some();
        self.retained_text_bytes = retained_text_bytes;
        self.completed_ids
            .insert(id.clone(), Completion::Indexed(index));
        self.pending_by_index.insert(
            index,
            Segment {
                id,
                index,
                text,
                start_seconds,
                end_seconds,
            },
        );

        Ok(CommitUpdate {
            segments: self.drain_contiguous(),
            removed_partial,
        })
    }

    #[cfg_attr(not(feature = "backend-openai-realtime"), allow(dead_code))]
    pub(crate) fn complete_unindexed(
        &mut self,
        id: String,
        text: String,
    ) -> Result<bool, AsrError> {
        if self.completed_ids.contains_key(&id) {
            return Ok(false);
        }
        if self.unindexed_final.len() >= self.limits.max_unindexed_finals {
            return Err(resource("too many unindexed final transcripts"));
        }

        let previous_bytes = self.partials.get(&id).map_or(0, |(_, text)| text.len());
        let retained_text_bytes = self.prospective_text_bytes(previous_bytes, text.len())?;
        let is_new_id = self.validate_new_id(&id, false)?;

        if is_new_id {
            self.insert_new_id(&id, false);
        }
        self.outstanding_items.remove(&id);
        self.partials.remove(&id);
        self.retained_text_bytes = retained_text_bytes;
        self.completed_ids.insert(id.clone(), Completion::Unindexed);
        self.unindexed_final.insert(id, text);
        Ok(true)
    }

    pub(crate) fn index_unindexed(
        &mut self,
        index: u64,
        id: &str,
        start_seconds: Option<f64>,
        end_seconds: Option<f64>,
    ) -> Result<CommitUpdate, AsrError> {
        match self.completed_ids.get(id).copied() {
            Some(Completion::Indexed(previous)) if previous == index => {
                return Ok(CommitUpdate::default())
            }
            Some(Completion::Indexed(_)) => {
                return Err(protocol(
                    "utterance ID was reused for another segment index",
                ))
            }
            Some(Completion::Unindexed) => {}
            None => return Err(protocol("unindexed final transcript is unknown")),
        }
        self.validate_index(index)?;
        if index > self.next_segment
            && self.pending_by_index.len() >= self.limits.max_pending_finals
        {
            return Err(resource("too many pending final segments"));
        }
        let text = self
            .unindexed_final
            .remove(id)
            .ok_or_else(|| protocol("unindexed final transcript is missing"))?;
        self.completed_ids
            .insert(id.to_owned(), Completion::Indexed(index));
        let removed_partial = self.partials.remove(id).is_some();
        self.pending_by_index.insert(
            index,
            Segment {
                id: id.to_owned(),
                index,
                text,
                start_seconds,
                end_seconds,
            },
        );
        Ok(CommitUpdate {
            segments: self.drain_contiguous(),
            removed_partial,
        })
    }

    pub(crate) fn completion_error(&self) -> Option<AsrError> {
        if !self.unindexed_final.is_empty() {
            Some(protocol(
                "final transcript has no committed ordering information",
            ))
        } else if !self.pending_by_index.is_empty() {
            Some(protocol("missing earlier segment at end of session"))
        } else {
            None
        }
    }

    pub(crate) fn settle(&mut self) {
        for (_, segment) in std::mem::take(&mut self.pending_by_index) {
            if segment.text.trim().is_empty() {
                self.retained_text_bytes =
                    self.retained_text_bytes.saturating_sub(segment.text.len());
            } else {
                self.transcript.segments.push(segment);
            }
        }

        let partial_bytes = self
            .partials
            .values()
            .map(|(_, text)| text.len())
            .fold(0usize, usize::saturating_add);
        self.retained_text_bytes = self.retained_text_bytes.saturating_sub(partial_bytes);
        self.partials.clear();

        let unindexed_bytes = self
            .unindexed_final
            .values()
            .map(String::len)
            .fold(0usize, usize::saturating_add);
        self.retained_text_bytes = self.retained_text_bytes.saturating_sub(unindexed_bytes);
        self.unindexed_final.clear();
        self.outstanding_items.clear();
    }

    fn validate_index(&self, index: u64) -> Result<(), AsrError> {
        if index < self.next_segment || self.pending_by_index.contains_key(&index) {
            Err(protocol("segment index was reused"))
        } else {
            Ok(())
        }
    }

    fn drain_contiguous(&mut self) -> Vec<Segment> {
        let mut committed = Vec::new();
        while let Some(segment) = self.pending_by_index.remove(&self.next_segment) {
            self.next_segment = self.next_segment.saturating_add(1);
            if segment.text.trim().is_empty() {
                self.retained_text_bytes =
                    self.retained_text_bytes.saturating_sub(segment.text.len());
            } else {
                self.transcript.segments.push(segment.clone());
                committed.push(segment);
            }
        }
        committed
    }

    fn prospective_text_bytes(&self, removed: usize, added: usize) -> Result<usize, AsrError> {
        let retained = self
            .retained_text_bytes
            .checked_sub(removed)
            .and_then(|bytes| bytes.checked_add(added))
            .ok_or_else(|| resource("transcript byte limit exceeded"))?;
        if retained > self.max_text_bytes {
            Err(resource("transcript byte limit exceeded"))
        } else {
            Ok(retained)
        }
    }

    fn validate_new_id(&self, id: &str, outstanding: bool) -> Result<bool, AsrError> {
        if self.registered_ids.contains(id) {
            return Ok(false);
        }
        if id.len() > self.limits.max_id_bytes {
            return Err(resource("protocol ID exceeds the per-item byte limit"));
        }
        if self.registered_ids.len() >= self.limits.max_registered_ids {
            return Err(resource("too many protocol IDs"));
        }
        let total = self
            .registered_id_bytes
            .checked_add(id.len())
            .ok_or_else(|| resource("protocol ID byte limit exceeded"))?;
        if total > self.limits.max_total_id_bytes {
            return Err(resource("protocol ID byte limit exceeded"));
        }
        if outstanding && self.outstanding_items.len() >= self.limits.max_outstanding_items {
            return Err(resource("too many unfinished protocol items"));
        }
        Ok(true)
    }

    fn insert_new_id(&mut self, id: &str, outstanding: bool) {
        self.registered_id_bytes += id.len();
        self.registered_ids.insert(id.to_owned());
        if outstanding {
            self.outstanding_items.insert(id.to_owned());
        }
    }
}

fn resource(message: &'static str) -> AsrError {
    AsrError::new(ErrorKind::ResourceLimit, "result", message)
}

fn protocol(message: &'static str) -> AsrError {
    AsrError::new(ErrorKind::Protocol, "result", message)
}

#[cfg(test)]
mod tests;
