use crate::data_gen::{map_role, TrainingItem};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use memmap2::Mmap;
use rust_functiongemma_core::chat_template::render_chat_template;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use tokenizers::Tokenizer;

/// Magic bytes written at the start of every versioned token cache file.
const CACHE_MAGIC: &[u8; 4] = b"PCAI";

/// Increment this whenever the binary layout of the cache changes.
const CACHE_VERSION: u32 = 1;

/// Total header size in bytes: 4 magic + 4 version + 16 hash + 8 reserved.
const CACHE_HEADER_SIZE: usize = 32;

/// Compute a 16-byte fingerprint of the tokenizer file on disk.
///
/// The fingerprint is derived from a 64-bit hash of the file content, stored
/// in the lower 8 bytes; the upper 8 bytes are zeroed.  Using
/// `DefaultHasher` is intentional: we only need fast, deterministic
/// identity checking, not cryptographic security.
fn compute_tokenizer_hash(tokenizer_path: &Path) -> Result<[u8; 16]> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let content = std::fs::read(tokenizer_path)
        .with_context(|| format!("Failed to read tokenizer at {}", tokenizer_path.display()))?;
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    let h = hasher.finish();
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&h.to_le_bytes());
    Ok(out)
}

/// Write the 32-byte cache header to `writer`.
fn write_cache_header(writer: &mut impl Write, tokenizer_hash: &[u8; 16]) -> Result<()> {
    writer.write_all(CACHE_MAGIC)?;
    writer.write_all(&CACHE_VERSION.to_le_bytes())?;
    writer.write_all(tokenizer_hash)?;
    writer.write_all(&[0u8; 8])?; // reserved
    Ok(())
}

/// Validate the 32-byte header at the start of `data`.
///
/// Checks magic bytes, version number, and tokenizer hash in order.
/// Returns a descriptive error for each failure mode so the caller can
/// surface an actionable message to the user.
fn validate_cache_header(data: &[u8], expected_hash: &[u8; 16]) -> Result<()> {
    if data.len() < CACHE_HEADER_SIZE {
        anyhow::bail!("Token cache file too small to contain header");
    }
    if &data[0..4] != CACHE_MAGIC {
        anyhow::bail!("Token cache has invalid magic bytes (not a PCAI cache file)");
    }
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if version != CACHE_VERSION {
        anyhow::bail!(
            "Token cache version mismatch: file has v{}, expected v{}. Rebuild the cache.",
            version,
            CACHE_VERSION
        );
    }
    let stored_hash = &data[8..24];
    if stored_hash != expected_hash {
        anyhow::bail!(
            "Token cache was built with a different tokenizer. \
             Rebuild the cache with the current tokenizer."
        );
    }
    Ok(())
}

pub struct Dataset {
    pub items: Vec<TrainingItem>,
    pub token_cache: Option<TokenCache>,
    pub chat_template: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TokenCacheEntry {
    pub offset: u64,
    pub len: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenCacheMeta {
    pub source_jsonl: String,
    pub tokenizer_path: String,
    pub item_count: usize,
}

pub struct TokenCache {
    mmap: Mmap,
    entries: Vec<TokenCacheEntry>,
    mask_mmap: Option<Mmap>,
    /// `true` when the binary files start with a `CACHE_HEADER_SIZE`-byte
    /// versioned header.  Legacy caches written before this change will have
    /// `has_header = false` and will continue to load correctly.
    has_header: bool,
}

impl Dataset {
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut items = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let item: TrainingItem = serde_json::from_str(&line)?;
            items.push(item);
        }
        Ok(Dataset {
            items,
            token_cache: None,
            chat_template: None,
        })
    }

    pub fn len(&self) -> usize {
        if let Some(cache) = &self.token_cache {
            cache.entries.len()
        } else {
            self.items.len()
        }
    }

    pub fn load_cached(cache_dir: &Path) -> Result<Self> {
        let cache = TokenCache::load(cache_dir)?;
        Ok(Dataset {
            items: Vec::new(),
            token_cache: Some(cache),
            chat_template: None,
        })
    }

    pub fn with_chat_template(mut self, template: Option<String>) -> Self {
        self.chat_template = template;
        self
    }

    pub fn build_token_cache(
        input_jsonl: &Path,
        tokenizer: &Tokenizer,
        tokenizer_path: &Path,
        output_dir: &Path,
        chat_template: Option<&str>,
    ) -> Result<TokenCacheMeta> {
        std::fs::create_dir_all(output_dir)?;
        let bin_path = output_dir.join("tokens.bin");
        let mask_path = output_dir.join("tokens.mask.bin");
        let idx_path = output_dir.join("tokens.idx.json");
        let meta_path = output_dir.join("tokens.meta.json");

        let file = File::open(input_jsonl)?;
        let reader = BufReader::new(file);
        let mut writer = BufWriter::new(File::create(&bin_path)?);
        let mut mask_writer = BufWriter::new(File::create(&mask_path)?);
        let mut entries = Vec::new();
        let mut offset: u64 = 0;

        // Write versioned headers before any token data.  `offset` counts
        // tokens (not bytes), so the stored entry offsets remain relative to
        // the first token regardless of header size.  `get_ids`/`get_mask`
        // compensate by adding CACHE_HEADER_SIZE to the byte address.
        let tok_hash = compute_tokenizer_hash(tokenizer_path)?;
        write_cache_header(&mut writer, &tok_hash)?;
        write_cache_header(&mut mask_writer, &tok_hash)?;

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let item: TrainingItem = serde_json::from_str(&line)?;
            let (ids, mask) = encode_item_with_mask(&item, tokenizer, chat_template)?;
            let len = ids.len() as u32;
            for id in &ids {
                writer.write_all(&id.to_le_bytes())?;
            }
            mask_writer.write_all(&mask)?;
            entries.push(TokenCacheEntry { offset, len });
            offset += len as u64;
        }
        writer.flush()?;
        mask_writer.flush()?;

        std::fs::write(&idx_path, serde_json::to_string_pretty(&entries)?)?;

        let meta = TokenCacheMeta {
            source_jsonl: input_jsonl.display().to_string(),
            tokenizer_path: tokenizer_path.display().to_string(),
            item_count: entries.len(),
        };
        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;
        Ok(meta)
    }

    pub fn get_batch(
        &self,
        start_idx: usize,
        batch_size: usize,
        tokenizer: Option<&Tokenizer>,
        device: &Device,
        pack_sequences: bool,
        max_seq_len: Option<usize>,
        eos_token_id: u32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dataset_len = if let Some(cache) = &self.token_cache {
            cache.entries.len()
        } else {
            self.items.len()
        };
        let end_idx = (start_idx + batch_size).min(dataset_len);
        let indices: Vec<usize> = (start_idx..end_idx).collect();
        self.get_batch_by_indices(
            &indices,
            tokenizer,
            device,
            pack_sequences,
            max_seq_len,
            eos_token_id,
        )
    }

    pub fn get_batch_by_indices(
        &self,
        indices: &[usize],
        tokenizer: Option<&Tokenizer>,
        device: &Device,
        pack_sequences: bool,
        max_seq_len: Option<usize>,
        eos_token_id: u32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let mut input_ids_batch = Vec::new();
        let mut mask_batch = Vec::new();
        let mut max_len = 0;

        if let Some(cache) = &self.token_cache {
            for &i in indices {
                let ids = cache.get_ids(i)?;
                let mask = cache.get_mask(i)?;
                max_len = max_len.max(ids.len());
                input_ids_batch.push(ids);
                mask_batch.push(mask);
            }
        } else {
            let tokenizer = tokenizer.context("Tokenizer required for non-cached dataset")?;
            for &i in indices {
                let item = &self.items[i];
                let (ids, mask) =
                    encode_item_with_mask(item, tokenizer, self.chat_template.as_deref())?;

                max_len = max_len.max(ids.len());
                input_ids_batch.push(ids);
                mask_batch.push(mask);
            }
        }

        let mut sequences = if pack_sequences {
            let (seqs, masks) = pack_token_sequences_with_mask(
                input_ids_batch,
                mask_batch,
                max_seq_len.unwrap_or(max_len),
                eos_token_id,
            );
            mask_batch = masks;
            seqs
        } else {
            input_ids_batch
        };

        if let Some(cap) = max_seq_len {
            for (seq, mask) in sequences.iter_mut().zip(mask_batch.iter_mut()) {
                truncate_sequence_with_mask(seq, mask, cap);
            }
        }

        if sequences.is_empty() {
            return Err(anyhow::anyhow!("Empty batch"));
        }

        max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        if let Some(cap) = max_seq_len {
            max_len = max_len.min(cap);
        }

        // Simple alignment/padding
        let mut padded_inputs = Vec::new();
        let mut padded_targets = Vec::new();
        let mut padded_masks = Vec::new();

        for (ids, mask) in sequences.drain(..).zip(mask_batch.drain(..)) {
            let mut input = ids.clone();
            let mut target = ids.clone();
            let mut loss_mask = mask.clone();

            // For causal LM, we shift targets
            input.pop(); // Remove last for input
            target.remove(0); // Remove first for target
            loss_mask.remove(0); // Align mask to targets

            while input.len() < max_len.saturating_sub(1) {
                input.push(0); // Pad with 0
                target.push(0);
                loss_mask.push(0);
            }
            padded_inputs.push(Tensor::new(input, device)?);
            padded_targets.push(Tensor::new(target, device)?);
            padded_masks.push(Tensor::new(loss_mask, device)?);
        }

        let inputs = Tensor::stack(&padded_inputs, 0)?;
        let targets = Tensor::stack(&padded_targets, 0)?;
        let masks = Tensor::stack(&padded_masks, 0)?;
        Ok((inputs, targets, masks))
    }
}

fn format_message_as_text(msg: &crate::data_gen::Message) -> Result<String> {
    let mut text = String::new();
    let role = map_role(&msg.role);
    text.push_str("<start_of_turn>");
    text.push_str(role);
    text.push('\n');
    if let Some(content) = &msg.content {
        text.push_str(content);
    }
    text.push_str("<end_of_turn>\n");
    Ok(text)
}

fn encode_item_with_mask(
    item: &TrainingItem,
    tokenizer: &Tokenizer,
    chat_template: Option<&str>,
) -> Result<(Vec<u32>, Vec<u8>)> {
    if let Some(template) = chat_template {
        let messages_value = serde_json::to_value(&item.messages)?;
        let full_text = render_chat_template(template, &messages_value, &item.tools, false)?;
        let full_encoding = tokenizer
            .encode(full_text.as_str(), true)
            .map_err(anyhow::Error::msg)?;
        let full_ids = full_encoding.get_ids().to_vec();
        let mut mask = vec![0u8; full_ids.len()];

        let mut search_from = 0usize;
        for msg in &item.messages {
            let trainable = matches!(msg.role.as_str(), "assistant" | "model");
            let content = msg.content.as_deref().unwrap_or("");
            if !trainable || content.is_empty() {
                continue;
            }
            let content_ids = tokenizer
                .encode(content, false)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .to_vec();
            if content_ids.is_empty() {
                continue;
            }
            if let Some(start) = find_subslice_from(&full_ids, &content_ids, search_from) {
                let end = start + content_ids.len();
                for idx in start..end {
                    mask[idx] = 1;
                }
                search_from = end;
            }
        }

        if !mask.iter().any(|&m| m == 1) {
            mask.iter_mut().for_each(|m| *m = 1);
        }

        return Ok((full_ids, mask));
    }

    let mut raw_ids = Vec::new();
    let mut raw_mask = Vec::new();
    let mut full_text = String::new();
    for msg in &item.messages {
        let text = format_message_as_text(msg)?;
        let encoding = tokenizer
            .encode(text.as_str(), false)
            .map_err(anyhow::Error::msg)?;
        let msg_ids = encoding.get_ids();
        let trainable = matches!(msg.role.as_str(), "assistant" | "model");
        raw_ids.extend_from_slice(msg_ids);
        raw_mask.extend(std::iter::repeat(if trainable { 1u8 } else { 0u8 }).take(msg_ids.len()));
        full_text.push_str(&text);
    }

    let full_encoding = tokenizer
        .encode(full_text, true)
        .map_err(anyhow::Error::msg)?;
    let full_ids = full_encoding.get_ids().to_vec();

    if full_ids.len() == raw_ids.len() {
        return Ok((raw_ids, raw_mask));
    }

    if let Some(start) = find_subslice(&full_ids, &raw_ids) {
        let mut mask = vec![0u8; full_ids.len()];
        mask[start..start + raw_mask.len()].copy_from_slice(&raw_mask);
        return Ok((full_ids, mask));
    }

    Ok((raw_ids, raw_mask))
}

fn find_subslice(haystack: &[u32], needle: &[u32]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    for start in 0..=(haystack.len() - needle.len()) {
        if haystack[start..start + needle.len()] == *needle {
            return Some(start);
        }
    }
    None
}

fn find_subslice_from(haystack: &[u32], needle: &[u32], start_at: usize) -> Option<usize> {
    if needle.is_empty() || start_at >= haystack.len() {
        return None;
    }
    let haystack = &haystack[start_at..];
    find_subslice(haystack, needle).map(|idx| idx + start_at)
}

fn truncate_sequence_with_mask(seq: &mut Vec<u32>, mask: &mut Vec<u8>, cap: usize) {
    if seq.len() <= cap {
        return;
    }
    let mut start = 0usize;
    if let Some(last_one) = mask.iter().rposition(|&m| m == 1) {
        if last_one + 1 > cap {
            start = last_one + 1 - cap;
        }
    }
    let end = start + cap;
    *seq = seq[start..end].to_vec();
    *mask = mask[start..end].to_vec();
}

fn pack_token_sequences_with_mask(
    mut sequences: Vec<Vec<u32>>,
    mut masks: Vec<Vec<u8>>,
    max_len: usize,
    eos_token_id: u32,
) -> (Vec<Vec<u32>>, Vec<Vec<u8>>) {
    let mut packed = Vec::new();
    let mut packed_masks = Vec::new();
    let mut current = Vec::new();
    let mut current_mask = Vec::new();

    for (seq, mask) in sequences.drain(..).zip(masks.drain(..)) {
        if seq.is_empty() {
            continue;
        }
        let extra = if current.is_empty() {
            seq.len()
        } else {
            seq.len() + 1
        };
        if current.len() + extra > max_len && !current.is_empty() {
            packed.push(current);
            packed_masks.push(current_mask);
            current = Vec::new();
            current_mask = Vec::new();
        }
        if !current.is_empty() {
            current.push(eos_token_id);
            current_mask.push(0);
        }
        current.extend(seq);
        current_mask.extend(mask);
    }

    if !current.is_empty() {
        packed.push(current);
        packed_masks.push(current_mask);
    }
    (packed, packed_masks)
}

impl TokenCache {
    /// Load a token cache without validating the tokenizer hash.
    ///
    /// Magic bytes and version number are still checked when the file starts
    /// with the `PCAI` marker.  Legacy caches that pre-date versioning are
    /// loaded transparently with `has_header = false`.
    pub fn load(cache_dir: &Path) -> Result<Self> {
        Self::load_with_validation(cache_dir, None)
    }

    /// Load a token cache and verify it was built with the given tokenizer.
    ///
    /// Returns an error if the tokenizer hash in the cache header does not
    /// match `tokenizer_path`, preventing silent data corruption from stale
    /// caches.
    pub fn load_validated(cache_dir: &Path, tokenizer_path: &Path) -> Result<Self> {
        Self::load_with_validation(cache_dir, Some(tokenizer_path))
    }

    fn load_with_validation(cache_dir: &Path, tokenizer_path: Option<&Path>) -> Result<Self> {
        let bin_path = cache_dir.join("tokens.bin");
        let mask_path = cache_dir.join("tokens.mask.bin");
        let idx_path = cache_dir.join("tokens.idx.json");

        let idx_raw = std::fs::read_to_string(&idx_path)?;
        let entries: Vec<TokenCacheEntry> = serde_json::from_str(&idx_raw)?;

        let file = File::open(&bin_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Detect whether this is a versioned cache by checking the magic bytes.
        let has_header = mmap.len() >= CACHE_HEADER_SIZE && &mmap[0..4] == CACHE_MAGIC;

        if has_header {
            if let Some(tp) = tokenizer_path {
                // Full validation: magic + version + tokenizer hash.
                let expected_hash = compute_tokenizer_hash(tp)?;
                validate_cache_header(&mmap, &expected_hash)?;
            } else {
                // Partial validation: magic + version only (no tokenizer path supplied).
                let version = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]);
                if version != CACHE_VERSION {
                    anyhow::bail!(
                        "Token cache version mismatch: file has v{}, expected v{}. \
                         Rebuild the cache.",
                        version,
                        CACHE_VERSION
                    );
                }
            }
        }

        let mask_mmap = if mask_path.exists() {
            let mask_file = File::open(&mask_path)?;
            let m = unsafe { Mmap::map(&mask_file)? };
            // Validate the mask file header when the token file had one.
            // We skip the tokenizer hash here because both files must have
            // been written together; a mismatch would be an extremely unusual
            // scenario (partial file replacement).  Magic + version is enough
            // to catch truncated or alien files.
            if has_header && m.len() >= CACHE_HEADER_SIZE && &m[0..4] == CACHE_MAGIC {
                let version = u32::from_le_bytes([m[4], m[5], m[6], m[7]]);
                if version != CACHE_VERSION {
                    anyhow::bail!(
                        "Token mask cache version mismatch: file has v{}, expected v{}. \
                         Rebuild the cache.",
                        version,
                        CACHE_VERSION
                    );
                }
            }
            Some(m)
        } else {
            None
        };

        Ok(Self {
            mmap,
            entries,
            mask_mmap,
            has_header,
        })
    }

    /// Return the token IDs for the entry at `index`.
    pub fn get_ids(&self, index: usize) -> Result<Vec<u32>> {
        let entry = self
            .entries
            .get(index)
            .context("Token cache index out of range")?;
        let header_offset = if self.has_header {
            CACHE_HEADER_SIZE
        } else {
            0
        };
        let start = header_offset + (entry.offset * 4) as usize;
        let end = start + (entry.len as usize * 4);
        let bytes = self
            .mmap
            .get(start..end)
            .context("Token cache slice out of range")?;
        let mut ids = Vec::with_capacity(entry.len as usize);
        for chunk in bytes.chunks_exact(4) {
            ids.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(ids)
    }

    /// Return the loss-mask bytes for the entry at `index`.
    ///
    /// Returns an all-ones mask when no mask file is present (train on every
    /// token).
    pub fn get_mask(&self, index: usize) -> Result<Vec<u8>> {
        let entry = self
            .entries
            .get(index)
            .context("Token cache index out of range")?;
        if let Some(mask_mmap) = &self.mask_mmap {
            let header_offset = if self.has_header {
                CACHE_HEADER_SIZE
            } else {
                0
            };
            let start = header_offset + entry.offset as usize;
            let end = start + (entry.len as usize);
            let bytes = mask_mmap
                .get(start..end)
                .context("Token cache mask slice out of range")?;
            Ok(bytes.to_vec())
        } else {
            Ok(vec![1u8; entry.len as usize])
        }
    }
}

#[cfg(test)]
mod cache_validation_tests {
    use super::*;
    use tempfile::TempDir;

    // ---- header round-trip ----

    #[test]
    fn test_cache_header_roundtrip() {
        let hash = [1u8; 16];
        let mut buf = Vec::new();
        write_cache_header(&mut buf, &hash).expect("TODO: Verify unwrap");
        assert_eq!(buf.len(), CACHE_HEADER_SIZE);
        assert_eq!(&buf[0..4], CACHE_MAGIC);
        validate_cache_header(&buf, &hash).expect("TODO: Verify unwrap");
    }

    // ---- individual failure modes ----

    #[test]
    fn test_cache_header_bad_magic() {
        let mut buf = vec![0u8; CACHE_HEADER_SIZE];
        buf[0..4].copy_from_slice(b"XXXX");
        let err = validate_cache_header(&buf, &[0u8; 16]).unwrap_err();
        assert!(
            err.to_string().contains("invalid magic"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_cache_header_version_mismatch() {
        let hash = [0u8; 16];
        let mut buf = Vec::new();
        write_cache_header(&mut buf, &hash).expect("TODO: Verify unwrap");
        // Overwrite version field with 99.
        buf[4..8].copy_from_slice(&99u32.to_le_bytes());
        let err = validate_cache_header(&buf, &hash).unwrap_err();
        assert!(
            err.to_string().contains("version mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_cache_header_tokenizer_mismatch() {
        let hash_a = [1u8; 16];
        let hash_b = [2u8; 16];
        let mut buf = Vec::new();
        write_cache_header(&mut buf, &hash_a).expect("TODO: Verify unwrap");
        let err = validate_cache_header(&buf, &hash_b).unwrap_err();
        assert!(
            err.to_string().contains("different tokenizer"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_cache_header_too_small() {
        let buf = vec![0u8; 10];
        let err = validate_cache_header(&buf, &[0u8; 16]).unwrap_err();
        assert!(
            err.to_string().contains("too small"),
            "unexpected error: {err}"
        );
    }

    // ---- tokenizer hash ----

    #[test]
    fn test_compute_tokenizer_hash_deterministic() {
        let dir = TempDir::new().expect("TODO: Verify unwrap");
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, b"test tokenizer content").expect("TODO: Verify unwrap");
        let hash1 = compute_tokenizer_hash(&path).expect("TODO: Verify unwrap");
        let hash2 = compute_tokenizer_hash(&path).expect("TODO: Verify unwrap");
        assert_eq!(hash1, hash2, "hash must be deterministic for the same file");
    }

    #[test]
    fn test_compute_tokenizer_hash_differs_for_different_content() {
        let dir = TempDir::new().expect("TODO: Verify unwrap");
        let path_a = dir.path().join("tok_a.json");
        let path_b = dir.path().join("tok_b.json");
        std::fs::write(&path_a, b"tokenizer A").expect("TODO: Verify unwrap");
        std::fs::write(&path_b, b"tokenizer B").expect("TODO: Verify unwrap");
        let hash_a = compute_tokenizer_hash(&path_a).expect("TODO: Verify unwrap");
        let hash_b = compute_tokenizer_hash(&path_b).expect("TODO: Verify unwrap");
        assert_ne!(
            hash_a, hash_b,
            "different content must yield different hashes"
        );
    }
}
