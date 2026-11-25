//! Framed entry stream for file/folder transfers (`UploadStart` /
//! `DownloadStart` bodies).
//!
//! A transfer rides ONE transport body: a sequence of framed entries
//! produced while lazily walking the source items (files and/or folders).
//! Every item enters the stream under its own name: the first `rel_path`
//! component is the item's file name. Layout per frame (all integers
//! little-endian):
//!
//! ```text
//! 0x01 u16 rel_len  rel_path (UTF-8, '/'-separated)  u64 size   FileBegin
//! 0x02 u32 len  chunk…                                          Chunk (file content)
//! 0x03 u64 bytes                                                    FileEnd
//! 0x04                                                              End (normal completion)
//! 0x05 u16 len  reason (UTF-8)                                      Abort
//! 0x06 u16 rel_len  rel_path  u16 len  reason                       Skip
//! ```
//!
//! `size` on FileBegin is the sender-declared file length; exactly `size`
//! chunk bytes follow. Both sides share the writer (`FolderFrameWriter`) and
//! the incremental parser (`FolderFrameDecoder`).

use bytes::{Buf, BufMut, BytesMut};
use std::path::PathBuf;

pub fn unique_destination_path(target: PathBuf) -> PathBuf {
    if !target.exists() {
        return target;
    }
    let parent = target.parent();
    let stem = target
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let ext = target
        .extension()
        .map(|s| format!(".{}", s.to_string_lossy().into_owned()))
        .unwrap_or_default();

    let join = |name: String| -> PathBuf {
        parent
            .map(|p| p.join(&name))
            .unwrap_or_else(|| PathBuf::from(name))
    };

    let first = join(format!("{stem} copy{ext}"));
    if !first.exists() {
        return first;
    }
    for n in 2..100_000 {
        let candidate = join(format!("{stem} copy {n}{ext}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    target
}

/// Frame kind markers.
pub mod kind {
    pub const FILE_BEGIN: u8 = 0x01;
    pub const CHUNK: u8 = 0x02;
    pub const FILE_END: u8 = 0x03;
    pub const END: u8 = 0x04;
    pub const ABORT: u8 = 0x05;
    pub const SKIP: u8 = 0x06;
}

/// Incremental writer producing the framed entry stream.
#[derive(Default)]
pub struct FolderFrameWriter {
    buf: BytesMut,
}

impl FolderFrameWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn file_begin(&mut self, rel_path: &str, size: u64) {
        self.buf.put_u8(kind::FILE_BEGIN);
        self.buf.put_u16_le(rel_path.len() as u16);
        self.buf.put_slice(rel_path.as_bytes());
        self.buf.put_u64_le(size);
    }

    pub fn chunk(&mut self, chunk: &[u8]) {
        self.buf.put_u8(kind::CHUNK);
        self.buf.put_u32_le(chunk.len() as u32);
        self.buf.put_slice(chunk);
    }

    pub fn file_end(&mut self, bytes: u64) {
        self.buf.put_u8(kind::FILE_END);
        self.buf.put_u64_le(bytes);
    }

    pub fn end(&mut self) {
        self.buf.put_u8(kind::END);
    }

    pub fn abort(&mut self, reason: &str) {
        self.buf.put_u8(kind::ABORT);
        self.buf.put_u16_le(reason.len() as u16);
        self.buf.put_slice(reason.as_bytes());
    }

    pub fn skip(&mut self, rel_path: &str, reason: &str) {
        self.buf.put_u8(kind::SKIP);
        self.buf.put_u16_le(rel_path.len() as u16);
        self.buf.put_slice(rel_path.as_bytes());
        self.buf.put_u16_le(reason.len() as u16);
        self.buf.put_slice(reason.as_bytes());
    }

    /// Take everything buffered so far.
    pub fn take(&mut self) -> BytesMut {
        self.buf.split()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

/// A fully parsed frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FolderFrame {
    FileBegin { rel_path: String, size: u64 },
    Chunk(Vec<u8>),
    FileEnd { bytes: u64 },
    End,
    Abort { reason: String },
    Skip { rel_path: String, reason: String },
}

/// Errors produced by the decoder on malformed streams.
#[derive(Debug)]
pub enum FolderDecodeError {
    /// The stream signalled an explicit Abort.
    Aborted(String),
    /// Anything else: truncated header, unknown kind, chunk without begin…
    Malformed(String),
}

/// Incremental decoder: feed transport chunks, get frames out. Tracks the
/// current file so CHUNK payloads are sliced against their length prefix;
/// a chunk payload may span any number of `push` calls.
#[derive(Default)]
pub struct FolderFrameDecoder {
    buf: BytesMut,
    /// (rel_path, declared size, consumed so far) of the file being read.
    current: Option<(String, u64, u64)>,
    /// Payload bytes still expected for the chunk being read. While this is
    /// nonzero the buffer holds raw payload, not frame headers.
    chunk_remaining: usize,
    finished: bool,
    aborted: bool,
}

impl FolderFrameDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_finished(&self) -> bool {
        self.finished || self.aborted
    }

    /// Feed raw stream bytes; returns every frame that became complete.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<FolderFrame>, FolderDecodeError> {
        if self.finished || self.aborted {
            return Ok(vec![]);
        }
        self.buf.extend_from_slice(chunk);
        let mut frames = Vec::new();
        loop {
            // Chunk payload mode: the bytes are raw file content, not frame
            // headers, and may arrive split across any number of pushes.
            if self.chunk_remaining > 0 {
                let Some((_, size, consumed)) = self.current.as_mut() else {
                    self.aborted = true;
                    return Err(FolderDecodeError::Malformed(
                        "chunk payload without FileBegin".into(),
                    ));
                };
                let take = self.chunk_remaining.min(self.buf.len());
                let payload = self.buf[..take].to_vec();
                self.buf.advance(take);
                self.chunk_remaining -= take;
                *consumed += take as u64;
                if *consumed == *size {
                    self.current = None;
                }
                if take > 0 {
                    frames.push(FolderFrame::Chunk(payload));
                }
                if self.buf.is_empty() {
                    break;
                }
            }
            let Some(kind) = self.buf.first().copied() else {
                break;
            };
            match kind {
                kind::FILE_BEGIN => {
                    if self.buf.len() < 3 {
                        break;
                    }
                    let rel_len = u16::from_le_bytes([self.buf[1], self.buf[2]]) as usize;
                    if self.buf.len() < 3 + rel_len + 8 {
                        break;
                    }
                    let rel_path = String::from_utf8_lossy(&self.buf[3..3 + rel_len]).into_owned();
                    let size_bytes = &self.buf[3 + rel_len..3 + rel_len + 8];
                    let size = u64::from_le_bytes(size_bytes.try_into().unwrap());
                    self.buf.advance(3 + rel_len + 8);
                    self.current = Some((rel_path.clone(), size, 0));
                    frames.push(FolderFrame::FileBegin { rel_path, size });
                }
                kind::CHUNK => {
                    let Some((_, size, consumed)) = self.current.as_ref() else {
                        self.aborted = true;
                        return Err(FolderDecodeError::Malformed(
                            "chunk before FileBegin".into(),
                        ));
                    };
                    if self.buf.len() < 5 {
                        break;
                    }
                    let chunk_len = u32::from_le_bytes(self.buf[1..5].try_into().unwrap()) as usize;
                    let file_remaining = *size - *consumed;
                    if chunk_len as u64 > file_remaining {
                        self.aborted = true;
                        return Err(FolderDecodeError::Malformed(
                            "chunk exceeds declared file size".into(),
                        ));
                    }
                    self.buf.advance(5);
                    self.chunk_remaining = chunk_len;
                }
                kind::FILE_END => {
                    if self.buf.len() < 9 {
                        break;
                    }
                    let bytes = u64::from_le_bytes(self.buf[1..9].try_into().unwrap());
                    self.buf.advance(9);
                    self.current = None;
                    frames.push(FolderFrame::FileEnd { bytes });
                }
                kind::END => {
                    self.buf.advance(1);
                    self.finished = true;
                    frames.push(FolderFrame::End);
                    break;
                }
                kind::ABORT => {
                    if self.buf.len() < 3 {
                        break;
                    }
                    let len = u16::from_le_bytes([self.buf[1], self.buf[2]]) as usize;
                    if self.buf.len() < 3 + len {
                        break;
                    }
                    let reason = String::from_utf8_lossy(&self.buf[3..3 + len]).into_owned();
                    self.buf.advance(3 + len);
                    self.aborted = true;
                    frames.push(FolderFrame::Abort { reason });
                    break;
                }
                kind::SKIP => {
                    if self.buf.len() < 5 {
                        break;
                    }
                    let rel_len = u16::from_le_bytes([self.buf[1], self.buf[2]]) as usize;
                    if self.buf.len() < 3 + rel_len + 2 {
                        break;
                    }
                    let reason_len =
                        u16::from_le_bytes([self.buf[3 + rel_len], self.buf[4 + rel_len]]) as usize;
                    if self.buf.len() < 5 + rel_len + reason_len {
                        break;
                    }
                    let rel_path = String::from_utf8_lossy(&self.buf[3..3 + rel_len]).into_owned();
                    let reason =
                        String::from_utf8_lossy(&self.buf[5 + rel_len..5 + rel_len + reason_len])
                            .into_owned();
                    self.buf.advance(5 + rel_len + reason_len);
                    frames.push(FolderFrame::Skip { rel_path, reason });
                }
                unknown => {
                    self.aborted = true;
                    return Err(FolderDecodeError::Malformed(format!(
                        "unknown frame kind 0x{unknown:02x}"
                    )));
                }
            }
        }
        Ok(frames)
    }
}

/// Async reader over a multi-root walk producing the framed entry stream.
/// Both the upload side (client, local items) and the download side (server,
/// remote items) drive it the same way. Walks lazily: directories are read
/// as the stream advances, so an aborted transfer stops the walk.
pub struct FolderStreamReader {
    /// Roots not yet started: (fs path, rel prefix).
    pending_roots: std::collections::VecDeque<(std::path::PathBuf, String)>,
    /// Root directory currently being walked and its rel prefix.
    current_root: Option<(std::path::PathBuf, String)>,
    /// Iterator over the directory currently being streamed.
    current_dir: Option<std::fs::ReadDir>,
    /// Subdirectories discovered but not yet entered.
    pending_dirs: std::collections::VecDeque<std::path::PathBuf>,
    /// File roots (or unreadable-root skips resolved) queued for streaming:
    /// (fs path, rel path).
    pending_files: std::collections::VecDeque<(std::path::PathBuf, String)>,
    /// Skip frames discovered while starting roots, drained into the stream.
    pending_skips: std::collections::VecDeque<(String, String)>,
    /// Open file being streamed, with remaining bytes.
    current: Option<(std::fs::File, u64)>,
    /// Outgoing framed bytes not yet consumed by the caller.
    out: BytesMut,
    writer: FolderFrameWriter,
    done: bool,
    /// Set from outside to abort the walk (task cancellation); the stream
    /// then emits an Abort frame and finishes.
    pub abort: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Snapshot for the UI ticker: (rel_path, transferred, size) of the
    /// file currently being streamed, if any.
    pub state: std::sync::Arc<std::sync::Mutex<FolderWalkState>>,
    pub files_done: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

/// Shared snapshot of the walk for progress display.
#[derive(Debug, Default, Clone)]
pub struct FolderWalkState {
    /// '/'-separated path of the file currently being streamed.
    pub current_file: String,
    /// Bytes of the current file already streamed.
    pub current_transferred: u64,
    /// Declared size of the current file (0 when unknown).
    pub current_size: u64,
    pub files_done: u64,
}

impl FolderStreamReader {
    /// Every root enters the stream under its own name: the first
    /// `rel_path` component is the root's file name. File roots stream as a
    /// single entry; directory roots stream their contents recursively
    /// underneath that prefix. A root without a file name (e.g. a
    /// filesystem root) streams its contents unprefixed.
    pub fn new(
        roots: Vec<std::path::PathBuf>,
        abort: std::sync::Arc<std::sync::atomic::AtomicBool>,
        state: std::sync::Arc<std::sync::Mutex<FolderWalkState>>,
    ) -> Self {
        let pending_roots = roots
            .into_iter()
            .map(|root| {
                let prefix = root
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                (root, prefix)
            })
            .collect();
        Self {
            pending_roots,
            current_root: None,
            current_dir: None,
            pending_dirs: std::collections::VecDeque::new(),
            pending_files: std::collections::VecDeque::new(),
            pending_skips: std::collections::VecDeque::new(),
            current: None,
            out: BytesMut::new(),
            writer: FolderFrameWriter::new(),
            done: false,
            abort,
            state,
            files_done: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Relative path ('/'-separated) of `path` under the current walk root,
    /// prefixed with the root's own name.
    fn rel_of(&self, path: &std::path::Path) -> String {
        let Some((root, prefix)) = self.current_root.as_ref() else {
            return path.to_string_lossy().into_owned();
        };
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        if prefix.is_empty() {
            rel
        } else if rel.is_empty() {
            prefix.clone()
        } else {
            format!("{prefix}/{rel}")
        }
    }

    /// Pull the next regular file from the walk, starting roots and
    /// descending into subdirectories as they are discovered. Returns the
    /// fs path plus its framed rel path.
    fn next_file(&mut self) -> std::io::Result<Option<(std::path::PathBuf, String)>> {
        loop {
            if let Some(skip) = self.pending_skips.pop_front() {
                self.writer.skip(&skip.0, &skip.1);
                self.out.extend(self.writer.take());
            }
            if let Some(queued) = self.pending_files.pop_front() {
                return Ok(Some(queued));
            }
            if let Some(iter) = self.current_dir.as_mut() {
                for entry in iter.by_ref() {
                    let Ok(entry) = entry else { continue };
                    let path = entry.path();
                    let Ok(file_type) = entry.file_type() else {
                        continue;
                    };
                    if file_type.is_dir() {
                        self.pending_dirs.push_back(path);
                        continue;
                    }
                    if file_type.is_file() {
                        let rel = self.rel_of(&path);
                        return Ok(Some((path, rel)));
                    }
                    // Symlinks and specials are skipped rather than followed.
                }
                self.current_dir = None;
            }
            // Current directory exhausted; enter the next queued one.
            if let Some(dir) = self.pending_dirs.pop_front() {
                self.current_dir = std::fs::read_dir(&dir).ok();
                continue;
            }
            // Current root exhausted; start the next one.
            let Some((root, prefix)) = self.pending_roots.pop_front() else {
                return Ok(None);
            };
            match std::fs::metadata(&root) {
                Ok(metadata) if metadata.is_dir() => {
                    self.current_root = Some((root.clone(), prefix));
                    self.pending_dirs.push_back(root);
                }
                Ok(metadata) if metadata.is_file() => {
                    let rel = if prefix.is_empty() {
                        root.to_string_lossy().into_owned()
                    } else {
                        prefix
                    };
                    self.pending_files.push_back((root, rel));
                }
                Ok(_) => {
                    // Symlinks and specials are skipped rather than followed.
                    self.pending_skips.push_back((prefix, "not a file".into()));
                }
                Err(_) => {
                    self.pending_skips.push_back((prefix, "unreadable".into()));
                }
            }
        }
    }
}

impl std::io::Read for FolderStreamReader {
    fn read(&mut self, out_buf: &mut [u8]) -> std::io::Result<usize> {
        loop {
            // Serve buffered bytes first.
            if !self.out.is_empty() {
                let take = self.out.len().min(out_buf.len());
                let bytes = self.out.split_to(take);
                out_buf[..take].copy_from_slice(&bytes);
                return Ok(take);
            }
            if self.done {
                return Ok(0);
            }
            // Abort requested from outside.
            if self.abort.load(std::sync::atomic::Ordering::Relaxed) {
                self.writer.abort("cancelled");
                self.out.extend(self.writer.take());
                self.done = true;
                continue;
            }
            // Stream the open file.
            if let Some((file, remaining)) = self.current.as_mut() {
                let take = (*remaining).min(out_buf.len() as u64).min(64 * 1024) as usize;
                let mut chunk = vec![0u8; take];
                let read = file.read(&mut chunk)?;
                if read == 0 {
                    // File truncated on disk: close the entry short.
                    self.writer.file_end(*remaining);
                    self.out.extend(self.writer.take());
                    self.current = None;
                    continue;
                }
                self.writer.chunk(&chunk[..read]);
                self.out.extend(self.writer.take());
                *remaining -= read as u64;
                if let Ok(mut state) = self.state.lock() {
                    state.current_transferred += read as u64;
                }
                if *remaining == 0 {
                    self.writer.file_end(0);
                    self.out.extend(self.writer.take());
                    if let Ok(mut state) = self.state.lock() {
                        state.files_done += 1;
                    }
                    self.files_done
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    self.current = None;
                }
                continue;
            }
            // Open the next file, or finish the walk.
            let Some((next, rel)) = self.next_file()? else {
                self.writer.end();
                self.out.extend(self.writer.take());
                self.done = true;
                continue;
            };
            let Ok(metadata) = std::fs::metadata(&next) else {
                self.writer.skip(&rel, "unreadable");
                self.out.extend(self.writer.take());
                continue;
            };
            let Ok(file) = std::fs::File::open(&next) else {
                self.writer.skip(&rel, "cannot open");
                self.out.extend(self.writer.take());
                continue;
            };
            let size = metadata.len();
            self.writer.file_begin(&rel, size);
            self.out.extend(self.writer.take());
            if let Ok(mut state) = self.state.lock() {
                state.current_file = rel.clone();
                state.current_transferred = 0;
                state.current_size = size;
            }
            self.current = Some((file, size));
        }
    }
}

/// The walk does synchronous local-file IO internally; expose it as a tokio
/// reader so it can ride `OutboundBody::Reader` (server download responses).
impl tokio::io::AsyncRead for FolderStreamReader {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        use std::io::Read;
        let unfilled = buf.initialize_unfilled();
        match self.read(unfilled) {
            Ok(0) => std::task::Poll::Ready(Ok(())),
            Ok(n) => {
                buf.advance(n);
                std::task::Poll::Ready(Ok(()))
            }
            Err(error) => std::task::Poll::Ready(Err(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::io::Write;

    /// Build a temp tree with files straddling every chunking boundary the
    /// transport can produce; returns (root, [(rel_path, content)]).
    fn build_tree() -> (std::path::PathBuf, Vec<(String, Vec<u8>)>) {
        let root = std::env::temp_dir().join(format!("vterm-folder-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let sizes = [0usize, 1, 1907, 65535, 65536, 65537, 73434, 200_000];
        let mut files = Vec::new();
        for (index, size) in sizes.iter().enumerate() {
            let rel = if index % 2 == 0 {
                format!("dir{index}/sub/file{index}.bin")
            } else {
                format!("file{index}.bin")
            };
            let path = root.join(&rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            // Content must not be uniform: derive each byte from its offset
            // so a shifted stream fails the content comparison.
            let content: Vec<u8> = (0..*size).map(|i| (i % 251) as u8).collect();
            std::fs::File::create(&path)
                .unwrap()
                .write_all(&content)
                .unwrap();
            files.push((rel, content));
        }
        (root, files)
    }

    /// Stream the tree with varying read buffer sizes, then decode with
    /// varying push sizes; every file must reassemble byte-identically.
    #[test]
    fn round_trip_survives_any_chunking() {
        let (root, files) = build_tree();
        let abort = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let state = std::sync::Arc::new(std::sync::Mutex::new(FolderWalkState::default()));
        let mut reader = FolderStreamReader::new(vec![root.clone()], abort, state);

        // Produce the stream through realistic read() buffer sizes, including
        // the transport's 60 KiB - 44 frame payload and tiny short buffers.
        let mut stream = Vec::new();
        let read_sizes = [61396usize, 1, 7, 65536, 100_000, 3];
        let mut cursor = 0usize;
        loop {
            let size = read_sizes[cursor % read_sizes.len()];
            cursor += 1;
            let mut buf = vec![0u8; size];
            let n = reader.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            stream.extend_from_slice(&buf[..n]);
            assert!(stream.len() < 4 * 1024 * 1024, "stream never terminates");
        }

        // Decode with another set of awkward push sizes.
        let mut decoder = FolderFrameDecoder::new();
        let push_sizes = [1usize, 61396, 3, 65537, 11];
        let mut cursor = 0usize;
        let mut offset = 0usize;
        let mut frames = Vec::new();
        while offset < stream.len() {
            let size = push_sizes[cursor % push_sizes.len()];
            cursor += 1;
            let end = (offset + size).min(stream.len());
            frames.extend(decoder.push(&stream[offset..end]).unwrap());
            offset = end;
        }
        assert!(decoder.is_finished(), "decoder did not see the End frame");

        // Reassemble and compare against the source tree.
        let mut assembled: Vec<(String, Vec<u8>)> = Vec::new();
        let mut current: Option<(String, Vec<u8>)> = None;
        let mut saw_end = false;
        for frame in frames {
            match frame {
                FolderFrame::FileBegin { rel_path, size } => {
                    assert!(current.is_none(), "FileBegin inside another file");
                    current = Some((rel_path, Vec::with_capacity(size as usize)));
                }
                FolderFrame::Chunk(data) => {
                    current
                        .as_mut()
                        .expect("chunk without begin")
                        .1
                        .extend(data);
                }
                FolderFrame::FileEnd { .. } => {
                    let (rel, content) = current.take().expect("end without begin");
                    assembled.push((rel, content));
                }
                FolderFrame::End => saw_end = true,
                other => panic!("unexpected frame {other:?}"),
            }
        }
        assert!(saw_end);

        let prefix = root.file_name().unwrap().to_string_lossy().into_owned();
        let mut expected: Vec<(String, Vec<u8>)> = files
            .into_iter()
            .map(|(rel, content)| (format!("{prefix}/{rel}"), content))
            .collect();
        expected.sort();
        assembled.sort();
        assert_eq!(assembled.len(), expected.len(), "file count mismatch");
        for ((rel_a, content_a), (rel_b, content_b)) in assembled.iter().zip(expected.iter()) {
            assert_eq!(rel_a, rel_b);
            assert_eq!(
                content_a.len(),
                content_b.len(),
                "length mismatch for {rel_a}"
            );
            assert!(content_a == content_b, "content mismatch for {rel_a}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A mixed selection (a directory, a plain file, and a missing path)
    /// streams every item under its own name; the missing root surfaces as
    /// a Skip frame instead of aborting the transfer.
    #[test]
    fn multi_root_streams_each_item_under_its_name() {
        let (root, files) = build_tree();
        let standalone = root.parent().unwrap().join(format!(
            "vterm-folder-standalone-{}.txt",
            std::process::id()
        ));
        let standalone_content: Vec<u8> = (0..1234u32).map(|i| (i % 241) as u8).collect();
        std::fs::File::create(&standalone)
            .unwrap()
            .write_all(&standalone_content)
            .unwrap();
        let missing = root
            .parent()
            .unwrap()
            .join(format!("vterm-folder-missing-{}", std::process::id()));

        let abort = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let state = std::sync::Arc::new(std::sync::Mutex::new(FolderWalkState::default()));
        let mut reader = FolderStreamReader::new(
            vec![root.clone(), standalone.clone(), missing.clone()],
            abort,
            state,
        );
        let mut stream = Vec::new();
        let mut buf = vec![0u8; 61396];
        loop {
            let n = reader.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            stream.extend_from_slice(&buf[..n]);
            assert!(stream.len() < 4 * 1024 * 1024, "stream never terminates");
        }

        let mut decoder = FolderFrameDecoder::new();
        let frames = decoder.push(&stream).unwrap();
        assert!(decoder.is_finished(), "decoder did not see the End frame");

        let root_name = root.file_name().unwrap().to_string_lossy().into_owned();
        let standalone_name = standalone
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let missing_name = missing.file_name().unwrap().to_string_lossy().into_owned();

        let mut assembled: Vec<(String, Vec<u8>)> = Vec::new();
        let mut current: Option<(String, Vec<u8>)> = None;
        let mut skips = Vec::new();
        for frame in frames {
            match frame {
                FolderFrame::FileBegin { rel_path, size } => {
                    assert!(current.is_none(), "FileBegin inside another file");
                    current = Some((rel_path, Vec::with_capacity(size as usize)));
                }
                FolderFrame::Chunk(data) => {
                    current
                        .as_mut()
                        .expect("chunk without begin")
                        .1
                        .extend(data);
                }
                FolderFrame::FileEnd { .. } => {
                    assembled.push(current.take().expect("end without begin"));
                }
                FolderFrame::End => {}
                FolderFrame::Skip { rel_path, .. } => skips.push(rel_path),
                other => panic!("unexpected frame {other:?}"),
            }
        }

        let mut expected: Vec<(String, Vec<u8>)> = files
            .into_iter()
            .map(|(rel, content)| (format!("{root_name}/{rel}"), content))
            .collect();
        expected.push((standalone_name, standalone_content));
        expected.sort();
        assembled.sort();
        assert_eq!(assembled, expected);
        assert_eq!(skips, vec![missing_name]);

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_file(&standalone);
    }
}
