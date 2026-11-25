use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use vterm_common::{
    AiConversationInfo, AiConversationMessage, AiConversationMessageKind, AiConversationRole,
    ErrorCode, FileListItem, FileSearchMatch, FileSearchResult, GitBranchInfo, GitChangedFile,
    GitCommitInfo, GitCommitMeta, GitFileStatus, GitRebaseAiHunkEvent, GitRebaseAiTarget,
    GitRebaseState, GitRebaseTodoCommit, GitStashEntry, GitSubmoduleInfo, KnownAppInfo,
    PathNameMatch, PortForwardInfo, RecentProjectDirectory, RelayServerInfo, SessionId,
    SessionInfo, TabGroup, TabGroupId, TabId, TabInfo, TabSplitPlacement, TerminalAttention,
    TerminalCommandHistory, TerminalCompletion, TerminalProfile, TerminalSize,
};

/// Non-secret settings advertised by each server to connected clients.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ServerInfo {
    pub config_port: u16,
    pub voice_enabled: bool,
    #[serde(default)]
    pub voice_polish_enabled: bool,
    pub git_commit_message_llm_valid: bool,
    pub translation_llm_valid: bool,
    /// Remote desktop (screen streaming) is enabled on the server. Drives
    /// client UI like the debug panel's screens bar. `#[serde(default)]` so
    /// older servers (without the field) deserialize as disabled.
    #[serde(default)]
    pub remote_desktop_enabled: bool,
}

const OUTPUT_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTO1";
const OUTPUT_BINARY_FRAME_HEADER_LEN: usize = 4 + 16 + 8;
const INPUT_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTI1";
const INPUT_BINARY_FRAME_HEADER_LEN: usize = 4 + 16 + 1;
const INPUT_BINARY_FRAME_CURSOR_HEADER_LEN: usize = INPUT_BINARY_FRAME_HEADER_LEN + 4 + 2;
const HISTORY_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTH1";
const HISTORY_SNAPSHOT_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTG1";
const HISTORY_BINARY_FRAME_HEADER_LEN: usize = 4 + 16 + 8 + 8;
const HISTORY_SNAPSHOT_BINARY_FRAME_HEADER_LEN: usize = 4 + 16 + 8;
const PATH_CONTENT_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTP1";
const FIND_FILES_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTN1";
const SEARCH_PATH_NAMES_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTY1";
const GIT_CHANGED_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTC2";
const GIT_CHANGED_BINARY_FRAME_MAGIC_V1: &[u8; 4] = b"VTC1";
const GIT_STAGE_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTS1";
const GIT_UNSTAGE_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTU1";
const GIT_DISCARD_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTQ1";
const GIT_LOG_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTL1";
const FILE_CONTENT_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTF1";
const GIT_DIFF_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTD1";
const GIT_BRANCHES_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTB1";
const AI_CONVERSATION_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTA1";
const AI_CONVERSATION_UPDATE_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTA2";
const GIT_ANALYSIS_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTA3";
const FILE_SAVE_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTF2";
const TRANSCRIBE_AUDIO_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTV2";
const GIT_REBASE_CONTENT_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTR1";
const GIT_REBASE_RESOLVE_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTR2";
const GIT_REBASE_START_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTR3";
const GIT_REBASE_AI_RESOLVE_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTR4";
const GIT_REBASE_AI_RESULT_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTR5";
const CHUNKED_BINARY_FRAME_MAGIC: &[u8; 4] = b"VTX1";
const CHUNKED_BINARY_FRAME_HEADER_LEN: usize = 4 + 1 + 8 + 8;
const CHUNK_FLAG_START: u8 = 0b01;
const CHUNK_FLAG_END: u8 = 0b10;
// Keep chunk payloads comfortably below the Noise transport single-message limit
// after adding the transport tag and chunk header.
pub const MAX_BINARY_TRANSPORT_CHUNK_SIZE: usize = 60 * 1024;

/// Byte buffer that serializes transparently but prints only its length in Debug.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ByteData(pub Vec<u8>);

impl std::fmt::Debug for ByteData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<{} bytes>", self.0.len())
    }
}

impl Deref for ByteData {
    type Target = Vec<u8>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ByteData {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Vec<u8>> for ByteData {
    fn from(value: Vec<u8>) -> Self {
        Self(value)
    }
}

impl From<ByteData> for Vec<u8> {
    fn from(value: ByteData) -> Self {
        value.0
    }
}

/// 0-based, end-exclusive range of current-file lines changed relative to git.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileDiffRange {
    pub start_line: u32,
    pub end_line: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionKind {
    Primary,
    Terminal,
    Request,
    Debug,
    RelayPrimary,
    RelayRequest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorFromEnd {
    pub lines_from_end: u32,
    pub column: u16,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FsWatchConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_content: Option<FsWatchFileContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git: Option<FsWatchGit>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub directories: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FsWatchFileContent {
    pub path: String,
    #[serde(default = "default_max_file_size")]
    pub max_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FsWatchGit {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugLaunchConfig {
    pub name: String,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub program: String,
    #[serde(default, rename = "type", alias = "adapterType")]
    pub adapter_type: String,
    #[serde(default)]
    pub request: String,
    #[serde(default, rename = "runtimeExecutable", alias = "runtime_executable")]
    pub runtime_executable: String,
    #[serde(default, rename = "runtimeArgs", alias = "runtime_args")]
    pub runtime_args: Vec<String>,
    #[serde(default, rename = "debugServer", alias = "debug_server")]
    pub debug_server: Option<u16>,
    #[serde(default, rename = "debugServerHost", alias = "debug_server_host")]
    pub debug_server_host: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub env: HashMap<String, String>,
    #[serde(
        default,
        rename = "preTasks",
        alias = "pre_tasks",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub pre_tasks: Vec<String>,
    #[serde(
        default,
        rename = "preLaunchTask",
        alias = "pre_launch_task",
        skip_serializing_if = "Option::is_none"
    )]
    pub pre_launch_task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    #[serde(default, rename = "adapterArgs", skip_serializing_if = "Vec::is_empty")]
    pub adapter_args: Vec<String>,
}

/// A named shell command exposed by a repository's `.vterm/commands.json` file.
///
/// The shell command lives in `command`; legacy files that used `value`
/// instead are still accepted (via the serde alias) and mapped to `command`.
/// `env` is an optional map of environment variables injected before the
/// command runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuickCommand {
    pub name: String,
    /// The shell command to run. Accepts the legacy `value` key for
    /// backward-compatible reads of older `commands.json` files.
    #[serde(alias = "value")]
    pub command: String,
    /// Optional environment variables to set before running the command.
    #[serde(default)]
    pub env: HashMap<String, String>,
}

/// Context passed with voice-input text polishing requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VoiceInputOptimizationContext {
    /// Free-form context that is not tied to a file or git revision.
    Transparent { session_id: SessionId },
    /// Context tied to a file at a specific git revision.
    Git {
        session_id: SessionId,
        file_path: String,
        git_hash: String,
        start_line: u32,
        end_line: u32,
    },
    /// Context tied to a file in the current working tree.
    File {
        session_id: SessionId,
        file_path: String,
        start_line: u32,
        end_line: u32,
    },
}

impl VoiceInputOptimizationContext {
    pub fn transparent(session_id: SessionId) -> Self {
        Self::Transparent { session_id }
    }

    pub fn git(
        session_id: SessionId,
        file_path: impl Into<String>,
        git_hash: impl Into<String>,
        start_line: u32,
        end_line: u32,
    ) -> Self {
        Self::Git {
            session_id,
            file_path: file_path.into(),
            git_hash: git_hash.into(),
            start_line,
            end_line,
        }
    }

    pub fn file(
        session_id: SessionId,
        file_path: impl Into<String>,
        start_line: u32,
        end_line: u32,
    ) -> Self {
        Self::File {
            session_id,
            file_path: file_path.into(),
            start_line,
            end_line,
        }
    }

    pub fn session_id(&self) -> SessionId {
        match self {
            Self::Transparent { session_id }
            | Self::Git { session_id, .. }
            | Self::File { session_id, .. } => *session_id,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TempFileUsage {
    StoreOnly,
    InsertIntoTerminal { bracketed_paste: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReverseProxyPurpose {
    ServerSettings,
}

/// All protocol messages between client and server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    // Handshake
    /// Initial handshake message with protocol and client version information
    Hello {
        protocol_version: u32,
        client_version: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        secret_key_hash: Option<String>,
        #[serde(default)]
        supports_binary_output: bool,
        #[serde(default)]
        supports_message_encryption: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        connection_kind: Option<ConnectionKind>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        auth_token: Option<String>,
        /// Terminal session selected before message encryption starts. Relay
        /// servers use this routing hint to connect a remote terminal to its
        /// owning shared server without decrypting the terminal data stream.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        target_session_id: Option<SessionId>,
        /// Short-lived capability generated by a relay for the shared server.
        /// Clients never set this field directly.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        relay_terminal_capability: Option<String>,
        /// Shared-server route selected before end-to-end request encryption.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        target_server_id: Option<String>,
        /// Single-use relay capability authorizing one request tunnel.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        relay_request_capability: Option<String>,
    },
    AuthChallenge {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        nonce: Option<String>,
        #[serde(default)]
        message_encryption: bool,
    },
    AuthResponse {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        proof: Option<String>,
    },
    /// Handshake acknowledgment with protocol and server version information
    HelloAck {
        protocol_version: u32,
        server_version: String,
        auth_token: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
        #[serde(default)]
        message_encryption: bool,
    },
    /// A server's non-secret capability/settings summary. Shared servers send
    /// their own information through the relay; the relay only forwards it.
    ServerInfo {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
        info: ServerInfo,
    },

    // Session management
    /// Create a new terminal session
    CreateSession {
        shell: String,
        profile_id: Option<String>,
        size: TerminalSize,
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        env: HashMap<String, String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cwd: Option<String>,
        /// Optional display name for the new session. When set, the server
        /// applies it as the session's custom name (equivalent to a rename).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        /// When set, the new session joins this session's group (split tab).
        /// Mutually exclusive with duplicate_session_id.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sibling_session_id: Option<SessionId>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
        /// Optional TabGroup target for the new session. If the group
        /// does not exist on the target server, the session remains ungrouped.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tab_group_id: Option<TabGroupId>,
        /// When set, inherit profile_id and cwd from this session (duplicate tab).
        /// Mutually exclusive with sibling_session_id.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duplicate_session_id: Option<SessionId>,
        /// Optional command to run in the newly created terminal after its
        /// shell has started in `cwd` (for example, `codex` or `claude`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        app: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        background: Option<[u8; 3]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        foreground: Option<[u8; 3]>,
    },
    /// Session created successfully
    SessionCreated {
        session_id: SessionId,
    },
    SplitTerminal {
        target_session_id: SessionId,
        placement: TabSplitPlacement,
        size: TerminalSize,
        /// Terminal profile to use for the new split session. When `None`, the
        /// server inherits the profile from the target session. Used by mobile
        /// clients to split with an explicit profile; desktop clients send
        /// `None` to inherit.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        profile_id: Option<String>,
    },
    ListSessions,
    SessionList {
        sessions: Vec<SessionInfo>,
    },
    ListTabGroups,
    TabGroupList {
        groups: Vec<TabGroup>,
    },
    CreateTabGroup {
        server_id: Option<String>,
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        initial_session_id: Option<SessionId>,
    },
    DeleteTabGroup {
        server_id: Option<String>,
        group_id: TabGroupId,
    },
    RenameTabGroup {
        server_id: Option<String>,
        group_id: TabGroupId,
        name: String,
    },
    MoveTabToGroup {
        server_id: Option<String>,
        session_id: SessionId,
        group_id: TabGroupId,
    },
    GetTabInfoList {
        server_id: Option<String>,
    },
    TabInfoList {
        server_id: Option<String>,
        tabs: Vec<TabInfo>,
    },
    UpdateTerminalLayout {
        tab_id: TabId,
        layout: vterm_common::TerminalLayout,
    },
    MoveSessionToTab {
        server_id: Option<String>,
        session_id: SessionId,
        tab_id: TabId,
    },
    ListTerminalProfiles,
    TerminalProfileList {
        profiles: Vec<TerminalProfile>,
    },
    ListPathContent {
        session_id: SessionId,
        path: String,
    },
    SetFsWatchers {
        session_id: SessionId,
        watches: FsWatchConfig,
    },
    PathContentList {
        session_id: SessionId,
        path: String,
        items: Vec<FileListItem>,
        #[serde(default)]
        listing_generation: u64,
    },
    FindFiles {
        session_id: SessionId,
        #[serde(default)]
        request_id: String,
        /// Explicit directory scope for content search. Empty preserves the
        /// legacy behavior of searching the session's viewing/working path.
        #[serde(default)]
        root_path: String,
        query: String,
        #[serde(default)]
        include: String,
        #[serde(default)]
        exclude: String,
        #[serde(default)]
        case_sensitive: bool,
        #[serde(default)]
        whole_word: bool,
        #[serde(default)]
        use_regex: bool,
        #[serde(default = "default_find_files_max_results")]
        max_results: usize,
    },
    FindFilesResult {
        session_id: SessionId,
        #[serde(default)]
        request_id: String,
        query: String,
        #[serde(default)]
        chunk_index: u32,
        #[serde(default)]
        is_final: bool,
        results: Vec<FileSearchResult>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Recursively search the entry *names* under `root_path` for a
    /// case-insensitive substring match (file or folder names). Unlike
    /// `FindFiles` (which greps file contents), this never reads file bodies —
    /// it's a cheap directory walk. Used by the tree-mode search bar.
    SearchPathNames {
        session_id: SessionId,
        request_id: String,
        root_path: String,
        query: String,
        #[serde(default = "default_search_path_names_max_results")]
        max_results: usize,
    },
    SearchPathNamesResult {
        session_id: SessionId,
        request_id: String,
        root_path: String,
        query: String,
        #[serde(default)]
        is_final: bool,
        results: Vec<PathNameMatch>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    RecentProjectDirectoryList {
        server_id: Option<String>,
        directories: Vec<RecentProjectDirectory>,
    },
    KnownAppList {
        server_id: Option<String>,
        apps: Vec<KnownAppInfo>,
    },
    RemoveRecentProjectDirectory {
        server_id: Option<String>,
        path: String,
    },
    TerminalCommandHistoryList {
        server_id: Option<String>,
        commands: Vec<TerminalCommandHistory>,
    },
    ListAiConversations {
        session_id: SessionId,
    },
    StopWatchingAiConversation {
        session_id: SessionId,
        conversation_id: String,
    },
    AiConversationList {
        session_id: SessionId,
        conversations: Vec<AiConversationInfo>,
    },
    ReadAiConversation {
        session_id: SessionId,
        conversation_id: String,
    },
    AiConversationMessages {
        session_id: SessionId,
        conversation_id: String,
        messages: Vec<AiConversationMessage>,
    },
    AiConversationMessagesUpdate {
        session_id: SessionId,
        conversation_id: String,
        start_index: usize,
        messages: Vec<AiConversationMessage>,
    },
    TranslateMessage {
        window_id: u64,
        component_id: u64,
        session_id: SessionId,
        message_id: String,
        text: String,
        /// File the text was selected from, for context (lets the translator
        /// recognize code-comment syntax). `None` for prose/markdown sources
        /// like chat messages or commit messages.
        #[serde(default)]
        source_file: Option<String>,
    },
    TranslateMessageResult {
        window_id: u64,
        component_id: u64,
        session_id: SessionId,
        message_id: String,
        translation: Option<String>,
        error: Option<String>,
    },
    GitChangedFiles {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        commit: Option<String>,
    },
    GitChangedFilesList {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        commit: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        repo_root: Option<String>,
        items: Vec<GitChangedFile>,
    },
    GitLog {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        commit: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        branch: Option<String>,
    },
    GitLogList {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        commit: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        branch: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        repo_root: Option<String>,
        commits: Vec<GitCommitInfo>,
    },
    GitDiff {
        session_id: SessionId,
        path: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        commit: Option<String>,
        /// When true, show staged changes (--cached) instead of working tree changes.
        #[serde(default)]
        cached: bool,
        /// When true, compare the requested revision directly against the working tree.
        #[serde(default)]
        compare_with_worktree: bool,
    },
    GitDiffResult {
        session_id: SessionId,
        path: String,
        /// Raw git diff output bytes. The client decodes these with the
        /// encoding selected by the user (default: lossy UTF-8) so a wrong
        /// encoding can be fixed locally without re-requesting.
        diff: Vec<u8>,
    },
    // Git write operations (client → server)
    GitStage {
        session_id: SessionId,
        paths: Vec<String>,
        /// Repo root the path list was generated from (from the last
        /// `GitChangedFilesList`); pins the operation to that repository so a
        /// session working path that moved into a nested repo in the meantime
        /// cannot redirect the pathspecs. `None` = resolve from the session's
        /// current working path (legacy behaviour).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        repo_root: Option<String>,
    },
    GitUnstage {
        session_id: SessionId,
        paths: Vec<String>,
        /// See `GitStage::repo_root`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        repo_root: Option<String>,
    },
    GitDiscard {
        session_id: SessionId,
        paths: Vec<String>,
        /// See `GitStage::repo_root`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        repo_root: Option<String>,
    },
    GitCommit {
        session_id: SessionId,
        message: String,
        /// When present, require the repository to still be detached at this
        /// commit before applying the operation. A short hash is accepted.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expected_detached_head: Option<String>,
        /// Existing local branch to check out, or the name of the branch to
        /// create, before committing. `None` commits on the detached HEAD.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        target_branch: Option<String>,
        /// Create and check out `target_branch` before committing. False means
        /// check out an existing local branch.
        #[serde(default)]
        create_branch: bool,
    },
    GitFetch {
        session_id: SessionId,
    },
    GitPull {
        session_id: SessionId,
    },
    GitPush {
        session_id: SessionId,
        /// `git push --all`.
        #[serde(default)]
        push_all: bool,
        /// `git push --force`.
        #[serde(default)]
        force: bool,
        /// `git push --force-with-lease`.
        #[serde(default)]
        force_with_lease: bool,
        /// `git push -u` (set upstream).
        #[serde(default)]
        set_upstream: bool,
        /// `git push --tags`.
        #[serde(default)]
        tags: bool,
        /// `git push --dry-run`.
        #[serde(default)]
        dry_run: bool,
        /// Remote to push to (e.g. `origin`). Empty falls back to git's
        /// configured default remote.
        #[serde(default)]
        remote: String,
        /// Branch (refspec) to push. Empty pushes HEAD / falls back to
        /// `--all`/default behaviour.
        #[serde(default)]
        branch: String,
    },
    GitRevert {
        session_id: SessionId,
        path: String,
        commit: String,
    },
    GitRevertResult {
        session_id: SessionId,
        path: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Request the server to generate a commit message using the agent.
    GitGenerateCommitMessageAgent {
        session_id: SessionId,
    },
    /// Server acknowledgement that commit-message generation was accepted and
    /// detached from the short-lived request connection.
    GitGenerateCommitMessageAccepted {
        session_id: SessionId,
    },
    /// Request the server to analyze a diff using the code analysis LLM.
    GitAnalyzeDiff {
        session_id: SessionId,
        path: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        commit: Option<String>,
        #[serde(default)]
        cached: bool,
        /// When true, compare the requested revision directly against the working tree.
        #[serde(default)]
        compare_with_worktree: bool,
    },

    // Git write operation responses (server → client)
    GitStageResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    GitUnstageResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    GitDiscardResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    GitCommitResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    GitFetchResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    GitPullResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    GitPushResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Server response with LLM-generated commit message.
    GitGenerateCommitMessageResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Server response with LLM diff analysis.
    GitAnalyzeDiffResult {
        session_id: SessionId,
        #[serde(default)]
        hunks: Vec<vterm_common::HunkAnalysis>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(default)]
        done: bool,
    },

    // Git stash operations
    GitStashList {
        session_id: SessionId,
    },
    GitStashListResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        repo_root: Option<String>,
        stashes: Vec<GitStashEntry>,
    },
    /// `git stash push --include-untracked`. `paths` empty stashes all
    /// changes; otherwise only the given pathspecs are stashed.
    GitStashPush {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        paths: Vec<String>,
    },
    GitStashApply {
        session_id: SessionId,
        index: u32,
    },
    GitStashPop {
        session_id: SessionId,
        index: u32,
    },
    GitStashDrop {
        session_id: SessionId,
        index: u32,
    },
    /// Shared response for GitStashPush / GitStashApply / GitStashPop /
    /// GitStashDrop.
    GitStashResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },

    // Git submodule operations
    GitSubmoduleList {
        session_id: SessionId,
    },
    GitSubmoduleListResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        repo_root: Option<String>,
        submodules: Vec<GitSubmoduleInfo>,
    },

    // Git branch operations
    GitBranches {
        session_id: SessionId,
    },
    GitBranchesResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        current: Option<String>,
        /// Full HEAD hash when the repository is in detached-HEAD state. The
        /// client shortens it for display while retaining the full value for
        /// stale-state validation.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detached_head: Option<String>,
        branches: Vec<GitBranchInfo>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        remote_tracking_hash: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tags: Vec<String>,
    },
    GitCheckout {
        session_id: SessionId,
        /// The ref to check out (e.g. `origin/feature/x`, `v1.2.0`,
        /// `remotes/origin/main`). Serialized as `ref` for clarity; accepts
        /// the legacy `branch` key from older clients.
        #[serde(default, alias = "branch", rename = "ref")]
        ref_: String,
        /// When true, create/reset a local branch named `branch_name` with
        /// `-B` before checking the ref out.
        #[serde(default)]
        create_branch: bool,
        /// Local branch name to use with `-B` (only meaningful when
        /// `create_branch` is true).
        #[serde(default)]
        branch_name: String,
        /// Pass `--track` so the new branch tracks the remote as upstream.
        #[serde(default)]
        track: bool,
        /// Pass `-f` (force).
        #[serde(default)]
        force: bool,
        /// Pass `--merge` (merge the ref into the current branch).
        #[serde(default)]
        merge: bool,
    },
    GitCheckoutResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    GitCreateBranch {
        session_id: SessionId,
        source_ref: String,
        branch: String,
    },
    GitCreateBranchResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    GitCreateWorktree {
        session_id: SessionId,
        branch: String,
        name: String,
    },
    GitCreateWorktreeResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        worktree_path: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        new_session_id: Option<SessionId>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    GitDeleteBranch {
        session_id: SessionId,
        branch: String,
        is_worktree: bool,
    },
    GitDeleteBranchResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    GitMerge {
        session_id: SessionId,
        branch: String,
    },
    GitMergeResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },

    // Git rebase operations (interactive rebase flow)
    /// Start: compute the todo list (commits unique to the current branch).
    GitRebaseStart {
        session_id: SessionId,
        target_branch: String,
    },
    GitRebaseStartResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_branch: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        commits: Vec<GitRebaseTodoCommit>,
    },
    /// Apply the (edited) todo list: runs `git rebase -i` with a
    /// GIT_SEQUENCE_EDITOR that writes the todo file.
    GitRebaseApply {
        session_id: SessionId,
        target_branch: String,
        commits: Vec<GitRebaseTodoCommit>,
    },
    GitRebaseApplyResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        state: Option<GitRebaseState>,
    },
    /// Three-way content of one conflicted file (index stages 1/2/3).
    GitRebaseConflictContent {
        session_id: SessionId,
        path: String,
    },
    GitRebaseConflictContentResult {
        session_id: SessionId,
        path: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ours: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        theirs: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        /// Commit metadata of the ours/theirs sides (HEAD and
        /// REBASE_HEAD/CHERRY_PICK_HEAD/MERGE_HEAD) for hover tips.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ours_commit: Option<GitCommitMeta>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        theirs_commit: Option<GitCommitMeta>,
    },
    /// Write the resolved content of one file and `git add` it.
    GitRebaseResolve {
        session_id: SessionId,
        path: String,
        content: String,
    },
    GitRebaseResolveResult {
        session_id: SessionId,
        path: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    GitRebaseContinue {
        session_id: SessionId,
    },
    GitRebaseContinueResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        state: Option<GitRebaseState>,
    },
    GitRebaseSkip {
        session_id: SessionId,
    },
    GitRebaseSkipResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        state: Option<GitRebaseState>,
    },
    GitRebaseAbort {
        session_id: SessionId,
    },
    GitRebaseAbortResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Start an AI conflict-resolution run over the given unresolved rebase
    /// hunks. Progress is streamed back as `GitRebaseAiResolveResult`
    /// messages (one per hunk state change; the final one has `done = true`).
    GitRebaseAiResolve {
        session_id: SessionId,
        /// Client-generated run identifier, echoed back in every
        /// `GitRebaseAiResolveResult` so the client can drop stale events
        /// from a superseded run.
        run_id: u32,
        /// Free-text user instruction guiding the resolution (may be empty).
        #[serde(default)]
        instruction: String,
        /// Display labels of the two sides (during a rebase, ours = the
        /// target branch, theirs = the incoming commit) so the model can
        /// reason about intent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ours_label: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        theirs_label: Option<String>,
        targets: Vec<GitRebaseAiTarget>,
    },
    /// Stop the session's in-flight AI resolve run (if any). The run stops
    /// after the hunk currently being analyzed; resolutions already streamed
    /// stay valid. Fire-and-forget: there is no result message.
    GitRebaseAiResolveCancel {
        session_id: SessionId,
    },
    /// One streamed event of an AI resolve run. `hunk` carries per-hunk
    /// progress; a run-level fatal error (e.g. LLM not configured) sets
    /// `error`. The last message of a run has `done = true`.
    GitRebaseAiResolveResult {
        session_id: SessionId,
        /// Echo of the run identifier from `GitRebaseAiResolve`.
        #[serde(default)]
        run_id: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        hunk: Option<GitRebaseAiHunkEvent>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(default)]
        done: bool,
    },

    FetchFile {
        session_id: SessionId,
        path: String,
        #[serde(default = "default_max_file_size")]
        max_size: u64,
    },
    FileContent {
        session_id: SessionId,
        path: String,
        #[serde(default)]
        data: Vec<u8>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        diff_ranges: Vec<FileDiffRange>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    SaveFile {
        session_id: SessionId,
        path: String,
        data: Vec<u8>,
    },
    SaveFileResult {
        session_id: SessionId,
        path: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Upload one attachment into a server-managed temporary directory.
    /// The request body contains the file's unframed bytes.
    TempFileUploadStart {
        task_id: String,
        session_id: SessionId,
        file_name: String,
        content_type: String,
        usage: TempFileUsage,
    },
    TempFileUploadResult {
        task_id: String,
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        bytes: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Upload one or more local items (files and/or folders) into a remote
    /// directory as ONE transfer task. The request body is a stream of
    /// framed entries (see [`crate::folder_stream`]): the client walks the
    /// local items lazily while streaming; the server creates parent
    /// directories from each entry's `rel_path` ('/'-separated, first
    /// component is the item's own name) and writes files incrementally.
    /// The reply is `UploadResult`.
    UploadStart {
        task_id: String,
        session_id: SessionId,
        dst_dir: String,
    },
    /// Outcome of [`Message::UploadStart`].
    UploadResult {
        task_id: String,
        session_id: SessionId,
        files: u64,
        bytes: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Download one or more remote items (files and/or folders) as ONE
    /// transfer task. The reply body is the same framed entry stream,
    /// produced by a lazy server-side walk (each item enters the stream
    /// under its own name); the client creates parent directories and
    /// writes files incrementally as entries arrive.
    DownloadStart {
        task_id: String,
        session_id: SessionId,
        paths: Vec<String>,
    },
    /// Outcome of [`Message::DownloadStart`] (response metadata; the
    /// body carried the framed entry stream).
    DownloadResult {
        task_id: String,
        session_id: SessionId,
        files: u64,
        bytes: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    CreatePath {
        session_id: SessionId,
        path: String,
        is_dir: bool,
    },
    CreatePathResult {
        session_id: SessionId,
        path: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    DeletePaths {
        session_id: SessionId,
        paths: Vec<String>,
    },
    DeletePathsResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    PastePaths {
        session_id: SessionId,
        src_paths: Vec<String>,
        dst_dir: String,
        is_cut: bool,
    },
    PastePathsResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    RenamePath {
        session_id: SessionId,
        old_path: String,
        new_path: String,
    },
    RenamePathResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Generate a starter configuration in the given absolute `.vterm` directory.
    GenerateVtermConfig {
        /// Session whose project is being configured. Only used for
        /// client-side relay routing (see `Message::session_id`), so the
        /// request reaches the server owning the session; the handler works
        /// purely off `vterm_path`. Optional for wire compatibility with
        /// older clients, which fall back to the directly connected server.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        session_id: Option<SessionId>,
        vterm_path: String,
        config_type: String,
    },
    GenerateVtermConfigResult {
        /// Echo of the request's routing key (see the rule in
        /// `docs/message-routing.md`); absent when the request omitted it or
        /// came from an older client.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        session_id: Option<SessionId>,
        vterm_path: String,
        config_type: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    StartStaticHttpServer {
        session_id: SessionId,
        path: String,
        /// The client needs a remotely reachable port. New servers create the
        /// forward before responding; older servers ignore this field and use
        /// the client's legacy `RequestReverseProxy` fallback instead.
        #[serde(default)]
        request_port_forward: bool,
    },
    StartStaticHttpServerResult {
        session_id: SessionId,
        path: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        port: Option<u16>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        share_token: Option<String>,
        /// `port` is already the final shared/remote forward port. The default
        /// preserves compatibility with results from older servers.
        #[serde(default)]
        port_forwarded: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Shared server asks the relay to coordinate a two-stage port forward.
    /// The relay asks the shared server to expose `target_port`, then creates
    /// its own forward to the resulting shared-server proxy port.
    RequestSharedPortForward {
        session_id: SessionId,
        target_port: u16,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        auth_token: Option<String>,
    },
    /// Relay responds with its final proxy port for clients to connect to.
    SharedPortForwardReady {
        session_id: SessionId,
        target_port: u16,
        relay_port: u16,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    DebugLaunchList {
        session_id: SessionId,
    },
    DebugLaunchListResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        repo_root: Option<String>,
        #[serde(default)]
        launches: Vec<DebugLaunchConfig>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    DebugStart {
        session_id: SessionId,
        launch_name: String,
    },
    DebugStartResult {
        session_id: SessionId,
        launch_name: String,
        success: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    DebugOutput {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        launch_name: Option<String>,
        stream: String,
        data: String,
    },
    DebugDap {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        launch_name: Option<String>,
        message: serde_json::Value,
    },
    QuickCommandList {
        session_id: SessionId,
    },
    QuickCommandListResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        repo_root: Option<String>,
        #[serde(default)]
        commands: Vec<QuickCommand>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    RunQuickCommand {
        session_id: SessionId,
        name: String,
    },
    RunQuickCommandResult {
        session_id: SessionId,
        name: String,
        success: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    ListRelayServers,
    RelayRegisterServer {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        request_port: Option<u16>,
    },
    RelayServerRegistered {
        server_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        terminal_tunnel_key: Option<String>,
    },
    RelayServerList {
        servers: Vec<RelayServerInfo>,
    },
    /// Shared server → relay (RelayPrimary channel): allocate a UDP
    /// pairing toward the screen-server's `target_port` on the shared
    /// server's host. `request_id` correlates the response.
    RelayAllocateUdpPairing {
        request_id: u64,
        target_port: u16,
    },
    /// Relay → shared server: pairing allocated; clients send media UDP to
    /// `host:port` and the relay forwards it to the pairing's target.
    RelayUdpPairingAllocated {
        request_id: u64,
        pairing: String,
        host: String,
        port: u16,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Shared server → relay: release a pairing (stream stopped).
    RelayReleaseUdpPairing {
        pairing: String,
    },

    // ── Screen-stream relay coordination (RelayPrimary channel) ──────────
    // The relay (localhost server) is the sole coordinator: it allocates
    // the UDP port and asks the shared server only for a token + the
    // screen-server's UDP port. No transparent tunnel needed for screen.
    //
    /// Relay → shared server: request a screen-stream token + the
    /// screen-server's UDP port for one source.
    RelayScreenTokenRequest {
        request_id: u64,
        source: String,
        client_label: String,
    },
    /// Shared server → relay: the token + screen-server UDP port (or error).
    RelayScreenTokenResponse {
        request_id: u64,
        token: String,
        screen_udp_port: u16,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Relay → shared server: revoke a token (client closed its stream).
    RelayScreenTokenRevoke {
        token: String,
    },
    /// Relay → shared server: list streamable sources.
    RelayScreenSourcesRequest {
        request_id: u64,
    },
    /// Shared server → relay: source list (or error).
    RelayScreenSourcesResponse {
        request_id: u64,
        #[serde(default)]
        sources: Vec<ScreenSourceInfo>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        revision: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    SessionBufferInfo {
        session_id: SessionId,
        start_offset: u64,
        end_offset: u64,
    },
    FetchHistoryRaw {
        session_id: SessionId,
        before_offset: u64,
        max_bytes: u32,
    },
    /// Debug-only request to dump the retained raw history buffer to disk.
    ExportSessionRawHistory {
        session_id: SessionId,
    },
    ExportSessionRawHistoryResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    HistoryRawChunk {
        session_id: SessionId,
        start_offset: u64,
        end_offset: u64,
        data: Vec<u8>,
    },
    /// Server-rendered snapshot of the session's visible screen. Replaying
    /// `data` on a fresh emulator of the session size reproduces the server
    /// grid state exactly at stream offset `base_offset`. Sent to a viewer
    /// on attach when the retained raw history window no longer covers a
    /// full repaint (replaying a trimmed window would leave the screen
    /// partially blank for full-screen TUI apps).
    HistorySnapshot {
        session_id: SessionId,
        base_offset: u64,
        data: Vec<u8>,
    },
    /// Close a session
    CloseSession {
        session_id: SessionId,
    },
    /// Close a whole tab and terminate all terminal sessions in it.
    CloseTab {
        tab_id: TabId,
    },
    RenameSession {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        session_name: Option<String>,
    },
    /// Session closed notification
    SessionClosed {
        session_id: SessionId,
    },
    /// Detach a session from its split group (set group_id to 0)
    DetachSession {
        session_id: SessionId,
    },
    ApplyEnvironment {
        session_id: SessionId,
        environment_id: String,
        enabled: bool,
    },
    ApplyEnvironmentResult {
        session_id: SessionId,
        environment_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    SetNotificationEnabled {
        session_id: SessionId,
        enabled: bool,
    },
    SetNotificationEnabledResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    SetAutoConfirmEnabled {
        session_id: SessionId,
        enabled: bool,
    },
    SetAutoConfirmEnabledResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Change the byte-stream decoder for a UART session.
    SetSessionEncoding {
        session_id: SessionId,
        encoding: String,
    },
    SetSessionEncodingResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Report the viewer's effective terminal default colors so the
    /// server-side emulator can answer OSC 10/11 color probes (used by
    /// OpenCode and similar TUIs to pick a light or dark theme). Sent on an
    /// explicit theme change; clears old application OSC overrides so the
    /// selected Host theme takes effect immediately. Fire-and-forget.
    SetTerminalThemeColors {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        background: Option<[u8; 3]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        foreground: Option<[u8; 3]>,
    },
    /// Clear a UART session's server-side screen: drop the retained
    /// raw-history window and reset the emulator, so attaching viewers
    /// (e.g. after a tab switch) no longer receive the pre-clear buffer.
    /// Stream offsets keep advancing monotonically.
    ClearSessionScreen {
        session_id: SessionId,
    },
    ClearSessionScreenResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },

    // Terminal I/O
    /// PTY output data (to display)
    Output {
        session_id: SessionId,
        start_offset: u64,
        data: Vec<u8>,
    },
    /// Keyboard input (to PTY)
    Input {
        session_id: SessionId,
        data: Vec<u8>,
        cursor_from_end: Option<CursorFromEnd>,
    },
    /// Terminal resize event
    Resize {
        session_id: SessionId,
        size: TerminalSize,
    },
    /// User-initiated terminal confirm (send `\r` to the session PTY).
    ///
    /// Routed over the primary connection (not the per-session terminal stream)
    /// so it works even when the target session is not the active tab and thus
    /// has no live terminal connection. The server writes `\r` to the PTY via
    /// the normal input path, which also resolves any pending attention prompt.
    ConfirmSession {
        session_id: SessionId,
    },
    /// Ask the server to type a ready-to-run `mcp add` command for the given
    /// agent app executable into the session PTY. The server builds the
    /// command from its own MCP config so the MCP endpoint/token never reach
    /// clients. No trailing newline is written — the user presses Enter.
    SetupMcpApp {
        session_id: SessionId,
        executable: String,
    },
    /// Result of a `SetupMcpApp` request: the setup line was typed into the
    /// session PTY, or `error` explains why it could not be. Sent on the
    /// request connection so it completes cleanly instead of surfacing as
    /// "Request handler for SetupMcpApp completed without a response".
    SetupMcpAppResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Result of a user-initiated terminal confirm (`\r` write).
    ConfirmSessionResult {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    // Streaming control
    /// Start streaming a session
    StreamStart {
        session_id: SessionId,
        /// The viewer's grid size at attach time. Lets the server seed the
        /// new viewer with the size it will actually render at, instead of
        /// the session's (possibly stale) PTY size until the first Resize
        /// arrives.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        initial_size: Option<TerminalSize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        background: Option<[u8; 3]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        foreground: Option<[u8; 3]>,
    },
    /// End streaming a session
    StreamEnd {
        session_id: SessionId,
    },
    /// Viewer joined notification (sent to host)
    ViewerJoined {
        session_id: SessionId,
        viewer_count: u32,
    },
    /// Viewer left notification (sent to host)
    ViewerLeft {
        session_id: SessionId,
        viewer_count: u32,
    },

    // Screen streaming (coordinated via vterm-screen-server)
    /// List the host's streamable sources (displays, iOS simulators,
    /// Android devices). Host-scoped, not tied to a terminal session.
    ListScreenSources {
        /// When set to a shared server's id, the request is routed through
        /// the relay to that server (see `relay_route_server_id`). `None`
        /// means the directly connected (local) server.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
    },
    /// Answer to [`ListScreenSources`].
    ScreenSourcesResult {
        #[serde(default)]
        sources: Vec<ScreenSourceInfo>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        /// Server the answer came from: `None` = the directly connected local
        /// server, `Some(id)` = a relayed shared server. Lets the client cache
        /// per-server source lists instead of letting concurrent answers
        /// (local + shared) race over a single global list.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        revision: Option<u64>,
    },
    ScreenSourcesChanged {
        #[serde(default)]
        sources: Vec<ScreenSourceInfo>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
        revision: u64,
    },
    /// Start streaming one source to this client. The server coordinates
    /// with the screen-server and answers with [`ScreenStreamReady`].
    StartScreenStream {
        /// Source spec (`main`, `display:N`, `sim:UDID`, `android:SERIAL`).
        source: String,
        /// When set to a shared server's id, the request is routed through
        /// the relay to that server (see `relay_route_server_id`). `None`
        /// means the directly connected (local) server.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
    },
    /// Answer to [`StartScreenStream`]: the UDP endpoint + one-time token
    /// the client uses for its media handshake. `error` set ⇒ the other
    /// fields are meaningless.
    ScreenStreamReady {
        source: String,
        /// Server that accepted the corresponding start request. Echoed so
        /// follow-up source refreshes stay on the same relay route.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
        #[serde(flatten)]
        endpoint: StreamEndpoint,
        /// 16-byte token, 32 lowercase hex chars.
        #[serde(default)]
        token: String,
        /// `h264` / `hevc`.
        #[serde(default = "default_screen_codec")]
        codec: String,
        /// Encoder input pixel format: `nv12` / `p010`.
        #[serde(default = "default_screen_format")]
        format: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Stop this client's stream of one source.
    StopScreenStream {
        source: String,
        /// When set to a shared server's id, the request is routed through
        /// the relay to that server (see `relay_route_server_id`). `None`
        /// means the directly connected (local) server.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
    },
    /// Ack for [`StopScreenStream`] (also the terminal state after an
    /// error-side teardown).
    ScreenStreamStopped {
        source: String,
        /// Server that accepted the corresponding stop request.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Server → client push about an active stream (source unplugged,
    /// pipeline died). Terminal for the stream unless stated otherwise.
    ScreenStreamNotify {
        source: String,
        #[serde(flatten)]
        event: ScreenStreamEvent,
    },
    /// Server → client: 5-second pipeline stats rollup for screen streams
    /// (capture/encode/network stage latencies).
    ScreenStreamStats {
        /// Rolling-window stats, all stages averaged.
        frames: u64,
        /// Mean capture→encode pickup, microseconds.
        avg_pickup_us: u64,
        /// Mean encode time, milliseconds.
        avg_encode_ms: f64,
        delivered: u64,
        dropped: u64,
        /// Mean network delivery (send → full ACK), milliseconds.
        avg_net_ms: f64,
    },
    /// Power on an iOS simulator on the screen-server host (`sim:UDID` spec).
    BootSimulator {
        spec: String,
        /// Route the operation to this shared server. `None` means the
        /// directly connected (local) server.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
    },
    /// Answer to [`Message::BootSimulator`].
    BootSimulatorResult {
        spec: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Power off an iOS simulator on the screen-server host (`sim:UDID` spec).
    ShutdownSimulator {
        spec: String,
        /// Route the operation to this shared server. `None` means the
        /// directly connected (local) server.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
    },
    /// Answer to [`Message::ShutdownSimulator`].
    ShutdownSimulatorResult {
        spec: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Copy the iOS simulator's pasteboard (device copy relay; `sim:UDID`
    /// spec). The answer carries the text.
    CopySimulatorClipboard {
        spec: String,
    },
    /// Answer to [`Message::CopySimulatorClipboard`].
    CopySimulatorClipboardResult {
        spec: String,
        #[serde(default)]
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Launch an Android emulator on the screen-server host (`avd:NAME`
    /// spec; any host, unlike the iOS Simulator control).
    BootAndroidEmulator {
        avd: String,
        /// Route the operation to this shared server. `None` means the
        /// directly connected (local) server.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
    },
    /// The operation was accepted; completion arrives as AndroidEmulatorBooted
    /// or AndroidEmulatorOperationFailed.
    BootAndroidEmulatorResult {
        avd: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        operation_id: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    AndroidEmulatorBooted {
        avd: String,
        serial: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        operation_id: Option<u64>,
    },
    AndroidEmulatorShutdown {
        serial: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        operation_id: Option<u64>,
    },
    AndroidEmulatorOperationFailed {
        operation: AndroidOperationKind,
        target: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        operation_id: Option<u64>,
        error: String,
    },
    /// Power off an Android emulator on the screen-server host
    /// (`android:SERIAL` spec; `adb emu kill`, emulator serials only).
    ShutdownAndroidEmulator {
        serial: String,
        /// Route the operation to this shared server. `None` means the
        /// directly connected (local) server.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
    },
    /// The operation was accepted; completion arrives as AndroidEmulatorShutdown
    /// or AndroidEmulatorOperationFailed.
    ShutdownAndroidEmulatorResult {
        serial: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        operation_id: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Rotate an Android device/emulator display by 90° (cycles the
    /// `user_rotation` setting; the stream re-negotiates the new geometry).
    RotateAndroidDevice {
        serial: String,
    },
    /// Answer to [`Message::RotateAndroidDevice`].
    RotateAndroidDeviceResult {
        serial: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },

    // Reverse proxy
    RequestReverseProxy {
        target_port: u16,
        auth_token: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        purpose: Option<ReverseProxyPurpose>,
    },
    ReverseProxyReady {
        target_port: u16,
        proxy_port: u16,
        reused_existing: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        settings_ticket: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    ListReverseProxies {
        auth_token: Option<String>,
    },
    ReverseProxyList {
        proxies: Vec<PortForwardInfo>,
    },

    // Terminal observer events
    TerminalAttentionRequired {
        attention: TerminalAttention,
    },
    TerminalAttentionResolved {
        session_id: SessionId,
    },
    TerminalTaskCompleted {
        completion: TerminalCompletion,
    },

    // Voice transcription
    /// Transcribe audio (f32 LE bytes, 16kHz mono, optionally gzip-compressed)
    TranscribeAudio {
        window_id: u64,
        event_id: String,
        /// Terminal context associated with this voice-input transcription.
        context: VoiceInputOptimizationContext,
        data: ByteData,
        #[serde(default)]
        compressed: bool,
    },
    /// Transcription result
    TranscribeAudioResult {
        window_id: u64,
        event_id: String,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        /// Server-measured gzip decompression time in milliseconds.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        decompress_ms: Option<f64>,
        /// Server-measured transcription (ASR) time in milliseconds.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        transcribe_ms: Option<f64>,
    },
    /// Polish transcribed voice-input text using contextual information.
    PolishVoiceInputText {
        window_id: u64,
        component_id: u64,
        request_id: String,
        text: String,
        context: VoiceInputOptimizationContext,
    },
    /// Result of polishing voice-input text. `text` is absent when `error` is set.
    PolishVoiceInputTextResult {
        window_id: u64,
        component_id: u64,
        session_id: SessionId,
        request_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },

    // Heartbeat
    /// Keep-alive ping
    Ping,
    /// Keep-alive pong
    Pong,

    // Errors
    /// Error response
    Error {
        code: ErrorCode,
        message: String,
    },
}

/// Source kind for screen streaming (mirrors the ctl IPC schema).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScreenSourceKind {
    Screen,
    Window,
    IosSimulator,
    AndroidDevice,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AndroidOperationKind {
    Boot,
    Shutdown,
}

/// Live status of a streamable source.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ScreenSourceStatus {
    Available,
    Busy,
    Streaming,
    Unavailable { reason: String },
}

/// One streamable source (see [`Message::ScreenSourcesResult`]).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScreenSourceInfo {
    pub spec: String,
    pub kind: ScreenSourceKind,
    pub title: String,
    #[serde(flatten)]
    pub status: ScreenSourceStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default)]
    pub primary: bool,
}

/// Where the client should direct its UDP media connection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "endpoint", rename_all = "snake_case")]
pub enum StreamEndpoint {
    /// Direct UDP to the screen-server. `host` may be empty, meaning
    /// "the same host as this control connection".
    Direct { host: String, port: u16 },
    /// Through a relay: clients send media UDP to the relay's allocated
    /// pairing port; `pairing` is an opaque id for logging/teardown.
    Relay {
        host: String,
        port: u16,
        pairing: String,
    },
}

/// Server → client stream event payload (see [`Message::ScreenStreamNotify`]).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ScreenStreamEvent {
    /// The pipeline died; the stream is gone.
    StreamDied { message: String },
    /// The source itself disappeared (device unplugged, simulator shut down).
    SourceUnavailable { reason: String },
}

pub fn default_max_file_size() -> u64 {
    1024 * 1024 // 1 MB
}

fn default_screen_codec() -> String {
    "h264".to_string()
}

fn default_screen_format() -> String {
    "nv12".to_string()
}

pub fn default_find_files_max_results() -> usize {
    10_000
}

pub fn default_search_path_names_max_results() -> usize {
    1000
}

#[derive(Debug, Default, Clone)]
pub struct TransportBinaryFrameDecoder {
    pending: Option<PendingBinaryFrame>,
}

#[derive(Debug, Clone)]
struct PendingBinaryFrame {
    total_len: usize,
    received_len: usize,
    data: Vec<u8>,
}

impl Message {
    /// Returns the variant name of this message (e.g. "Hello", "Output", "CreateSession").
    pub fn msg_name(&self) -> &'static str {
        match self {
            Message::Hello { .. } => "Hello",
            Message::AuthChallenge { .. } => "AuthChallenge",
            Message::AuthResponse { .. } => "AuthResponse",
            Message::HelloAck { .. } => "HelloAck",
            Message::ServerInfo { .. } => "ServerInfo",
            Message::CreateSession { .. } => "CreateSession",
            Message::SessionCreated { .. } => "SessionCreated",
            Message::SplitTerminal { .. } => "SplitTerminal",
            Message::ListSessions => "ListSessions",
            Message::SessionList { .. } => "SessionList",
            Message::ListTabGroups => "ListTabGroups",
            Message::TabGroupList { .. } => "TabGroupList",
            Message::CreateTabGroup { .. } => "CreateTabGroup",
            Message::DeleteTabGroup { .. } => "DeleteTabGroup",
            Message::RenameTabGroup { .. } => "RenameTabGroup",
            Message::MoveTabToGroup { .. } => "MoveTabToGroup",
            Message::GetTabInfoList { .. } => "GetTabInfoList",
            Message::TabInfoList { .. } => "TabInfoList",
            Message::UpdateTerminalLayout { .. } => "UpdateTerminalLayout",
            Message::MoveSessionToTab { .. } => "MoveSessionToTab",
            Message::ListTerminalProfiles => "ListTerminalProfiles",
            Message::TerminalProfileList { .. } => "TerminalProfileList",
            Message::CloseSession { .. } => "CloseSession",
            Message::CloseTab { .. } => "CloseTab",
            Message::RenameSession { .. } => "RenameSession",
            Message::SessionClosed { .. } => "SessionClosed",
            Message::DetachSession { .. } => "DetachSession",
            Message::ApplyEnvironment { .. } => "ApplyEnvironment",
            Message::ApplyEnvironmentResult { .. } => "ApplyEnvironmentResult",
            Message::SetNotificationEnabled { .. } => "SetNotificationEnabled",
            Message::SetNotificationEnabledResult { .. } => "SetNotificationEnabledResult",
            Message::SetAutoConfirmEnabled { .. } => "SetAutoConfirmEnabled",
            Message::SetAutoConfirmEnabledResult { .. } => "SetAutoConfirmEnabledResult",
            Message::SetSessionEncoding { .. } => "SetSessionEncoding",
            Message::SetSessionEncodingResult { .. } => "SetSessionEncodingResult",
            Message::SetTerminalThemeColors { .. } => "SetTerminalThemeColors",
            Message::ClearSessionScreen { .. } => "ClearSessionScreen",
            Message::ClearSessionScreenResult { .. } => "ClearSessionScreenResult",
            Message::ListPathContent { .. } => "ListPathContent",
            Message::SetFsWatchers { .. } => "SetFsWatchers",
            Message::PathContentList { .. } => "PathContentList",
            Message::FindFiles { .. } => "FindFiles",
            Message::FindFilesResult { .. } => "FindFilesResult",
            Message::SearchPathNames { .. } => "SearchPathNames",
            Message::SearchPathNamesResult { .. } => "SearchPathNamesResult",
            Message::FetchFile { .. } => "FetchFile",
            Message::FileContent { .. } => "FileContent",
            Message::SaveFile { .. } => "SaveFile",
            Message::SaveFileResult { .. } => "SaveFileResult",
            Message::TempFileUploadStart { .. } => "TempFileUploadStart",
            Message::TempFileUploadResult { .. } => "TempFileUploadResult",
            Message::UploadStart { .. } => "UploadStart",
            Message::UploadResult { .. } => "UploadResult",
            Message::DownloadStart { .. } => "DownloadStart",
            Message::DownloadResult { .. } => "DownloadResult",
            Message::CreatePath { .. } => "CreatePath",
            Message::CreatePathResult { .. } => "CreatePathResult",
            Message::DeletePaths { .. } => "DeletePaths",
            Message::DeletePathsResult { .. } => "DeletePathsResult",
            Message::PastePaths { .. } => "PastePaths",
            Message::PastePathsResult { .. } => "PastePathsResult",
            Message::RenamePath { .. } => "RenamePath",
            Message::RenamePathResult { .. } => "RenamePathResult",
            Message::GenerateVtermConfig { .. } => "GenerateVtermConfig",
            Message::GenerateVtermConfigResult { .. } => "GenerateVtermConfigResult",
            Message::StartStaticHttpServer { .. } => "StartStaticHttpServer",
            Message::StartStaticHttpServerResult { .. } => "StartStaticHttpServerResult",
            Message::RequestSharedPortForward { .. } => "RequestSharedPortForward",
            Message::SharedPortForwardReady { .. } => "SharedPortForwardReady",
            Message::DebugLaunchList { .. } => "DebugLaunchList",
            Message::DebugLaunchListResult { .. } => "DebugLaunchListResult",
            Message::DebugStart { .. } => "DebugStart",
            Message::DebugStartResult { .. } => "DebugStartResult",
            Message::DebugOutput { .. } => "DebugOutput",
            Message::DebugDap { .. } => "DebugDap",
            Message::QuickCommandList { .. } => "QuickCommandList",
            Message::QuickCommandListResult { .. } => "QuickCommandListResult",
            Message::RunQuickCommand { .. } => "RunQuickCommand",
            Message::RunQuickCommandResult { .. } => "RunQuickCommandResult",
            Message::RecentProjectDirectoryList { .. } => "RecentProjectDirectoryList",
            Message::KnownAppList { .. } => "KnownAppList",
            Message::RemoveRecentProjectDirectory { .. } => "RemoveRecentProjectDirectory",
            Message::TerminalCommandHistoryList { .. } => "TerminalCommandHistoryList",
            Message::ListAiConversations { .. } => "ListAiConversations",
            Message::StopWatchingAiConversation { .. } => "StopWatchingAiConversation",
            Message::AiConversationList { .. } => "AiConversationList",
            Message::ReadAiConversation { .. } => "ReadAiConversation",
            Message::AiConversationMessages { .. } => "AiConversationMessages",
            Message::AiConversationMessagesUpdate { .. } => "AiConversationMessagesUpdate",
            Message::TranslateMessage { .. } => "TranslateMessage",
            Message::TranslateMessageResult { .. } => "TranslateMessageResult",
            Message::GitChangedFiles { .. } => "GitChangedFiles",
            Message::GitChangedFilesList { .. } => "GitChangedFilesList",
            Message::GitLog { .. } => "GitLog",
            Message::GitLogList { .. } => "GitLogList",
            Message::GitDiff { .. } => "GitDiff",
            Message::GitDiffResult { .. } => "GitDiffResult",
            Message::GitStage { .. } => "GitStage",
            Message::GitUnstage { .. } => "GitUnstage",
            Message::GitDiscard { .. } => "GitDiscard",
            Message::GitCommit { .. } => "GitCommit",
            Message::GitFetch { .. } => "GitFetch",
            Message::GitPull { .. } => "GitPull",
            Message::GitPush { .. } => "GitPush",
            Message::GitRevert { .. } => "GitRevert",
            Message::GitGenerateCommitMessageAgent { .. } => "GitGenerateCommitMessageAgent",
            Message::GitGenerateCommitMessageAccepted { .. } => "GitGenerateCommitMessageAccepted",
            Message::GitAnalyzeDiff { .. } => "GitAnalyzeDiff",
            Message::GitStageResult { .. } => "GitStageResult",
            Message::GitUnstageResult { .. } => "GitUnstageResult",
            Message::GitDiscardResult { .. } => "GitDiscardResult",
            Message::GitCommitResult { .. } => "GitCommitResult",
            Message::GitFetchResult { .. } => "GitFetchResult",
            Message::GitPullResult { .. } => "GitPullResult",
            Message::GitPushResult { .. } => "GitPushResult",
            Message::GitRevertResult { .. } => "GitRevertResult",
            Message::GitGenerateCommitMessageResult { .. } => "GitGenerateCommitMessageResult",
            Message::GitAnalyzeDiffResult { .. } => "GitAnalyzeDiffResult",
            Message::GitStashList { .. } => "GitStashList",
            Message::GitStashListResult { .. } => "GitStashListResult",
            Message::GitStashPush { .. } => "GitStashPush",
            Message::GitStashApply { .. } => "GitStashApply",
            Message::GitStashPop { .. } => "GitStashPop",
            Message::GitStashDrop { .. } => "GitStashDrop",
            Message::GitStashResult { .. } => "GitStashResult",
            Message::GitSubmoduleList { .. } => "GitSubmoduleList",
            Message::GitSubmoduleListResult { .. } => "GitSubmoduleListResult",
            Message::GitBranches { .. } => "GitBranches",
            Message::GitBranchesResult { .. } => "GitBranchesResult",
            Message::GitCheckout { .. } => "GitCheckout",
            Message::GitCheckoutResult { .. } => "GitCheckoutResult",
            Message::GitCreateBranch { .. } => "GitCreateBranch",
            Message::GitCreateBranchResult { .. } => "GitCreateBranchResult",
            Message::GitCreateWorktree { .. } => "GitCreateWorktree",
            Message::GitCreateWorktreeResult { .. } => "GitCreateWorktreeResult",
            Message::GitDeleteBranch { .. } => "GitDeleteBranch",
            Message::GitDeleteBranchResult { .. } => "GitDeleteBranchResult",
            Message::GitMerge { .. } => "GitMerge",
            Message::GitMergeResult { .. } => "GitMergeResult",
            Message::GitRebaseStart { .. } => "GitRebaseStart",
            Message::GitRebaseStartResult { .. } => "GitRebaseStartResult",
            Message::GitRebaseApply { .. } => "GitRebaseApply",
            Message::GitRebaseApplyResult { .. } => "GitRebaseApplyResult",
            Message::GitRebaseConflictContent { .. } => "GitRebaseConflictContent",
            Message::GitRebaseConflictContentResult { .. } => "GitRebaseConflictContentResult",
            Message::GitRebaseResolve { .. } => "GitRebaseResolve",
            Message::GitRebaseResolveResult { .. } => "GitRebaseResolveResult",
            Message::GitRebaseContinue { .. } => "GitRebaseContinue",
            Message::GitRebaseContinueResult { .. } => "GitRebaseContinueResult",
            Message::GitRebaseSkip { .. } => "GitRebaseSkip",
            Message::GitRebaseSkipResult { .. } => "GitRebaseSkipResult",
            Message::GitRebaseAbort { .. } => "GitRebaseAbort",
            Message::GitRebaseAbortResult { .. } => "GitRebaseAbortResult",
            Message::GitRebaseAiResolve { .. } => "GitRebaseAiResolve",
            Message::GitRebaseAiResolveCancel { .. } => "GitRebaseAiResolveCancel",
            Message::GitRebaseAiResolveResult { .. } => "GitRebaseAiResolveResult",
            Message::Output { .. } => "Output",
            Message::Input { .. } => "Input",
            Message::Resize { .. } => "Resize",
            Message::ConfirmSession { .. } => "ConfirmSession",
            Message::SetupMcpApp { .. } => "SetupMcpApp",
            Message::SetupMcpAppResult { .. } => "SetupMcpAppResult",
            Message::ConfirmSessionResult { .. } => "ConfirmSessionResult",
            Message::StreamStart { .. } => "StreamStart",
            Message::StreamEnd { .. } => "StreamEnd",
            Message::ViewerJoined { .. } => "ViewerJoined",
            Message::ViewerLeft { .. } => "ViewerLeft",
            Message::ListScreenSources { .. } => "ListScreenSources",
            Message::ScreenSourcesResult { .. } => "ScreenSourcesResult",
            Message::ScreenSourcesChanged { .. } => "ScreenSourcesChanged",
            Message::StartScreenStream { .. } => "StartScreenStream",
            Message::ScreenStreamReady { .. } => "ScreenStreamReady",
            Message::StopScreenStream { .. } => "StopScreenStream",
            Message::ScreenStreamStopped { .. } => "ScreenStreamStopped",
            Message::ScreenStreamNotify { .. } => "ScreenStreamNotify",
            Message::ScreenStreamStats { .. } => "ScreenStreamStats",
            Message::BootSimulator { .. } => "BootSimulator",
            Message::BootSimulatorResult { .. } => "BootSimulatorResult",
            Message::ShutdownSimulator { .. } => "ShutdownSimulator",
            Message::ShutdownSimulatorResult { .. } => "ShutdownSimulatorResult",
            Message::CopySimulatorClipboard { .. } => "CopySimulatorClipboard",
            Message::CopySimulatorClipboardResult { .. } => "CopySimulatorClipboardResult",
            Message::BootAndroidEmulator { .. } => "BootAndroidEmulator",
            Message::BootAndroidEmulatorResult { .. } => "BootAndroidEmulatorResult",
            Message::AndroidEmulatorBooted { .. } => "AndroidEmulatorBooted",
            Message::AndroidEmulatorShutdown { .. } => "AndroidEmulatorShutdown",
            Message::AndroidEmulatorOperationFailed { .. } => "AndroidEmulatorOperationFailed",
            Message::ShutdownAndroidEmulator { .. } => "ShutdownAndroidEmulator",
            Message::ShutdownAndroidEmulatorResult { .. } => "ShutdownAndroidEmulatorResult",
            Message::RotateAndroidDevice { .. } => "RotateAndroidDevice",
            Message::RotateAndroidDeviceResult { .. } => "RotateAndroidDeviceResult",
            Message::SessionBufferInfo { .. } => "SessionBufferInfo",
            Message::FetchHistoryRaw { .. } => "FetchHistoryRaw",
            Message::ExportSessionRawHistory { .. } => "ExportSessionRawHistory",
            Message::ExportSessionRawHistoryResult { .. } => "ExportSessionRawHistoryResult",
            Message::HistoryRawChunk { .. } => "HistoryRawChunk",
            Message::HistorySnapshot { .. } => "HistorySnapshot",
            Message::RequestReverseProxy { .. } => "RequestReverseProxy",
            Message::ReverseProxyReady { .. } => "ReverseProxyReady",
            Message::ListReverseProxies { .. } => "ListReverseProxies",
            Message::ReverseProxyList { .. } => "ReverseProxyList",
            Message::RequestSharedPortForward { .. } => "RequestSharedPortForward",
            Message::SharedPortForwardReady { .. } => "SharedPortForwardReady",
            Message::ListRelayServers => "ListRelayServers",
            Message::RelayRegisterServer { .. } => "RelayRegisterServer",
            Message::RelayServerRegistered { .. } => "RelayServerRegistered",
            Message::RelayServerList { .. } => "RelayServerList",
            Message::RelayAllocateUdpPairing { .. } => "RelayAllocateUdpPairing",
            Message::RelayUdpPairingAllocated { .. } => "RelayUdpPairingAllocated",
            Message::RelayReleaseUdpPairing { .. } => "RelayReleaseUdpPairing",
            Message::RelayScreenTokenRequest { .. } => "RelayScreenTokenRequest",
            Message::RelayScreenTokenResponse { .. } => "RelayScreenTokenResponse",
            Message::RelayScreenTokenRevoke { .. } => "RelayScreenTokenRevoke",
            Message::RelayScreenSourcesRequest { .. } => "RelayScreenSourcesRequest",
            Message::RelayScreenSourcesResponse { .. } => "RelayScreenSourcesResponse",
            Message::TerminalAttentionRequired { .. } => "TerminalAttentionRequired",
            Message::TerminalAttentionResolved { .. } => "TerminalAttentionResolved",
            Message::TerminalTaskCompleted { .. } => "TerminalTaskCompleted",
            Message::Ping => "Ping",
            Message::Pong => "Pong",
            Message::Error { .. } => "Error",
            Message::TranscribeAudio { .. } => "TranscribeAudio",
            Message::TranscribeAudioResult { .. } => "TranscribeAudioResult",
            Message::PolishVoiceInputText { .. } => "PolishVoiceInputText",
            Message::PolishVoiceInputTextResult { .. } => "PolishVoiceInputTextResult",
        }
    }

    /// Serialize a message into JSON bytes for transport.
    pub fn serialize(&self) -> anyhow::Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| anyhow::anyhow!("Failed to serialize message: {}", e))
    }

    /// Deserialize a message from JSON bytes received from transport.
    pub fn deserialize(data: &[u8]) -> anyhow::Result<Self> {
        serde_json::from_slice(data)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize message: {}", e))
    }

    pub fn from_transport_binary(data: &[u8]) -> anyhow::Result<Self> {
        if data.starts_with(OUTPUT_BINARY_FRAME_MAGIC) {
            return Self::from_binary_output_frame(data);
        }
        if data.starts_with(INPUT_BINARY_FRAME_MAGIC) {
            return Self::from_binary_input_frame(data);
        }
        if data.starts_with(HISTORY_BINARY_FRAME_MAGIC) {
            return Self::from_binary_history_frame(data);
        }
        if data.starts_with(HISTORY_SNAPSHOT_BINARY_FRAME_MAGIC) {
            return Self::from_binary_history_snapshot_frame(data);
        }
        if data.starts_with(PATH_CONTENT_BINARY_FRAME_MAGIC) {
            return Self::from_binary_path_content_frame(data);
        }
        if data.starts_with(FIND_FILES_BINARY_FRAME_MAGIC) {
            return Self::from_binary_find_files_frame(data);
        }
        if data.starts_with(SEARCH_PATH_NAMES_BINARY_FRAME_MAGIC) {
            return Self::from_binary_search_path_names_frame(data);
        }
        if data.starts_with(GIT_CHANGED_BINARY_FRAME_MAGIC)
            || data.starts_with(GIT_CHANGED_BINARY_FRAME_MAGIC_V1)
        {
            return Self::from_binary_git_changed_frame(data);
        }
        if data.starts_with(GIT_STAGE_BINARY_FRAME_MAGIC) {
            return Self::from_binary_git_path_op_frame(
                data,
                GIT_STAGE_BINARY_FRAME_MAGIC,
                |session_id, paths, repo_root| Message::GitStage {
                    session_id,
                    paths,
                    repo_root,
                },
            );
        }
        if data.starts_with(GIT_UNSTAGE_BINARY_FRAME_MAGIC) {
            return Self::from_binary_git_path_op_frame(
                data,
                GIT_UNSTAGE_BINARY_FRAME_MAGIC,
                |session_id, paths, repo_root| Message::GitUnstage {
                    session_id,
                    paths,
                    repo_root,
                },
            );
        }
        if data.starts_with(GIT_DISCARD_BINARY_FRAME_MAGIC) {
            return Self::from_binary_git_path_op_frame(
                data,
                GIT_DISCARD_BINARY_FRAME_MAGIC,
                |session_id, paths, repo_root| Message::GitDiscard {
                    session_id,
                    paths,
                    repo_root,
                },
            );
        }
        if data.starts_with(GIT_LOG_BINARY_FRAME_MAGIC) {
            return Self::from_binary_git_log_frame(data);
        }
        if data.starts_with(FILE_CONTENT_BINARY_FRAME_MAGIC) {
            return Self::from_binary_file_content_frame(data);
        }
        if data.starts_with(FILE_SAVE_BINARY_FRAME_MAGIC) {
            return Self::from_binary_file_save_frame(data);
        }
        if data.starts_with(GIT_DIFF_BINARY_FRAME_MAGIC) {
            return Self::from_binary_git_diff_frame(data);
        }
        if data.starts_with(GIT_BRANCHES_BINARY_FRAME_MAGIC) {
            return Self::from_binary_git_branches_frame(data);
        }
        if data.starts_with(AI_CONVERSATION_BINARY_FRAME_MAGIC) {
            return Self::from_binary_ai_conversation_frame(data);
        }
        if data.starts_with(AI_CONVERSATION_UPDATE_BINARY_FRAME_MAGIC) {
            return Self::from_binary_ai_conversation_update_frame(data);
        }
        if data.starts_with(GIT_ANALYSIS_BINARY_FRAME_MAGIC) {
            return Self::from_binary_git_analysis_frame(data);
        }
        if data.starts_with(TRANSCRIBE_AUDIO_BINARY_FRAME_MAGIC) {
            return Self::from_binary_transcribe_audio_frame(data);
        }
        if data.starts_with(GIT_REBASE_CONTENT_BINARY_FRAME_MAGIC) {
            return Self::from_binary_git_rebase_content_frame(data);
        }
        if data.starts_with(GIT_REBASE_RESOLVE_BINARY_FRAME_MAGIC) {
            return Self::from_binary_git_rebase_resolve_frame(data);
        }
        if data.starts_with(GIT_REBASE_START_BINARY_FRAME_MAGIC) {
            return Self::from_binary_git_rebase_start_frame(data);
        }
        if data.starts_with(GIT_REBASE_AI_RESOLVE_BINARY_FRAME_MAGIC) {
            return Self::from_binary_git_rebase_ai_resolve_frame(data);
        }
        if data.starts_with(GIT_REBASE_AI_RESULT_BINARY_FRAME_MAGIC) {
            return Self::from_binary_git_rebase_ai_result_frame(data);
        }

        Self::deserialize(data)
    }

    /// Serialize a message into JSON text for WebSocket text frames.
    pub fn to_text(&self) -> anyhow::Result<String> {
        serde_json::to_string(self)
            .map_err(|e| anyhow::anyhow!("Failed to serialize message: {}", e))
    }

    /// Deserialize a message from JSON text received from a WebSocket frame.
    pub fn from_text(data: &str) -> anyhow::Result<Self> {
        serde_json::from_str(data)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize message: {}", e))
    }

    pub fn to_binary_frame(&self) -> Option<Vec<u8>> {
        match self {
            Message::Output {
                session_id,
                start_offset,
                data,
            } => {
                let mut frame = Vec::with_capacity(OUTPUT_BINARY_FRAME_HEADER_LEN + data.len());
                frame.extend_from_slice(OUTPUT_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                frame.extend_from_slice(&start_offset.to_le_bytes());
                frame.extend_from_slice(data);
                Some(frame)
            }
            Message::Input {
                session_id,
                data,
                cursor_from_end,
            } => {
                let header_len = if cursor_from_end.is_some() {
                    INPUT_BINARY_FRAME_CURSOR_HEADER_LEN
                } else {
                    INPUT_BINARY_FRAME_HEADER_LEN
                };
                let mut frame = Vec::with_capacity(header_len + data.len());
                frame.extend_from_slice(INPUT_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                match cursor_from_end {
                    Some(cursor) => {
                        frame.push(1);
                        frame.extend_from_slice(&cursor.lines_from_end.to_le_bytes());
                        frame.extend_from_slice(&cursor.column.to_le_bytes());
                    }
                    None => frame.push(0),
                }
                frame.extend_from_slice(data);
                Some(frame)
            }
            Message::HistoryRawChunk {
                session_id,
                start_offset,
                end_offset,
                data,
            } => {
                let mut frame = Vec::with_capacity(HISTORY_BINARY_FRAME_HEADER_LEN + data.len());
                frame.extend_from_slice(HISTORY_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                frame.extend_from_slice(&start_offset.to_le_bytes());
                frame.extend_from_slice(&end_offset.to_le_bytes());
                frame.extend_from_slice(data);
                Some(frame)
            }
            Message::HistorySnapshot {
                session_id,
                base_offset,
                data,
            } => {
                let mut frame =
                    Vec::with_capacity(HISTORY_SNAPSHOT_BINARY_FRAME_HEADER_LEN + data.len());
                frame.extend_from_slice(HISTORY_SNAPSHOT_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                frame.extend_from_slice(&base_offset.to_le_bytes());
                frame.extend_from_slice(data);
                Some(frame)
            }
            Message::PathContentList {
                session_id,
                path,
                items,
                listing_generation,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(PATH_CONTENT_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                append_string(&mut frame, path)?;
                frame.extend_from_slice(&listing_generation.to_le_bytes());
                append_u32(&mut frame, items.len())?;
                for item in items {
                    append_string(&mut frame, &item.name)?;
                    frame.push(u8::from(item.is_dir.unwrap_or(false)));
                }
                for item in items {
                    let mut flags = 0u8;
                    if item.git_modified {
                        flags |= 0b0000_0001;
                    }
                    if item.git_ignored {
                        flags |= 0b0000_0010;
                    }
                    frame.push(flags);
                }
                Some(frame)
            }
            Message::FindFilesResult {
                session_id,
                request_id,
                query,
                chunk_index,
                is_final,
                results,
                error,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(FIND_FILES_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                append_string(&mut frame, request_id)?;
                append_string(&mut frame, query)?;
                frame.extend_from_slice(&chunk_index.to_le_bytes());
                frame.push(u8::from(*is_final));
                append_optional_string(&mut frame, error.as_deref())?;
                append_u32(&mut frame, results.len())?;
                for result in results {
                    append_string(&mut frame, &result.file)?;
                    append_string(&mut frame, &result.path)?;
                    append_u32(&mut frame, result.matches.len())?;
                    for item in &result.matches {
                        frame.extend_from_slice(&item.line.to_le_bytes());
                        append_string(&mut frame, &item.text)?;
                    }
                }
                Some(frame)
            }
            Message::SearchPathNamesResult {
                session_id,
                request_id,
                root_path,
                query,
                is_final,
                results,
                error,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(SEARCH_PATH_NAMES_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                append_string(&mut frame, request_id)?;
                append_string(&mut frame, root_path)?;
                append_string(&mut frame, query)?;
                frame.push(u8::from(*is_final));
                append_optional_string(&mut frame, error.as_deref())?;
                append_u32(&mut frame, results.len())?;
                for result in results {
                    append_string(&mut frame, &result.path)?;
                    append_string(&mut frame, &result.name)?;
                    frame.push(u8::from(result.is_dir));
                }
                Some(frame)
            }
            Message::GitChangedFilesList {
                session_id,
                commit,
                repo_root,
                items,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(GIT_CHANGED_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                append_u32(&mut frame, items.len())?;
                for item in items {
                    frame.push(git_status_to_u8(&item.status));
                    frame.push(u8::from(item.staged));
                    frame.push(u8::from(item.is_submodule));
                    frame.push(u8::from(item.submodule_dirty));
                    append_string(&mut frame, &item.file)?;
                }
                append_optional_string(&mut frame, commit.as_deref())?;
                append_optional_string(&mut frame, repo_root.as_deref())?;
                Some(frame)
            }
            Message::GitStage {
                session_id,
                paths,
                repo_root,
            } => Some(git_path_op_binary_frame(
                GIT_STAGE_BINARY_FRAME_MAGIC,
                *session_id,
                paths,
                repo_root.as_deref(),
            )?),
            Message::GitUnstage {
                session_id,
                paths,
                repo_root,
            } => Some(git_path_op_binary_frame(
                GIT_UNSTAGE_BINARY_FRAME_MAGIC,
                *session_id,
                paths,
                repo_root.as_deref(),
            )?),
            Message::GitDiscard {
                session_id,
                paths,
                repo_root,
            } => Some(git_path_op_binary_frame(
                GIT_DISCARD_BINARY_FRAME_MAGIC,
                *session_id,
                paths,
                repo_root.as_deref(),
            )?),
            Message::GitLogList {
                session_id,
                commit,
                branch,
                repo_root,
                commits,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(GIT_LOG_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                append_u32(&mut frame, commits.len())?;
                for commit in commits {
                    append_string(&mut frame, &commit.hash)?;
                    append_string(&mut frame, &commit.message)?;
                    append_string(&mut frame, &commit.author)?;
                    append_string(&mut frame, &commit.date)?;
                    append_string(&mut frame, &commit.exact_time)?;
                    append_u32(&mut frame, commit.parents.len())?;
                    for parent in &commit.parents {
                        append_string(&mut frame, parent)?;
                    }
                }
                append_optional_string(&mut frame, commit.as_deref())?;
                append_optional_string(&mut frame, branch.as_deref())?;
                append_optional_string(&mut frame, repo_root.as_deref())?;
                Some(frame)
            }
            Message::FileContent {
                session_id,
                path,
                data,
                diff_ranges,
                error,
            } if error.is_none() => {
                let mut frame = Vec::new();
                frame.extend_from_slice(FILE_CONTENT_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                append_string(&mut frame, path)?;
                append_u32(&mut frame, diff_ranges.len())?;
                for range in diff_ranges {
                    frame.extend_from_slice(&range.start_line.to_le_bytes());
                    frame.extend_from_slice(&range.end_line.to_le_bytes());
                }
                frame.extend_from_slice(data);
                Some(frame)
            }
            Message::SaveFile {
                session_id,
                path,
                data,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(FILE_SAVE_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                append_string(&mut frame, path)?;
                frame.extend_from_slice(data);
                Some(frame)
            }
            Message::GitDiffResult {
                session_id,
                path,
                diff,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(GIT_DIFF_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                append_string(&mut frame, path)?;
                frame.extend_from_slice(diff);
                Some(frame)
            }
            Message::GitBranchesResult {
                session_id,
                current,
                detached_head,
                branches,
                remote_tracking_hash,
                tags,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(GIT_BRANCHES_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                append_optional_string(&mut frame, current.as_deref())?;
                append_u32(&mut frame, branches.len())?;
                for branch in branches {
                    append_string(&mut frame, &branch.name)?;
                    frame.push(u8::from(branch.is_current));
                    frame.push(u8::from(branch.is_remote));
                }
                append_optional_string(&mut frame, remote_tracking_hash.as_deref())?;
                append_u32(&mut frame, tags.len())?;
                for tag in tags {
                    append_string(&mut frame, tag)?;
                }
                // Optional per-branch worktree paths — present in newer
                // protocol versions. Order matches `branches` above; entries
                // are optional strings (Some when the branch is checked out in
                // a linked worktree, None otherwise).
                let has_worktree_paths =
                    branches.iter().any(|branch| branch.worktree_path.is_some());
                if has_worktree_paths {
                    append_u32(&mut frame, branches.len())?;
                    for branch in branches {
                        append_optional_string(&mut frame, branch.worktree_path.as_deref())?;
                    }
                } else if detached_head.is_some() {
                    // A zero count distinguishes the following detached hash
                    // from the optional per-branch worktree section.
                    append_u32(&mut frame, 0)?;
                }
                if detached_head.is_some() {
                    append_optional_string(&mut frame, detached_head.as_deref())?;
                }
                Some(frame)
            }
            // Rebase messages with potentially large file payloads go
            // through the binary transport (chunked) instead of JSON text.
            Message::GitRebaseConflictContentResult {
                session_id,
                path,
                base,
                ours,
                theirs,
                error,
                ours_commit,
                theirs_commit,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(GIT_REBASE_CONTENT_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                append_string(&mut frame, path)?;
                let mut flags = 0u8;
                if base.is_some() {
                    flags |= 0b0000_0001;
                }
                if ours.is_some() {
                    flags |= 0b0000_0010;
                }
                if theirs.is_some() {
                    flags |= 0b0000_0100;
                }
                if error.is_some() {
                    flags |= 0b0000_1000;
                }
                if ours_commit.is_some() {
                    flags |= 0b0001_0000;
                }
                if theirs_commit.is_some() {
                    flags |= 0b0010_0000;
                }
                frame.push(flags);
                if let Some(base) = base {
                    append_string(&mut frame, base)?;
                }
                if let Some(ours) = ours {
                    append_string(&mut frame, ours)?;
                }
                if let Some(theirs) = theirs {
                    append_string(&mut frame, theirs)?;
                }
                if let Some(error) = error {
                    append_string(&mut frame, error)?;
                }
                // Appended after the legacy fields; the reader has no
                // trailing-bytes check, so old parsers ignore these.
                if let Some(meta) = ours_commit {
                    append_string(&mut frame, &meta.hash)?;
                    append_string(&mut frame, &meta.message)?;
                    append_string(&mut frame, &meta.author)?;
                    append_string(&mut frame, &meta.date)?;
                }
                if let Some(meta) = theirs_commit {
                    append_string(&mut frame, &meta.hash)?;
                    append_string(&mut frame, &meta.message)?;
                    append_string(&mut frame, &meta.author)?;
                    append_string(&mut frame, &meta.date)?;
                }
                Some(frame)
            }
            Message::GitRebaseResolve {
                session_id,
                path,
                content,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(GIT_REBASE_RESOLVE_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                append_string(&mut frame, path)?;
                append_string(&mut frame, content)?;
                Some(frame)
            }
            Message::GitRebaseStartResult {
                session_id,
                error,
                source_branch,
                commits,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(GIT_REBASE_START_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                append_optional_string(&mut frame, error.as_deref())?;
                append_optional_string(&mut frame, source_branch.as_deref())?;
                append_u32(&mut frame, commits.len())?;
                for commit in commits {
                    append_string(&mut frame, &commit.hash)?;
                    append_string(&mut frame, &commit.message)?;
                    append_string(&mut frame, &commit.author)?;
                    append_string(&mut frame, &commit.date)?;
                    append_string(&mut frame, &commit.action)?;
                }
                Some(frame)
            }
            // AI resolve run payloads carry conflict hunk contents, so they
            // use the chunked binary transport like the other rebase frames.
            Message::GitRebaseAiResolve {
                session_id,
                run_id,
                instruction,
                ours_label,
                theirs_label,
                targets,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(GIT_REBASE_AI_RESOLVE_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                frame.extend_from_slice(&run_id.to_le_bytes());
                append_string(&mut frame, instruction)?;
                let mut flags = 0u8;
                if ours_label.is_some() {
                    flags |= 0b0000_0001;
                }
                if theirs_label.is_some() {
                    flags |= 0b0000_0010;
                }
                frame.push(flags);
                if let Some(ours_label) = ours_label {
                    append_string(&mut frame, ours_label)?;
                }
                if let Some(theirs_label) = theirs_label {
                    append_string(&mut frame, theirs_label)?;
                }
                append_u32(&mut frame, targets.len())?;
                for target in targets {
                    append_string(&mut frame, &target.path)?;
                    append_u32(&mut frame, target.hunks.len())?;
                    for hunk in &target.hunks {
                        frame.extend_from_slice(&hunk.idx.to_le_bytes());
                        append_lines(&mut frame, &hunk.base)?;
                        append_lines(&mut frame, &hunk.ours)?;
                        append_lines(&mut frame, &hunk.theirs)?;
                    }
                }
                Some(frame)
            }
            Message::GitRebaseAiResolveResult {
                session_id,
                run_id,
                hunk,
                error,
                done,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(GIT_REBASE_AI_RESULT_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                frame.extend_from_slice(&run_id.to_le_bytes());
                let mut flags = 0u8;
                if *done {
                    flags |= 0b0000_0001;
                }
                if error.is_some() {
                    flags |= 0b0000_0010;
                }
                if hunk.is_some() {
                    flags |= 0b0000_0100;
                }
                frame.push(flags);
                if let Some(error) = error {
                    append_string(&mut frame, error)?;
                }
                if let Some(hunk) = hunk {
                    append_string(&mut frame, &hunk.path)?;
                    frame.extend_from_slice(&hunk.idx.to_le_bytes());
                    frame.push(match hunk.status.as_str() {
                        "running" => 0u8,
                        "done" => 1u8,
                        _ => 2u8,
                    });
                    let mut hunk_flags = 0u8;
                    if hunk.lines.is_some() {
                        hunk_flags |= 0b0000_0001;
                    }
                    if hunk.strategy.is_some() {
                        hunk_flags |= 0b0000_0010;
                    }
                    if hunk.explanation.is_some() {
                        hunk_flags |= 0b0000_0100;
                    }
                    if hunk.confidence.is_some() {
                        hunk_flags |= 0b0000_1000;
                    }
                    if hunk.error.is_some() {
                        hunk_flags |= 0b0001_0000;
                    }
                    frame.push(hunk_flags);
                    if let Some(lines) = &hunk.lines {
                        append_lines(&mut frame, lines)?;
                    }
                    if let Some(strategy) = &hunk.strategy {
                        append_string(&mut frame, strategy)?;
                    }
                    if let Some(explanation) = &hunk.explanation {
                        append_string(&mut frame, explanation)?;
                    }
                    if let Some(confidence) = hunk.confidence {
                        frame.extend_from_slice(&confidence.to_le_bytes());
                    }
                    if let Some(hunk_error) = &hunk.error {
                        append_string(&mut frame, hunk_error)?;
                    }
                }
                Some(frame)
            }
            Message::AiConversationMessages {
                session_id,
                conversation_id,
                messages,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(AI_CONVERSATION_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                append_string(&mut frame, conversation_id)?;
                append_u32(&mut frame, messages.len())?;
                for message in messages {
                    append_string(&mut frame, &message.id)?;
                    frame.push(ai_role_to_u8(&message.role));
                    frame.push(ai_message_kind_to_u8(&message.kind));
                    append_string(&mut frame, &message.time)?;
                    append_string(&mut frame, &message.model)?;
                    append_string(&mut frame, &message.markdown)?;
                }
                Some(frame)
            }
            Message::AiConversationMessagesUpdate {
                session_id,
                conversation_id,
                start_index,
                messages,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(AI_CONVERSATION_UPDATE_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                append_string(&mut frame, conversation_id)?;
                append_u32(&mut frame, *start_index)?;
                append_u32(&mut frame, messages.len())?;
                for message in messages {
                    append_string(&mut frame, &message.id)?;
                    frame.push(ai_role_to_u8(&message.role));
                    frame.push(ai_message_kind_to_u8(&message.kind));
                    append_string(&mut frame, &message.time)?;
                    append_string(&mut frame, &message.model)?;
                    append_string(&mut frame, &message.markdown)?;
                }
                Some(frame)
            }
            Message::GitAnalyzeDiffResult {
                session_id,
                hunks,
                error: _,
                done,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(GIT_ANALYSIS_BINARY_FRAME_MAGIC);
                frame.extend_from_slice(session_id.as_bytes());
                frame.push(if *done { 1 } else { 0 });
                append_u32(&mut frame, hunks.len())?;
                for hunk in hunks {
                    frame.extend_from_slice(&hunk.old_start.to_le_bytes());
                    frame.extend_from_slice(&hunk.old_count.to_le_bytes());
                    frame.extend_from_slice(&hunk.new_start.to_le_bytes());
                    frame.extend_from_slice(&hunk.new_count.to_le_bytes());
                    append_string(&mut frame, &hunk.analysis)?;
                }
                Some(frame)
            }
            Message::TranscribeAudio {
                data,
                compressed,
                window_id,
                event_id,
                context,
            } => {
                let mut frame = Vec::new();
                frame.extend_from_slice(TRANSCRIBE_AUDIO_BINARY_FRAME_MAGIC);
                frame.push(u8::from(*compressed));
                frame.extend_from_slice(&window_id.to_le_bytes());
                append_string(&mut frame, event_id)?;
                let context_json = serde_json::to_string(context).ok()?;
                append_string(&mut frame, &context_json)?;
                frame.extend_from_slice(data);
                Some(frame)
            }
            _ => None,
        }
    }

    pub fn to_transport_binary_frames(&self) -> Option<Vec<Vec<u8>>> {
        let payload = self.to_binary_frame()?;
        Some(chunk_transport_binary_payload(&payload))
    }

    fn from_binary_output_frame(data: &[u8]) -> anyhow::Result<Self> {
        if data.len() < OUTPUT_BINARY_FRAME_HEADER_LEN {
            anyhow::bail!(
                "Binary output frame too short: expected at least {} bytes, got {}",
                OUTPUT_BINARY_FRAME_HEADER_LEN,
                data.len()
            );
        }

        let mut session_id_bytes = [0u8; 16];
        session_id_bytes.copy_from_slice(&data[4..20]);

        let mut start_offset_bytes = [0u8; 8];
        start_offset_bytes.copy_from_slice(&data[20..28]);

        Ok(Message::Output {
            session_id: SessionId::from_bytes(session_id_bytes),
            start_offset: u64::from_le_bytes(start_offset_bytes),
            data: data[28..].to_vec(),
        })
    }

    fn from_binary_input_frame(data: &[u8]) -> anyhow::Result<Self> {
        if data.len() < INPUT_BINARY_FRAME_HEADER_LEN {
            anyhow::bail!(
                "Binary input frame too short: expected at least {} bytes, got {}",
                INPUT_BINARY_FRAME_HEADER_LEN,
                data.len()
            );
        }

        let mut session_id_bytes = [0u8; 16];
        session_id_bytes.copy_from_slice(&data[4..20]);
        let cursor_flag = data[20];
        let (cursor_from_end, data_offset) = match cursor_flag {
            0 => (None, INPUT_BINARY_FRAME_HEADER_LEN),
            1 => {
                if data.len() < INPUT_BINARY_FRAME_CURSOR_HEADER_LEN {
                    anyhow::bail!(
                        "Binary input frame too short for cursor metadata: expected at least {} bytes, got {}",
                        INPUT_BINARY_FRAME_CURSOR_HEADER_LEN,
                        data.len()
                    );
                }

                let mut lines_from_end_bytes = [0u8; 4];
                lines_from_end_bytes.copy_from_slice(&data[21..25]);
                let mut column_bytes = [0u8; 2];
                column_bytes.copy_from_slice(&data[25..27]);
                (
                    Some(CursorFromEnd {
                        lines_from_end: u32::from_le_bytes(lines_from_end_bytes),
                        column: u16::from_le_bytes(column_bytes),
                    }),
                    INPUT_BINARY_FRAME_CURSOR_HEADER_LEN,
                )
            }
            other => anyhow::bail!("Unknown binary input cursor flag {}", other),
        };

        Ok(Message::Input {
            session_id: SessionId::from_bytes(session_id_bytes),
            data: data[data_offset..].to_vec(),
            cursor_from_end,
        })
    }

    fn from_binary_history_frame(data: &[u8]) -> anyhow::Result<Self> {
        if data.len() < HISTORY_BINARY_FRAME_HEADER_LEN {
            anyhow::bail!(
                "Binary history frame too short: expected at least {} bytes, got {}",
                HISTORY_BINARY_FRAME_HEADER_LEN,
                data.len()
            );
        }

        let mut session_id_bytes = [0u8; 16];
        session_id_bytes.copy_from_slice(&data[4..20]);

        let mut start_offset_bytes = [0u8; 8];
        start_offset_bytes.copy_from_slice(&data[20..28]);

        let mut end_offset_bytes = [0u8; 8];
        end_offset_bytes.copy_from_slice(&data[28..36]);

        Ok(Message::HistoryRawChunk {
            session_id: SessionId::from_bytes(session_id_bytes),
            start_offset: u64::from_le_bytes(start_offset_bytes),
            end_offset: u64::from_le_bytes(end_offset_bytes),
            data: data[36..].to_vec(),
        })
    }

    fn from_binary_history_snapshot_frame(data: &[u8]) -> anyhow::Result<Self> {
        if data.len() < HISTORY_SNAPSHOT_BINARY_FRAME_HEADER_LEN {
            anyhow::bail!(
                "Binary history snapshot frame too short: expected at least {} bytes, got {}",
                HISTORY_SNAPSHOT_BINARY_FRAME_HEADER_LEN,
                data.len()
            );
        }

        let mut session_id_bytes = [0u8; 16];
        session_id_bytes.copy_from_slice(&data[4..20]);

        let mut base_offset_bytes = [0u8; 8];
        base_offset_bytes.copy_from_slice(&data[20..28]);

        Ok(Message::HistorySnapshot {
            session_id: SessionId::from_bytes(session_id_bytes),
            base_offset: u64::from_le_bytes(base_offset_bytes),
            data: data[28..].to_vec(),
        })
    }

    fn from_binary_path_content_frame(data: &[u8]) -> anyhow::Result<Self> {
        let (session_id, mut offset) =
            read_binary_session_header(data, PATH_CONTENT_BINARY_FRAME_MAGIC)?;
        let path = read_string(data, &mut offset)?;
        let listing_generation = read_u64(data, &mut offset)?;
        let count = read_u32(data, &mut offset)? as usize;
        let mut items = Vec::with_capacity(count);
        for _ in 0..count {
            let name = read_string(data, &mut offset)?;
            let is_dir = read_u8(data, &mut offset)? != 0;
            items.push(if is_dir {
                FileListItem::directory(name)
            } else {
                FileListItem::file(name)
            });
        }
        for item in &mut items {
            if offset >= data.len() {
                break;
            }
            let flags = read_u8(data, &mut offset)?;
            item.git_modified = flags & 0b0000_0001 != 0;
            item.git_ignored = flags & 0b0000_0010 != 0;
        }
        Ok(Message::PathContentList {
            session_id,
            path,
            items,
            listing_generation,
        })
    }

    fn from_binary_find_files_frame(data: &[u8]) -> anyhow::Result<Self> {
        let (session_id, mut offset) =
            read_binary_session_header(data, FIND_FILES_BINARY_FRAME_MAGIC)?;
        let request_id = read_string(data, &mut offset)?;
        let query = read_string(data, &mut offset)?;
        let chunk_index = read_u32(data, &mut offset)?;
        let is_final = read_u8(data, &mut offset)? != 0;
        let error = read_optional_string(data, &mut offset)?;
        let count = read_u32(data, &mut offset)? as usize;
        let mut results = Vec::with_capacity(count);
        for _ in 0..count {
            let file = read_string(data, &mut offset)?;
            let path = read_string(data, &mut offset)?;
            let match_count = read_u32(data, &mut offset)? as usize;
            let mut matches = Vec::with_capacity(match_count);
            for _ in 0..match_count {
                let line = read_u32(data, &mut offset)?;
                let text = read_string(data, &mut offset)?;
                matches.push(FileSearchMatch { line, text });
            }
            results.push(FileSearchResult {
                file,
                path,
                matches,
            });
        }
        Ok(Message::FindFilesResult {
            session_id,
            request_id,
            query,
            chunk_index,
            is_final,
            results,
            error,
        })
    }

    fn from_binary_search_path_names_frame(data: &[u8]) -> anyhow::Result<Self> {
        let (session_id, mut offset) =
            read_binary_session_header(data, SEARCH_PATH_NAMES_BINARY_FRAME_MAGIC)?;
        let request_id = read_string(data, &mut offset)?;
        let root_path = read_string(data, &mut offset)?;
        let query = read_string(data, &mut offset)?;
        let is_final = read_u8(data, &mut offset)? != 0;
        let error = read_optional_string(data, &mut offset)?;
        let count = read_u32(data, &mut offset)? as usize;
        let mut results = Vec::with_capacity(count);
        for _ in 0..count {
            let path = read_string(data, &mut offset)?;
            let name = read_string(data, &mut offset)?;
            let is_dir = read_u8(data, &mut offset)? != 0;
            results.push(PathNameMatch { path, name, is_dir });
        }
        Ok(Message::SearchPathNamesResult {
            session_id,
            request_id,
            root_path,
            query,
            is_final,
            results,
            error,
        })
    }

    fn from_binary_git_changed_frame(data: &[u8]) -> anyhow::Result<Self> {
        let v2 = data.starts_with(GIT_CHANGED_BINARY_FRAME_MAGIC);
        let magic = if v2 {
            GIT_CHANGED_BINARY_FRAME_MAGIC
        } else {
            GIT_CHANGED_BINARY_FRAME_MAGIC_V1
        };
        let (session_id, mut offset) = read_binary_session_header(data, magic)?;
        let count = read_u32(data, &mut offset)? as usize;
        let mut items = Vec::with_capacity(count);
        for _ in 0..count {
            let status = git_status_from_u8(read_u8(data, &mut offset)?)?;
            let staged = read_u8(data, &mut offset)? != 0;
            let is_submodule = v2 && read_u8(data, &mut offset)? != 0;
            let submodule_dirty = v2 && read_u8(data, &mut offset)? != 0;
            let file = read_string(data, &mut offset)?;
            items.push(GitChangedFile {
                file,
                status,
                staged,
                is_submodule,
                submodule_dirty,
            });
        }
        let commit = if offset < data.len() {
            read_optional_string(data, &mut offset)?
        } else {
            None
        };
        let repo_root = if offset < data.len() {
            read_optional_string(data, &mut offset)?
        } else {
            None
        };
        Ok(Message::GitChangedFilesList {
            session_id,
            commit,
            repo_root,
            items,
        })
    }

    fn from_binary_git_path_op_frame<F>(
        data: &[u8],
        magic: &[u8; 4],
        build: F,
    ) -> anyhow::Result<Self>
    where
        F: FnOnce(SessionId, Vec<String>, Option<String>) -> Self,
    {
        let (session_id, mut offset) = read_binary_session_header(data, magic)?;
        let count = read_u32(data, &mut offset)? as usize;
        let mut paths = Vec::with_capacity(count);
        for _ in 0..count {
            paths.push(read_string(data, &mut offset)?);
        }
        // Optional trailing repo_root (absent on pre-extension frames).
        let repo_root = if offset < data.len() {
            match read_u8(data, &mut offset)? {
                1 => Some(read_string(data, &mut offset)?),
                _ => None,
            }
        } else {
            None
        };
        Ok(build(session_id, paths, repo_root))
    }

    fn from_binary_git_log_frame(data: &[u8]) -> anyhow::Result<Self> {
        let (session_id, mut offset) =
            read_binary_session_header(data, GIT_LOG_BINARY_FRAME_MAGIC)?;
        let count = read_u32(data, &mut offset)? as usize;
        let mut commits = Vec::with_capacity(count);
        for _ in 0..count {
            let hash = read_string(data, &mut offset)?;
            let message = read_string(data, &mut offset)?;
            let author = read_string(data, &mut offset)?;
            let date = read_string(data, &mut offset)?;
            let exact_time = read_string(data, &mut offset)?;
            let parent_count = if offset < data.len() {
                read_u32(data, &mut offset)? as usize
            } else {
                0
            };
            let mut parents = Vec::with_capacity(parent_count);
            for _ in 0..parent_count {
                parents.push(read_string(data, &mut offset)?);
            }
            commits.push(GitCommitInfo {
                hash,
                message,
                author,
                date,
                exact_time,
                parents,
            });
        }
        let commit = if offset < data.len() {
            read_optional_string(data, &mut offset)?
        } else {
            None
        };
        let branch = if offset < data.len() {
            read_optional_string(data, &mut offset)?
        } else {
            None
        };
        let repo_root = if offset < data.len() {
            read_optional_string(data, &mut offset)?
        } else {
            None
        };
        Ok(Message::GitLogList {
            session_id,
            commit,
            branch,
            repo_root,
            commits,
        })
    }

    fn from_binary_file_content_frame(data: &[u8]) -> anyhow::Result<Self> {
        let (session_id, mut offset) =
            read_binary_session_header(data, FILE_CONTENT_BINARY_FRAME_MAGIC)?;
        let path = read_string(data, &mut offset)?;
        let range_count = read_u32(data, &mut offset)? as usize;
        let mut diff_ranges = Vec::with_capacity(range_count);
        for _ in 0..range_count {
            let start_line = read_u32(data, &mut offset)?;
            let end_line = read_u32(data, &mut offset)?;
            diff_ranges.push(FileDiffRange {
                start_line,
                end_line,
            });
        }
        let file_data = data[offset..].to_vec();
        Ok(Message::FileContent {
            session_id,
            path,
            data: file_data,
            diff_ranges,
            error: None,
        })
    }

    fn from_binary_file_save_frame(data: &[u8]) -> anyhow::Result<Self> {
        let (session_id, mut offset) =
            read_binary_session_header(data, FILE_SAVE_BINARY_FRAME_MAGIC)?;
        let path = read_string(data, &mut offset)?;
        let file_data = data[offset..].to_vec();
        Ok(Message::SaveFile {
            session_id,
            path,
            data: file_data,
        })
    }

    fn from_binary_git_diff_frame(data: &[u8]) -> anyhow::Result<Self> {
        let (session_id, mut offset) =
            read_binary_session_header(data, GIT_DIFF_BINARY_FRAME_MAGIC)?;
        let path = read_string(data, &mut offset)?;
        // Raw bytes: the client decodes with the user-selected encoding.
        let diff = data[offset..].to_vec();
        Ok(Message::GitDiffResult {
            session_id,
            path,
            diff,
        })
    }

    fn from_binary_git_branches_frame(data: &[u8]) -> anyhow::Result<Self> {
        let (session_id, mut offset) =
            read_binary_session_header(data, GIT_BRANCHES_BINARY_FRAME_MAGIC)?;
        let current = read_optional_string(data, &mut offset)?;
        let count = read_u32(data, &mut offset)? as usize;
        let mut branches = Vec::with_capacity(count);
        for _ in 0..count {
            branches.push(GitBranchInfo {
                name: read_string(data, &mut offset)?,
                is_current: read_u8(data, &mut offset)? != 0,
                is_remote: read_u8(data, &mut offset)? != 0,
                worktree_path: None,
            });
        }
        // Read optional remote_tracking_hash — present in newer protocol versions
        let remote_tracking_hash = if offset < data.len() {
            read_optional_string(data, &mut offset)?
        } else {
            None
        };
        // Read optional tags list — present in newer protocol versions
        let tags = if offset < data.len() {
            let tag_count = read_u32(data, &mut offset)? as usize;
            let mut tags = Vec::with_capacity(tag_count);
            for _ in 0..tag_count {
                tags.push(read_string(data, &mut offset)?);
            }
            tags
        } else {
            Vec::new()
        };
        // Read optional per-branch worktree paths — present in newer protocol
        // versions. Order matches `branches`; an optional string per branch.
        if offset < data.len() {
            let wt_count = read_u32(data, &mut offset)? as usize;
            if wt_count == branches.len() {
                for branch in branches.iter_mut() {
                    branch.worktree_path = read_optional_string(data, &mut offset)?;
                }
            } else if wt_count != 0 {
                anyhow::bail!(
                    "Git branches binary frame has {wt_count} worktree paths for {} branches",
                    branches.len()
                );
            }
        }
        let detached_head = if offset < data.len() {
            read_optional_string(data, &mut offset)?
        } else {
            None
        };
        if offset != data.len() {
            anyhow::bail!(
                "Git branches binary frame has {} trailing bytes",
                data.len() - offset
            );
        }
        Ok(Message::GitBranchesResult {
            session_id,
            current,
            detached_head,
            branches,
            remote_tracking_hash,
            tags,
        })
    }

    fn from_binary_git_rebase_content_frame(data: &[u8]) -> anyhow::Result<Self> {
        let (session_id, mut offset) =
            read_binary_session_header(data, GIT_REBASE_CONTENT_BINARY_FRAME_MAGIC)?;
        let path = read_string(data, &mut offset)?;
        let flags = read_u8(data, &mut offset)?;
        let base = if flags & 0b0000_0001 != 0 {
            Some(read_string(data, &mut offset)?)
        } else {
            None
        };
        let ours = if flags & 0b0000_0010 != 0 {
            Some(read_string(data, &mut offset)?)
        } else {
            None
        };
        let theirs = if flags & 0b0000_0100 != 0 {
            Some(read_string(data, &mut offset)?)
        } else {
            None
        };
        let error = if flags & 0b0000_1000 != 0 {
            Some(read_string(data, &mut offset)?)
        } else {
            None
        };
        // Appended extension blocks (absent in frames from older servers).
        let mut read_meta = |data: &[u8], offset: &mut usize| -> anyhow::Result<GitCommitMeta> {
            Ok(GitCommitMeta {
                hash: read_string(data, offset)?,
                message: read_string(data, offset)?,
                author: read_string(data, offset)?,
                date: read_string(data, offset)?,
            })
        };
        let ours_commit = if flags & 0b0001_0000 != 0 {
            Some(read_meta(data, &mut offset)?)
        } else {
            None
        };
        let theirs_commit = if flags & 0b0010_0000 != 0 {
            Some(read_meta(data, &mut offset)?)
        } else {
            None
        };
        Ok(Message::GitRebaseConflictContentResult {
            session_id,
            path,
            base,
            ours,
            theirs,
            error,
            ours_commit,
            theirs_commit,
        })
    }

    fn from_binary_git_rebase_resolve_frame(data: &[u8]) -> anyhow::Result<Self> {
        let (session_id, mut offset) =
            read_binary_session_header(data, GIT_REBASE_RESOLVE_BINARY_FRAME_MAGIC)?;
        let path = read_string(data, &mut offset)?;
        let content = read_string(data, &mut offset)?;
        Ok(Message::GitRebaseResolve {
            session_id,
            path,
            content,
        })
    }

    fn from_binary_git_rebase_start_frame(data: &[u8]) -> anyhow::Result<Self> {
        let (session_id, mut offset) =
            read_binary_session_header(data, GIT_REBASE_START_BINARY_FRAME_MAGIC)?;
        let error = read_optional_string(data, &mut offset)?;
        let source_branch = read_optional_string(data, &mut offset)?;
        let count = read_u32(data, &mut offset)? as usize;
        let mut commits = Vec::with_capacity(count);
        for _ in 0..count {
            commits.push(GitRebaseTodoCommit {
                hash: read_string(data, &mut offset)?,
                message: read_string(data, &mut offset)?,
                author: read_string(data, &mut offset)?,
                date: read_string(data, &mut offset)?,
                action: read_string(data, &mut offset)?,
            });
        }
        Ok(Message::GitRebaseStartResult {
            session_id,
            error,
            source_branch,
            commits,
        })
    }

    fn from_binary_git_rebase_ai_resolve_frame(data: &[u8]) -> anyhow::Result<Self> {
        let (session_id, mut offset) =
            read_binary_session_header(data, GIT_REBASE_AI_RESOLVE_BINARY_FRAME_MAGIC)?;
        let run_id = read_u32(data, &mut offset)?;
        let instruction = read_string(data, &mut offset)?;
        let flags = read_u8(data, &mut offset)?;
        let ours_label = if flags & 0b0000_0001 != 0 {
            Some(read_string(data, &mut offset)?)
        } else {
            None
        };
        let theirs_label = if flags & 0b0000_0010 != 0 {
            Some(read_string(data, &mut offset)?)
        } else {
            None
        };
        let target_count = read_u32(data, &mut offset)? as usize;
        let mut targets = Vec::with_capacity(target_count);
        for _ in 0..target_count {
            let path = read_string(data, &mut offset)?;
            let hunk_count = read_u32(data, &mut offset)? as usize;
            let mut hunks = Vec::with_capacity(hunk_count);
            for _ in 0..hunk_count {
                hunks.push(vterm_common::GitRebaseAiHunk {
                    idx: read_u32(data, &mut offset)?,
                    base: read_lines(data, &mut offset)?,
                    ours: read_lines(data, &mut offset)?,
                    theirs: read_lines(data, &mut offset)?,
                });
            }
            targets.push(GitRebaseAiTarget { path, hunks });
        }
        Ok(Message::GitRebaseAiResolve {
            session_id,
            run_id,
            instruction,
            ours_label,
            theirs_label,
            targets,
        })
    }

    fn from_binary_git_rebase_ai_result_frame(data: &[u8]) -> anyhow::Result<Self> {
        let (session_id, mut offset) =
            read_binary_session_header(data, GIT_REBASE_AI_RESULT_BINARY_FRAME_MAGIC)?;
        let run_id = read_u32(data, &mut offset)?;
        let flags = read_u8(data, &mut offset)?;
        let done = flags & 0b0000_0001 != 0;
        let error = if flags & 0b0000_0010 != 0 {
            Some(read_string(data, &mut offset)?)
        } else {
            None
        };
        let hunk = if flags & 0b0000_0100 != 0 {
            let path = read_string(data, &mut offset)?;
            let idx = read_u32(data, &mut offset)?;
            let status = match read_u8(data, &mut offset)? {
                0 => "running",
                1 => "done",
                _ => "error",
            }
            .to_string();
            let hunk_flags = read_u8(data, &mut offset)?;
            let lines = if hunk_flags & 0b0000_0001 != 0 {
                Some(read_lines(data, &mut offset)?)
            } else {
                None
            };
            let strategy = if hunk_flags & 0b0000_0010 != 0 {
                Some(read_string(data, &mut offset)?)
            } else {
                None
            };
            let explanation = if hunk_flags & 0b0000_0100 != 0 {
                Some(read_string(data, &mut offset)?)
            } else {
                None
            };
            let confidence = if hunk_flags & 0b0000_1000 != 0 {
                Some(read_f32(data, &mut offset)?)
            } else {
                None
            };
            let hunk_error = if hunk_flags & 0b0001_0000 != 0 {
                Some(read_string(data, &mut offset)?)
            } else {
                None
            };
            Some(GitRebaseAiHunkEvent {
                path,
                idx,
                status,
                lines,
                strategy,
                explanation,
                confidence,
                error: hunk_error,
            })
        } else {
            None
        };
        Ok(Message::GitRebaseAiResolveResult {
            session_id,
            run_id,
            hunk,
            error,
            done,
        })
    }

    fn from_binary_ai_conversation_frame(data: &[u8]) -> anyhow::Result<Self> {
        let (session_id, mut offset) =
            read_binary_session_header(data, AI_CONVERSATION_BINARY_FRAME_MAGIC)?;
        let conversation_id = read_string(data, &mut offset)?;
        let count = read_u32(data, &mut offset)? as usize;
        let mut messages = Vec::with_capacity(count);
        for _ in 0..count {
            messages.push(AiConversationMessage {
                id: read_string(data, &mut offset)?,
                role: ai_role_from_u8(read_u8(data, &mut offset)?)?,
                kind: ai_message_kind_from_u8(read_u8(data, &mut offset)?)?,
                time: read_string(data, &mut offset)?,
                model: read_string(data, &mut offset)?,
                markdown: read_string(data, &mut offset)?,
            });
        }
        Ok(Message::AiConversationMessages {
            session_id,
            conversation_id,
            messages,
        })
    }

    fn from_binary_ai_conversation_update_frame(data: &[u8]) -> anyhow::Result<Self> {
        let (session_id, mut offset) =
            read_binary_session_header(data, AI_CONVERSATION_UPDATE_BINARY_FRAME_MAGIC)?;
        let conversation_id = read_string(data, &mut offset)?;
        let start_index = read_u32(data, &mut offset)? as usize;
        let count = read_u32(data, &mut offset)? as usize;
        let mut messages = Vec::with_capacity(count);
        for _ in 0..count {
            messages.push(AiConversationMessage {
                id: read_string(data, &mut offset)?,
                role: ai_role_from_u8(read_u8(data, &mut offset)?)?,
                kind: ai_message_kind_from_u8(read_u8(data, &mut offset)?)?,
                time: read_string(data, &mut offset)?,
                model: read_string(data, &mut offset)?,
                markdown: read_string(data, &mut offset)?,
            });
        }
        Ok(Message::AiConversationMessagesUpdate {
            session_id,
            conversation_id,
            start_index,
            messages,
        })
    }

    fn from_binary_git_analysis_frame(data: &[u8]) -> anyhow::Result<Self> {
        let (session_id, mut offset) =
            read_binary_session_header(data, GIT_ANALYSIS_BINARY_FRAME_MAGIC)?;
        let done = *data.get(offset).unwrap_or(&0) != 0;
        offset += 1;
        let hunk_count = read_u32(data, &mut offset)? as usize;
        let mut hunks = Vec::with_capacity(hunk_count);
        for _ in 0..hunk_count {
            let old_start = read_u32(data, &mut offset)?;
            let old_count = read_u32(data, &mut offset)?;
            let new_start = read_u32(data, &mut offset)?;
            let new_count = read_u32(data, &mut offset)?;
            let analysis = read_string(data, &mut offset)?;
            hunks.push(vterm_common::HunkAnalysis {
                old_start,
                old_count,
                new_start,
                new_count,
                analysis,
            });
        }
        Ok(Message::GitAnalyzeDiffResult {
            session_id,
            hunks,
            error: None,
            done,
        })
    }

    fn from_binary_transcribe_audio_frame(data: &[u8]) -> anyhow::Result<Self> {
        if data.len() < 5 {
            anyhow::bail!(
                "Binary transcribe audio frame too short: expected at least 5 bytes, got {}",
                data.len()
            );
        }
        if !data.starts_with(TRANSCRIBE_AUDIO_BINARY_FRAME_MAGIC) {
            anyhow::bail!("Invalid binary transcribe audio frame magic");
        }
        let mut offset = 4;
        let compressed = read_u8(data, &mut offset)? != 0;
        let window_id = read_u64(data, &mut offset)?;
        let event_id = read_string(data, &mut offset)?;
        let context: VoiceInputOptimizationContext =
            serde_json::from_str(&read_string(data, &mut offset)?)
                .context("Invalid voice transcription context")?;
        let audio_data = data[offset..].to_vec();
        Ok(Message::TranscribeAudio {
            data: audio_data.into(),
            compressed,
            window_id,
            event_id,
            context,
        })
    }

    /// Get the session ID if this message is associated with a session
    pub fn session_id(&self) -> Option<SessionId> {
        match self {
            Message::SessionCreated { session_id } => Some(*session_id),
            Message::SplitTerminal {
                target_session_id, ..
            } => Some(*target_session_id),
            Message::SessionBufferInfo { session_id, .. } => Some(*session_id),
            Message::FetchHistoryRaw { session_id, .. } => Some(*session_id),
            Message::ExportSessionRawHistory { session_id } => Some(*session_id),
            Message::ExportSessionRawHistoryResult { session_id, .. } => Some(*session_id),
            Message::HistoryRawChunk { session_id, .. } => Some(*session_id),
            Message::HistorySnapshot { session_id, .. } => Some(*session_id),
            Message::CloseSession { session_id } => Some(*session_id),
            Message::RenameSession { session_id, .. } => Some(*session_id),
            Message::MoveTabToGroup { session_id, .. } => Some(*session_id),
            Message::MoveSessionToTab { session_id, .. } => Some(*session_id),
            Message::ApplyEnvironment { session_id, .. } => Some(*session_id),
            Message::ApplyEnvironmentResult { session_id, .. } => Some(*session_id),
            Message::SetNotificationEnabled { session_id, .. } => Some(*session_id),
            Message::SetNotificationEnabledResult { session_id, .. } => Some(*session_id),
            Message::SetAutoConfirmEnabled { session_id, .. } => Some(*session_id),
            Message::SetAutoConfirmEnabledResult { session_id, .. } => Some(*session_id),
            Message::SetSessionEncoding { session_id, .. } => Some(*session_id),
            Message::SetSessionEncodingResult { session_id, .. } => Some(*session_id),
            Message::SetTerminalThemeColors { session_id, .. } => Some(*session_id),
            Message::ClearSessionScreen { session_id } => Some(*session_id),
            Message::ClearSessionScreenResult { session_id, .. } => Some(*session_id),
            Message::ListPathContent { session_id, .. } => Some(*session_id),
            Message::SetFsWatchers { session_id, .. } => Some(*session_id),
            Message::PathContentList { session_id, .. } => Some(*session_id),
            Message::FindFiles { session_id, .. } => Some(*session_id),
            Message::FindFilesResult { session_id, .. } => Some(*session_id),
            Message::SearchPathNames { session_id, .. } => Some(*session_id),
            Message::SearchPathNamesResult { session_id, .. } => Some(*session_id),
            Message::ListAiConversations { session_id } => Some(*session_id),
            Message::StopWatchingAiConversation { session_id, .. } => Some(*session_id),
            Message::AiConversationList { session_id, .. } => Some(*session_id),
            Message::ReadAiConversation { session_id, .. } => Some(*session_id),
            Message::AiConversationMessages { session_id, .. } => Some(*session_id),
            Message::AiConversationMessagesUpdate { session_id, .. } => Some(*session_id),
            Message::TranslateMessage { session_id, .. } => Some(*session_id),
            Message::TranslateMessageResult { session_id, .. } => Some(*session_id),
            Message::GitChangedFiles { session_id, .. } => Some(*session_id),
            Message::GitChangedFilesList { session_id, .. } => Some(*session_id),
            Message::GitLog { session_id, .. } => Some(*session_id),
            Message::GitLogList { session_id, .. } => Some(*session_id),
            Message::GitDiff { session_id, .. } => Some(*session_id),
            Message::GitDiffResult { session_id, .. } => Some(*session_id),
            Message::GitStage { session_id, .. } => Some(*session_id),
            Message::GitUnstage { session_id, .. } => Some(*session_id),
            Message::GitDiscard { session_id, .. } => Some(*session_id),
            Message::GitCommit { session_id, .. } => Some(*session_id),
            Message::GitFetch { session_id } => Some(*session_id),
            Message::GitPull { session_id } => Some(*session_id),
            Message::GitPush { session_id, .. } => Some(*session_id),
            Message::GitRevert { session_id, .. } => Some(*session_id),
            Message::GitGenerateCommitMessageAgent { session_id } => Some(*session_id),
            Message::GitGenerateCommitMessageAccepted { session_id } => Some(*session_id),
            Message::GitAnalyzeDiff { session_id, .. } => Some(*session_id),
            Message::GitStageResult { session_id, .. } => Some(*session_id),
            Message::GitUnstageResult { session_id, .. } => Some(*session_id),
            Message::GitDiscardResult { session_id, .. } => Some(*session_id),
            Message::GitCommitResult { session_id, .. } => Some(*session_id),
            Message::GitFetchResult { session_id, .. } => Some(*session_id),
            Message::GitPullResult { session_id, .. } => Some(*session_id),
            Message::GitPushResult { session_id, .. } => Some(*session_id),
            Message::GitRevertResult { session_id, .. } => Some(*session_id),
            Message::GitGenerateCommitMessageResult { session_id, .. } => Some(*session_id),
            Message::GitAnalyzeDiffResult { session_id, .. } => Some(*session_id),
            Message::GitStashList { session_id } => Some(*session_id),
            Message::GitStashListResult { session_id, .. } => Some(*session_id),
            Message::GitStashPush { session_id, .. } => Some(*session_id),
            Message::GitStashApply { session_id, .. } => Some(*session_id),
            Message::GitStashPop { session_id, .. } => Some(*session_id),
            Message::GitStashDrop { session_id, .. } => Some(*session_id),
            Message::GitStashResult { session_id, .. } => Some(*session_id),
            Message::GitSubmoduleList { session_id } => Some(*session_id),
            Message::GitSubmoduleListResult { session_id, .. } => Some(*session_id),
            Message::GitBranches { session_id } => Some(*session_id),
            Message::GitBranchesResult { session_id, .. } => Some(*session_id),
            Message::GitCheckout { session_id, .. } => Some(*session_id),
            Message::GitCheckoutResult { session_id, .. } => Some(*session_id),
            Message::GitCreateBranch { session_id, .. } => Some(*session_id),
            Message::GitCreateBranchResult { session_id, .. } => Some(*session_id),
            Message::GitCreateWorktree { session_id, .. } => Some(*session_id),
            Message::GitCreateWorktreeResult { session_id, .. } => Some(*session_id),
            Message::GitDeleteBranch { session_id, .. } => Some(*session_id),
            Message::GitDeleteBranchResult { session_id, .. } => Some(*session_id),
            Message::GitMerge { session_id, .. } => Some(*session_id),
            Message::GitMergeResult { session_id, .. } => Some(*session_id),
            Message::GitRebaseStart { session_id, .. } => Some(*session_id),
            Message::GitRebaseStartResult { session_id, .. } => Some(*session_id),
            Message::GitRebaseApply { session_id, .. } => Some(*session_id),
            Message::GitRebaseApplyResult { session_id, .. } => Some(*session_id),
            Message::GitRebaseConflictContent { session_id, .. } => Some(*session_id),
            Message::GitRebaseConflictContentResult { session_id, .. } => Some(*session_id),
            Message::GitRebaseResolve { session_id, .. } => Some(*session_id),
            Message::GitRebaseResolveResult { session_id, .. } => Some(*session_id),
            Message::GitRebaseContinue { session_id, .. } => Some(*session_id),
            Message::GitRebaseContinueResult { session_id, .. } => Some(*session_id),
            Message::GitRebaseSkip { session_id, .. } => Some(*session_id),
            Message::GitRebaseSkipResult { session_id, .. } => Some(*session_id),
            Message::GitRebaseAbort { session_id, .. } => Some(*session_id),
            Message::GitRebaseAbortResult { session_id, .. } => Some(*session_id),
            Message::GitRebaseAiResolve { session_id, .. } => Some(*session_id),
            Message::GitRebaseAiResolveCancel { session_id } => Some(*session_id),
            Message::GitRebaseAiResolveResult { session_id, .. } => Some(*session_id),
            Message::FetchFile { session_id, .. } => Some(*session_id),
            Message::FileContent { session_id, .. } => Some(*session_id),
            Message::SaveFile { session_id, .. } => Some(*session_id),
            Message::SaveFileResult { session_id, .. } => Some(*session_id),
            Message::TempFileUploadStart { session_id, .. }
            | Message::TempFileUploadResult { session_id, .. } => Some(*session_id),
            Message::UploadStart { session_id, .. }
            | Message::UploadResult { session_id, .. }
            | Message::DownloadStart { session_id, .. }
            | Message::DownloadResult { session_id, .. } => Some(*session_id),
            Message::CreatePath { session_id, .. } => Some(*session_id),
            Message::CreatePathResult { session_id, .. } => Some(*session_id),
            Message::DeletePaths { session_id, .. } => Some(*session_id),
            Message::DeletePathsResult { session_id, .. } => Some(*session_id),
            Message::PastePaths { session_id, .. } => Some(*session_id),
            Message::PastePathsResult { session_id, .. } => Some(*session_id),
            Message::RenamePath { session_id, .. } => Some(*session_id),
            Message::RenamePathResult { session_id, .. } => Some(*session_id),
            // Optional field: older clients send `None` and their request
            // stays on the directly connected server (legacy behaviour).
            Message::GenerateVtermConfig { session_id, .. } => *session_id,
            Message::GenerateVtermConfigResult { session_id, .. } => *session_id,
            Message::StartStaticHttpServer { session_id, .. } => Some(*session_id),
            Message::StartStaticHttpServerResult { session_id, .. } => Some(*session_id),
            Message::RequestSharedPortForward { session_id, .. } => Some(*session_id),
            Message::SharedPortForwardReady { session_id, .. } => Some(*session_id),
            Message::DebugLaunchList { session_id } => Some(*session_id),
            Message::DebugLaunchListResult { session_id, .. } => Some(*session_id),
            Message::DebugStart { session_id, .. } => Some(*session_id),
            Message::DebugStartResult { session_id, .. } => Some(*session_id),
            Message::DebugOutput { session_id, .. } => Some(*session_id),
            Message::DebugDap { session_id, .. } => Some(*session_id),
            Message::QuickCommandList { session_id } => Some(*session_id),
            Message::QuickCommandListResult { session_id, .. } => Some(*session_id),
            Message::RunQuickCommand { session_id, .. } => Some(*session_id),
            Message::RunQuickCommandResult { session_id, .. } => Some(*session_id),
            Message::SessionClosed { session_id } => Some(*session_id),
            Message::Output { session_id, .. } => Some(*session_id),
            Message::Input { session_id, .. } => Some(*session_id),
            Message::Resize { session_id, .. } => Some(*session_id),
            Message::ConfirmSession { session_id } => Some(*session_id),
            Message::SetupMcpApp { session_id, .. } => Some(*session_id),
            Message::SetupMcpAppResult { session_id, .. } => Some(*session_id),
            Message::ConfirmSessionResult { session_id, .. } => Some(*session_id),
            Message::StreamStart { session_id, .. } => Some(*session_id),
            Message::StreamEnd { session_id } => Some(*session_id),
            Message::ViewerJoined { session_id, .. } => Some(*session_id),
            Message::ViewerLeft { session_id, .. } => Some(*session_id),
            Message::TerminalAttentionRequired { attention } => Some(attention.session_id),
            Message::TerminalAttentionResolved { session_id } => Some(*session_id),
            Message::TerminalTaskCompleted { completion } => Some(completion.session_id),
            _ => None,
        }
    }

    /// Server id embedded in the message for relay routing, if any.
    ///
    /// Request messages without a `session_id` (e.g. `RemoveRecentProjectDirectory`)
    /// carry an explicit `server_id` so the client can still route them to the
    /// owning server's relay request tunnel.
    pub fn relay_route_server_id(&self) -> Option<String> {
        match self {
            Message::RemoveRecentProjectDirectory { server_id, .. } => server_id.clone(),
            Message::BootSimulator { server_id, .. }
            | Message::ShutdownSimulator { server_id, .. }
            | Message::BootAndroidEmulator { server_id, .. }
            | Message::ShutdownAndroidEmulator { server_id, .. } => server_id.clone(),
            // Screen-stream messages carry server_id but are handled by the
            // relay (localhost server) directly — they do NOT go through a
            // transparent tunnel. The relay's handler inspects server_id and
            // coordinates with the shared server via RelayPrimary channel.
            _ => None,
        }
    }

    /// Check if this is a control message (not data)
    pub fn is_control(&self) -> bool {
        matches!(
            self,
            Message::Hello { .. }
                | Message::AuthChallenge { .. }
                | Message::AuthResponse { .. }
                | Message::HelloAck { .. }
                | Message::CreateSession { .. }
                | Message::SessionCreated { .. }
                | Message::SplitTerminal { .. }
                | Message::ListSessions
                | Message::SessionList { .. }
                | Message::ListTabGroups
                | Message::TabGroupList { .. }
                | Message::CreateTabGroup { .. }
                | Message::DeleteTabGroup { .. }
                | Message::RenameTabGroup { .. }
                | Message::MoveTabToGroup { .. }
                | Message::GetTabInfoList { .. }
                | Message::TabInfoList { .. }
                | Message::UpdateTerminalLayout { .. }
                | Message::MoveSessionToTab { .. }
                | Message::ListTerminalProfiles
                | Message::TerminalProfileList { .. }
                | Message::ListPathContent { .. }
                | Message::SetFsWatchers { .. }
                | Message::PathContentList { .. }
                | Message::FindFiles { .. }
                | Message::FindFilesResult { .. }
                | Message::SearchPathNames { .. }
                | Message::SearchPathNamesResult { .. }
                | Message::RecentProjectDirectoryList { .. }
                | Message::KnownAppList { .. }
                | Message::RemoveRecentProjectDirectory { .. }
                | Message::TerminalCommandHistoryList { .. }
                | Message::ListAiConversations { .. }
                | Message::StopWatchingAiConversation { .. }
                | Message::AiConversationList { .. }
                | Message::ReadAiConversation { .. }
                | Message::AiConversationMessages { .. }
                | Message::AiConversationMessagesUpdate { .. }
                | Message::TranslateMessage { .. }
                | Message::TranslateMessageResult { .. }
                | Message::GitChangedFiles { .. }
                | Message::GitChangedFilesList { .. }
                | Message::GitLog { .. }
                | Message::GitLogList { .. }
                | Message::GitDiff { .. }
                | Message::GitDiffResult { .. }
                | Message::GitStage { .. }
                | Message::GitUnstage { .. }
                | Message::GitDiscard { .. }
                | Message::GitCommit { .. }
                | Message::GitFetch { .. }
                | Message::GitPull { .. }
                | Message::GitPush { .. }
                | Message::GitRevert { .. }
                | Message::GitGenerateCommitMessageAgent { .. }
                | Message::GitAnalyzeDiff { .. }
                | Message::GitStageResult { .. }
                | Message::GitUnstageResult { .. }
                | Message::GitDiscardResult { .. }
                | Message::GitCommitResult { .. }
                | Message::GitFetchResult { .. }
                | Message::GitPullResult { .. }
                | Message::GitPushResult { .. }
                | Message::GitRevertResult { .. }
                | Message::GitGenerateCommitMessageAccepted { .. }
                | Message::GitGenerateCommitMessageResult { .. }
                | Message::GitAnalyzeDiffResult { .. }
                | Message::GitBranches { .. }
                | Message::GitBranchesResult { .. }
                | Message::GitCheckout { .. }
                | Message::GitCheckoutResult { .. }
                | Message::GitCreateBranch { .. }
                | Message::GitCreateBranchResult { .. }
                | Message::GitCreateWorktree { .. }
                | Message::GitCreateWorktreeResult { .. }
                | Message::GitDeleteBranch { .. }
                | Message::GitDeleteBranchResult { .. }
                | Message::GitMerge { .. }
                | Message::GitMergeResult { .. }
                | Message::GitRebaseStart { .. }
                | Message::GitRebaseStartResult { .. }
                | Message::GitRebaseApply { .. }
                | Message::GitRebaseApplyResult { .. }
                | Message::GitRebaseConflictContent { .. }
                | Message::GitRebaseConflictContentResult { .. }
                | Message::GitRebaseResolve { .. }
                | Message::GitRebaseResolveResult { .. }
                | Message::GitRebaseContinue { .. }
                | Message::GitRebaseContinueResult { .. }
                | Message::GitRebaseSkip { .. }
                | Message::GitRebaseSkipResult { .. }
                | Message::GitRebaseAbort { .. }
                | Message::GitRebaseAbortResult { .. }
                | Message::GitRebaseAiResolve { .. }
                | Message::GitRebaseAiResolveCancel { .. }
                | Message::GitRebaseAiResolveResult { .. }
                | Message::FetchFile { .. }
                | Message::FileContent { .. }
                | Message::SaveFile { .. }
                | Message::SaveFileResult { .. }
                | Message::TempFileUploadResult { .. }
                | Message::UploadResult { .. }
                | Message::DownloadResult { .. }
                | Message::CreatePath { .. }
                | Message::CreatePathResult { .. }
                | Message::DeletePaths { .. }
                | Message::DeletePathsResult { .. }
                | Message::PastePaths { .. }
                | Message::PastePathsResult { .. }
                | Message::RenamePath { .. }
                | Message::RenamePathResult { .. }
                | Message::GenerateVtermConfig { .. }
                | Message::GenerateVtermConfigResult { .. }
                | Message::StartStaticHttpServer { .. }
                | Message::StartStaticHttpServerResult { .. }
                | Message::DebugLaunchList { .. }
                | Message::DebugLaunchListResult { .. }
                | Message::DebugStart { .. }
                | Message::DebugStartResult { .. }
                | Message::DebugOutput { .. }
                | Message::DebugDap { .. }
                | Message::QuickCommandList { .. }
                | Message::QuickCommandListResult { .. }
                | Message::RunQuickCommand { .. }
                | Message::RunQuickCommandResult { .. }
                | Message::ListRelayServers
                | Message::SessionBufferInfo { .. }
                | Message::FetchHistoryRaw { .. }
                | Message::ExportSessionRawHistory { .. }
                | Message::ExportSessionRawHistoryResult { .. }
                | Message::HistoryRawChunk { .. }
                | Message::HistorySnapshot { .. }
                | Message::CloseSession { .. }
                | Message::CloseTab { .. }
                | Message::RenameSession { .. }
                | Message::ApplyEnvironment { .. }
                | Message::SetSessionEncoding { .. }
                | Message::SetTerminalThemeColors { .. }
                | Message::ClearSessionScreen { .. }
                | Message::SessionClosed { .. }
                | Message::StreamStart { .. }
                | Message::StreamEnd { .. }
                | Message::ViewerJoined { .. }
                | Message::ViewerLeft { .. }
                | Message::ListScreenSources { .. }
                | Message::ScreenSourcesResult { .. }
                | Message::ScreenSourcesChanged { .. }
                | Message::StartScreenStream { .. }
                | Message::ScreenStreamReady { .. }
                | Message::StopScreenStream { .. }
                | Message::ScreenStreamStopped { .. }
                | Message::ScreenStreamNotify { .. }
                | Message::ScreenStreamStats { .. }
                | Message::BootSimulator { .. }
                | Message::BootSimulatorResult { .. }
                | Message::ShutdownSimulator { .. }
                | Message::ShutdownSimulatorResult { .. }
                | Message::BootAndroidEmulator { .. }
                | Message::BootAndroidEmulatorResult { .. }
                | Message::AndroidEmulatorBooted { .. }
                | Message::AndroidEmulatorShutdown { .. }
                | Message::AndroidEmulatorOperationFailed { .. }
                | Message::ShutdownAndroidEmulator { .. }
                | Message::ShutdownAndroidEmulatorResult { .. }
                | Message::RotateAndroidDevice { .. }
                | Message::RotateAndroidDeviceResult { .. }
                | Message::TerminalAttentionRequired { .. }
                | Message::TerminalAttentionResolved { .. }
                | Message::TerminalTaskCompleted { .. }
                | Message::RequestReverseProxy { .. }
                | Message::ReverseProxyReady { .. }
                | Message::ListReverseProxies { .. }
                | Message::ReverseProxyList { .. }
                | Message::Ping
                | Message::Pong
                | Message::TranscribeAudio { .. }
                | Message::TranscribeAudioResult { .. }
                | Message::PolishVoiceInputText { .. }
                | Message::PolishVoiceInputTextResult { .. }
                | Message::Error { .. }
        )
    }

    pub fn is_terminal_stream_message(&self) -> bool {
        matches!(
            self,
            Message::SessionBufferInfo { .. }
                | Message::HistorySnapshot { .. }
                | Message::Output { .. }
                | Message::Input { .. }
                | Message::Resize { .. }
                | Message::StreamStart { .. }
                | Message::StreamEnd { .. }
                | Message::ViewerJoined { .. }
                | Message::ViewerLeft { .. }
        )
    }

    pub fn is_terminal_client_message(&self) -> bool {
        matches!(
            self,
            Message::Input { .. }
                | Message::Resize { .. }
                | Message::StreamStart { .. }
                | Message::StreamEnd { .. }
                | Message::SetFsWatchers { .. }
                | Message::SetTerminalThemeColors { .. }
        )
    }

    pub fn is_request_message(&self) -> bool {
        matches!(
            self,
            Message::ListPathContent { .. }
                | Message::FindFiles { .. }
                | Message::SearchPathNames { .. }
                | Message::GetTabInfoList { .. }
                | Message::RemoveRecentProjectDirectory { .. }
                | Message::ListAiConversations { .. }
                | Message::ReadAiConversation { .. }
                | Message::TranslateMessage { .. }
                | Message::GitChangedFiles { .. }
                | Message::GitLog { .. }
                | Message::GitDiff { .. }
                | Message::GitStage { .. }
                | Message::GitUnstage { .. }
                | Message::GitDiscard { .. }
                | Message::GitCommit { .. }
                | Message::GitFetch { .. }
                | Message::GitPull { .. }
                | Message::GitPush { .. }
                | Message::GitRevert { .. }
                | Message::GitGenerateCommitMessageAgent { .. }
                | Message::GitAnalyzeDiff { .. }
                | Message::GitBranches { .. }
                | Message::GitStashList { .. }
                | Message::GitStashPush { .. }
                | Message::GitStashApply { .. }
                | Message::GitStashPop { .. }
                | Message::GitStashDrop { .. }
                | Message::GitSubmoduleList { .. }
                | Message::GitCheckout { .. }
                | Message::GitCreateBranch { .. }
                | Message::GitCreateWorktree { .. }
                | Message::GitDeleteBranch { .. }
                | Message::GitMerge { .. }
                // Interactive rebase requests are session-scoped like the rest
                // of the git family; they must ride the request channel so a
                // session living on a shared server is routed there instead of
                // hitting the local server's session check and being dropped.
                | Message::GitRebaseStart { .. }
                | Message::GitRebaseApply { .. }
                | Message::GitRebaseConflictContent { .. }
                | Message::GitRebaseResolve { .. }
                | Message::GitRebaseContinue { .. }
                | Message::GitRebaseSkip { .. }
                | Message::GitRebaseAbort { .. }
                | Message::GitRebaseAiResolve { .. }
                | Message::GitRebaseAiResolveCancel { .. }
                | Message::FetchFile { .. }
                | Message::SaveFile { .. }
                | Message::TempFileUploadStart { .. }
                | Message::UploadStart { .. }
                | Message::DownloadStart { .. }
                | Message::DebugLaunchList { .. }
                | Message::DebugStart { .. }
                | Message::QuickCommandList { .. }
                | Message::RunQuickCommand { .. }
                | Message::FetchHistoryRaw { .. }
                | Message::ExportSessionRawHistory { .. }
                | Message::TranscribeAudio { .. }
                | Message::PolishVoiceInputText { .. }
                | Message::CreatePath { .. }
                | Message::DeletePaths { .. }
                | Message::PastePaths { .. }
                | Message::RenamePath { .. }
                | Message::GenerateVtermConfig { .. }
                | Message::StartStaticHttpServer { .. }
                | Message::RequestSharedPortForward { .. }
                | Message::ListScreenSources { .. }
                | Message::StartScreenStream { .. }
                | Message::StopScreenStream { .. }
                | Message::BootSimulator { .. }
                | Message::ShutdownSimulator { .. }
                | Message::BootAndroidEmulator { .. }
                | Message::ShutdownAndroidEmulator { .. }
                | Message::RotateAndroidDevice { .. }
                | Message::ApplyEnvironment { .. }
                | Message::SetNotificationEnabled { .. }
                | Message::SetAutoConfirmEnabled { .. }
                | Message::SetSessionEncoding { .. }
                | Message::ClearSessionScreen { .. }
                | Message::ConfirmSession { .. }
                | Message::SetupMcpApp { .. }
        )
    }

    pub fn is_request_response(&self) -> bool {
        matches!(
            self,
            Message::PathContentList { .. }
                | Message::FindFilesResult { .. }
                | Message::SearchPathNamesResult { .. }
                | Message::TabInfoList { .. }
                | Message::RecentProjectDirectoryList { .. }
                | Message::AiConversationList { .. }
                | Message::AiConversationMessages { .. }
                | Message::TranslateMessageResult { .. }
                | Message::GitChangedFilesList { .. }
                | Message::GitLogList { .. }
                | Message::GitDiffResult { .. }
                | Message::GitStageResult { .. }
                | Message::GitUnstageResult { .. }
                | Message::GitDiscardResult { .. }
                | Message::GitCommitResult { .. }
                | Message::GitFetchResult { .. }
                | Message::GitPullResult { .. }
                | Message::GitPushResult { .. }
                | Message::GitRevertResult { .. }
                | Message::GitGenerateCommitMessageAccepted { .. }
                | Message::GitGenerateCommitMessageResult { .. }
                | Message::GitAnalyzeDiffResult { .. }
                | Message::GitBranchesResult { .. }
                | Message::GitStashListResult { .. }
                | Message::GitStashResult { .. }
                | Message::GitSubmoduleListResult { .. }
                | Message::GitCheckoutResult { .. }
                | Message::GitCreateBranchResult { .. }
                | Message::GitCreateWorktreeResult { .. }
                | Message::GitDeleteBranchResult { .. }
                | Message::GitMergeResult { .. }
                | Message::GitRebaseStartResult { .. }
                | Message::GitRebaseApplyResult { .. }
                | Message::GitRebaseConflictContentResult { .. }
                | Message::GitRebaseResolveResult { .. }
                | Message::GitRebaseContinueResult { .. }
                | Message::GitRebaseSkipResult { .. }
                | Message::GitRebaseAbortResult { .. }
                | Message::GitRebaseAiResolveResult { .. }
                | Message::FileContent { .. }
                | Message::SaveFileResult { .. }
                | Message::TempFileUploadResult { .. }
                | Message::UploadResult { .. }
                | Message::DownloadResult { .. }
                | Message::CreatePathResult { .. }
                | Message::DeletePathsResult { .. }
                | Message::PastePathsResult { .. }
                | Message::RenamePathResult { .. }
                | Message::GenerateVtermConfigResult { .. }
                | Message::StartStaticHttpServerResult { .. }
                | Message::SharedPortForwardReady { .. }
                | Message::DebugLaunchListResult { .. }
                | Message::DebugStartResult { .. }
                | Message::DebugOutput { .. }
                | Message::QuickCommandListResult { .. }
                | Message::RunQuickCommandResult { .. }
                | Message::HistoryRawChunk { .. }
                | Message::ExportSessionRawHistoryResult { .. }
                | Message::TranscribeAudioResult { .. }
                | Message::PolishVoiceInputTextResult { .. }
                | Message::Error { .. }
                | Message::ScreenSourcesResult { .. }
                | Message::ScreenStreamReady { .. }
                | Message::ScreenStreamStopped { .. }
                | Message::BootSimulatorResult { .. }
                | Message::ShutdownSimulatorResult { .. }
                | Message::BootAndroidEmulatorResult { .. }
                | Message::ShutdownAndroidEmulatorResult { .. }
                | Message::RotateAndroidDeviceResult { .. }
                | Message::ApplyEnvironmentResult { .. }
                | Message::SetNotificationEnabledResult { .. }
                | Message::SetAutoConfirmEnabledResult { .. }
                | Message::SetSessionEncodingResult { .. }
                | Message::ClearSessionScreenResult { .. }
                | Message::ConfirmSessionResult { .. }
                | Message::SetupMcpAppResult { .. }
        )
    }

    pub fn expected_request_response(&self, response: &Message) -> bool {
        if matches!(response, Message::Error { .. }) {
            return true;
        }

        matches!(
            (self, response),
            (
                Message::UploadStart { session_id, .. },
                Message::UploadResult {
                    session_id: response_session_id,
                    ..
                }
            )
            | (
                Message::DownloadStart { session_id, .. },
                Message::DownloadResult {
                    session_id: response_session_id,
                    ..
                }
            )
            | (
                Message::ListPathContent { session_id, .. },
                Message::PathContentList {
                    session_id: response_session_id,
                    ..
                }
            ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::TempFileUploadStart {
                    task_id,
                    session_id,
                    ..
                },
                Message::TempFileUploadResult {
                    task_id: response_task_id,
                    session_id: response_session_id,
                    ..
                }
            ) if task_id == response_task_id && session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::FindFiles {
                    session_id,
                    request_id,
                    ..
                },
                Message::FindFilesResult {
                    session_id: response_session_id,
                    request_id: response_request_id,
                    ..
                }
            ) if session_id == response_session_id && request_id == response_request_id
        ) || matches!(
            (self, response),
            (
                Message::SearchPathNames {
                    session_id,
                    request_id,
                    ..
                },
                Message::SearchPathNamesResult {
                    session_id: response_session_id,
                    request_id: response_request_id,
                    ..
                }
            ) if session_id == response_session_id && request_id == response_request_id
        ) || matches!(
            (self, response),
            (
                Message::GetTabInfoList { server_id },
                Message::TabInfoList {
                    server_id: response_server_id,
                    ..
                }
            ) if server_id == response_server_id
        ) || matches!(
            (self, response),
            (
                Message::RemoveRecentProjectDirectory { .. },
                Message::RecentProjectDirectoryList { .. }
            )
        ) || matches!(
            (self, response),
            (
                Message::ListAiConversations { session_id },
                Message::AiConversationList {
                    session_id: response_session_id,
                    ..
                }
            ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::ReadAiConversation {
                    session_id,
                    conversation_id,
                },
                Message::AiConversationMessages {
                    session_id: response_session_id,
                    conversation_id: response_conversation_id,
                    ..
                }
            ) if session_id == response_session_id && conversation_id == response_conversation_id
        ) || matches!(
            (self, response),
            (
                Message::TranslateMessage {
                    window_id,
                    component_id,
                    session_id,
                    message_id,
                    ..
                },
                Message::TranslateMessageResult {
                    window_id: response_window_id,
                    component_id: response_component_id,
                    session_id: response_session_id,
                    message_id: response_message_id,
                    ..
                }
            ) if window_id == response_window_id
                && component_id == response_component_id
                && session_id == response_session_id
                && message_id == response_message_id
        ) || matches!(
            (self, response),
            (
                Message::DebugLaunchList { session_id },
                Message::DebugLaunchListResult {
                    session_id: response_session_id,
                    ..
                }
            ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::DebugStart {
                    session_id,
                    launch_name,
                },
                Message::DebugStartResult {
                    session_id: response_session_id,
                    launch_name: response_launch_name,
                    ..
                }
            ) if session_id == response_session_id && launch_name == response_launch_name
        ) || matches!(
            (self, response),
            (
                Message::QuickCommandList { session_id },
                Message::QuickCommandListResult {
                    session_id: response_session_id,
                    ..
                }
            ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::RunQuickCommand { session_id, name },
                Message::RunQuickCommandResult {
                    session_id: response_session_id,
                    name: response_name,
                    ..
                }
            ) if session_id == response_session_id && name == response_name
        ) || matches!(
            (self, response),
            (
                Message::SetupMcpApp { session_id, .. },
                Message::SetupMcpAppResult {
                    session_id: response_session_id,
                    ..
                }
            ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::GitChangedFiles {
                    session_id, commit, ..
                },
                Message::GitChangedFilesList {
                    session_id: response_session_id,
                    commit: response_commit,
                    ..
                }
            ) if session_id == response_session_id && commit == response_commit
        ) || matches!(
            (self, response),
            (
                Message::GitLog {
                    session_id,
                    commit,
                    branch,
                    ..
                },
                Message::GitLogList {
                    session_id: response_session_id,
                    commit: response_commit,
                    branch: response_branch,
                    ..
                }
            ) if session_id == response_session_id && commit == response_commit && branch == response_branch
        ) || matches!(
            (self, response),
            (
                Message::GitDiff {
                    session_id, path, ..
                },
                Message::GitDiffResult {
                    session_id: response_session_id,
                    path: response_path,
                    ..
                }
            ) if session_id == response_session_id && path == response_path
        ) || matches!(
            (self, response),
            (
                Message::GitStage { session_id, .. },
                Message::GitStageResult {
                    session_id: response_session_id,
                    ..
                }
            )
                | (
                    Message::GitUnstage { session_id, .. },
                    Message::GitUnstageResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitDiscard { session_id, .. },
                    Message::GitDiscardResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitCommit { session_id, .. },
                    Message::GitCommitResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitFetch { session_id },
                    Message::GitFetchResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitPull { session_id },
                    Message::GitPullResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitPush { session_id, .. },
                    Message::GitPushResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitRevert { session_id, .. },
                    Message::GitRevertResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitGenerateCommitMessageAgent { session_id },
                    Message::GitGenerateCommitMessageAccepted {
                        session_id: response_session_id,
                    }
                )
                | (
                    Message::GitAnalyzeDiff { session_id, .. },
                    Message::GitAnalyzeDiffResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitRebaseAiResolve { session_id, .. },
                    Message::GitRebaseAiResolveResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitBranches { session_id },
                    Message::GitBranchesResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitStashList { session_id },
                    Message::GitStashListResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitStashPush { session_id, .. },
                    Message::GitStashResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitStashApply { session_id, .. },
                    Message::GitStashResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitStashPop { session_id, .. },
                    Message::GitStashResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitStashDrop { session_id, .. },
                    Message::GitStashResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitSubmoduleList { session_id },
                    Message::GitSubmoduleListResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitCheckout { session_id, .. },
                    Message::GitCheckoutResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitCreateBranch { session_id, .. },
                    Message::GitCreateBranchResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitCreateWorktree { session_id, .. },
                    Message::GitCreateWorktreeResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitDeleteBranch { session_id, .. },
                    Message::GitDeleteBranchResult {
                        session_id: response_session_id,
                        ..
                    }
                )
                | (
                    Message::GitMerge { session_id, .. },
                    Message::GitMergeResult {
                        session_id: response_session_id,
                        ..
                    }
                ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::GitRebaseStart { session_id, .. },
                Message::GitRebaseStartResult {
                    session_id: response_session_id,
                    ..
                }
            ) | (
                Message::GitRebaseApply { session_id, .. },
                Message::GitRebaseApplyResult {
                    session_id: response_session_id,
                    ..
                }
            ) | (
                Message::GitRebaseConflictContent { session_id, .. },
                Message::GitRebaseConflictContentResult {
                    session_id: response_session_id,
                    ..
                }
            ) | (
                Message::GitRebaseResolve { session_id, .. },
                Message::GitRebaseResolveResult {
                    session_id: response_session_id,
                    ..
                }
            ) | (
                Message::GitRebaseContinue { session_id, .. },
                Message::GitRebaseContinueResult {
                    session_id: response_session_id,
                    ..
                }
            ) | (
                Message::GitRebaseSkip { session_id, .. },
                Message::GitRebaseSkipResult {
                    session_id: response_session_id,
                    ..
                }
            ) | (
                Message::GitRebaseAbort { session_id, .. },
                Message::GitRebaseAbortResult {
                    session_id: response_session_id,
                    ..
                }
            ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::FetchHistoryRaw { session_id, .. },
                Message::HistoryRawChunk {
                    session_id: response_session_id,
                    ..
                }
            ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::ExportSessionRawHistory { session_id },
                Message::ExportSessionRawHistoryResult {
                    session_id: response_session_id,
                    ..
                }
            ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::FetchFile {
                    session_id, path, ..
                },
                Message::FileContent {
                    session_id: response_session_id,
                    path: response_path,
                    ..
                }
            ) if session_id == response_session_id && path == response_path
        ) || matches!(
            (self, response),
            (
                Message::SaveFile { session_id, path, .. },
                Message::SaveFileResult {
                    session_id: response_session_id,
                    path: response_path,
                    ..
                }
            ) if session_id == response_session_id && path == response_path
        ) || matches!(
            (self, response),
            (
                Message::CreatePath {
                    session_id, path, ..
                },
                Message::CreatePathResult {
                    session_id: response_session_id,
                    path: response_path,
                    ..
                }
            ) if session_id == response_session_id && path == response_path
        ) || matches!(
            (self, response),
            (
                Message::DeletePaths { session_id, .. },
                Message::DeletePathsResult {
                    session_id: response_session_id,
                    ..
                }
            ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::PastePaths { session_id, .. },
                Message::PastePathsResult {
                    session_id: response_session_id,
                    ..
                }
            ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::RenamePath { session_id, .. },
                Message::RenamePathResult {
                    session_id: response_session_id,
                    ..
                }
            ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::GenerateVtermConfig {
                    vterm_path,
                    config_type,
                    ..
                },
                Message::GenerateVtermConfigResult {
                    vterm_path: response_path,
                    config_type: response_type,
                    ..
                }
            ) if vterm_path == response_path && config_type == response_type
        ) || matches!(
            (self, response),
            (
                Message::StartStaticHttpServer { session_id, path, .. },
                Message::StartStaticHttpServerResult { session_id: response_session_id, path: response_path, .. }
            ) if session_id == response_session_id && path == response_path
        ) || matches!(
            (self, response),
            (
                Message::RequestSharedPortForward { session_id, target_port, .. },
                Message::SharedPortForwardReady { session_id: response_session_id, target_port: response_target_port, .. }
            ) if session_id == response_session_id && target_port == response_target_port
        ) || matches!(
            (self, response),
            (
                Message::TranscribeAudio { .. },
                Message::TranscribeAudioResult { .. }
            )
        ) || matches!(
            (self, response),
            (
                Message::PolishVoiceInputText {
                    window_id,
                    component_id,
                    request_id,
                    context,
                    ..
                },
                Message::PolishVoiceInputTextResult {
                    window_id: response_window_id,
                    component_id: response_component_id,
                    session_id: response_session_id,
                    request_id: response_request_id,
                    ..
                }
            ) if window_id == response_window_id
                && component_id == response_component_id
                && context.session_id() == *response_session_id
                && request_id == response_request_id
        ) || matches!(
            (self, response),
            (
                Message::ApplyEnvironment {
                    session_id,
                    environment_id,
                    ..
                },
                Message::ApplyEnvironmentResult {
                    session_id: response_session_id,
                    environment_id: response_environment_id,
                    ..
                }
            ) if session_id == response_session_id && environment_id == response_environment_id
        ) || matches!(
            (self, response),
            (
                Message::SetNotificationEnabled { session_id, .. },
                Message::SetNotificationEnabledResult {
                    session_id: response_session_id,
                    ..
                }
            ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::SetAutoConfirmEnabled { session_id, .. },
                Message::SetAutoConfirmEnabledResult {
                    session_id: response_session_id,
                    ..
                }
            ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::SetSessionEncoding { session_id, .. },
                Message::SetSessionEncodingResult {
                    session_id: response_session_id,
                    ..
                }
            ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::ClearSessionScreen { session_id },
                Message::ClearSessionScreenResult {
                    session_id: response_session_id,
                    ..
                }
            ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::ConfirmSession { session_id },
                Message::ConfirmSessionResult {
                    session_id: response_session_id,
                    ..
                }
            ) if session_id == response_session_id
        ) || matches!(
            (self, response),
            (
                Message::ListScreenSources { server_id },
                Message::ScreenSourcesResult {
                    server_id: response_server_id,
                    ..
                }
            ) if server_id == response_server_id
        ) || matches!(
            (self, response),
            (
                Message::StartScreenStream { source, server_id },
                Message::ScreenStreamReady {
                    source: response_source,
                    server_id: response_server_id,
                    ..
                }
            ) if source == response_source && server_id == response_server_id
        ) || matches!(
            (self, response),
            (
                Message::StopScreenStream { source, server_id },
                Message::ScreenStreamStopped {
                    source: response_source,
                    server_id: response_server_id,
                    ..
                }
            ) if source == response_source && server_id == response_server_id
        ) || matches!(
            (self, response),
            (
                Message::BootSimulator { spec, server_id },
                Message::BootSimulatorResult {
                    spec: response_spec,
                    server_id: response_server_id,
                    ..
                }
            ) if spec == response_spec && server_id == response_server_id
        ) || matches!(
            (self, response),
            (
                Message::ShutdownSimulator { spec, server_id },
                Message::ShutdownSimulatorResult {
                    spec: response_spec,
                    server_id: response_server_id,
                    ..
                }
            ) if spec == response_spec && server_id == response_server_id
        ) || matches!(
            (self, response),
            (
                Message::BootAndroidEmulator { avd, server_id },
                Message::BootAndroidEmulatorResult {
                    avd: response_avd,
                    server_id: response_server_id,
                    ..
                }
            ) if avd == response_avd && server_id == response_server_id
        ) || matches!(
            (self, response),
            (
                Message::ShutdownAndroidEmulator { serial, server_id },
                Message::ShutdownAndroidEmulatorResult {
                    serial: response_serial,
                    server_id: response_server_id,
                    ..
                }
            ) if serial == response_serial && server_id == response_server_id
        )
    }

    pub fn completes_request_response(&self, response: &Message) -> bool {
        if !self.expected_request_response(response) && !response.is_request_response() {
            return false;
        }
        match response {
            Message::FindFilesResult {
                is_final, error, ..
            } => *is_final || error.is_some(),
            Message::GitAnalyzeDiffResult { done, error, .. } => *done || error.is_some(),
            _ => true,
        }
    }
}

impl TransportBinaryFrameDecoder {
    pub fn decode_message_frame(&mut self, data: &[u8]) -> anyhow::Result<Option<Message>> {
        let payload = match self.decode_payload(data)? {
            Some(payload) => payload,
            None => return Ok(None),
        };
        Message::from_transport_binary(&payload).map(Some)
    }

    fn decode_payload(&mut self, data: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
        if !data.starts_with(CHUNKED_BINARY_FRAME_MAGIC) {
            if self.pending.is_some() {
                anyhow::bail!(
                    "Received non-chunked binary frame while a chunked message is pending"
                );
            }
            return Ok(Some(data.to_vec()));
        }

        if data.len() < CHUNKED_BINARY_FRAME_HEADER_LEN {
            anyhow::bail!(
                "Chunked binary frame too short: expected at least {} bytes, got {}",
                CHUNKED_BINARY_FRAME_HEADER_LEN,
                data.len()
            );
        }

        let flags = data[4];
        let mut total_len_bytes = [0u8; 8];
        total_len_bytes.copy_from_slice(&data[5..13]);
        let total_len = usize::try_from(u64::from_le_bytes(total_len_bytes)).map_err(|_| {
            anyhow::anyhow!("Chunked binary frame total length exceeds platform size")
        })?;

        let mut offset_bytes = [0u8; 8];
        offset_bytes.copy_from_slice(&data[13..21]);
        let offset = usize::try_from(u64::from_le_bytes(offset_bytes))
            .map_err(|_| anyhow::anyhow!("Chunked binary frame offset exceeds platform size"))?;

        let chunk = &data[21..];
        let is_start = flags & CHUNK_FLAG_START != 0;
        let is_end = flags & CHUNK_FLAG_END != 0;

        if is_start {
            if offset != 0 {
                anyhow::bail!(
                    "Chunked binary frame start offset must be 0, got {}",
                    offset
                );
            }
            if self.pending.is_some() {
                anyhow::bail!(
                    "Received chunked binary frame start while another message is pending"
                );
            }
            if chunk.len() > total_len {
                anyhow::bail!(
                    "Chunked binary frame start payload exceeds declared length: {} > {}",
                    chunk.len(),
                    total_len
                );
            }

            let mut data = Vec::with_capacity(total_len);
            data.extend_from_slice(chunk);
            let received_len = chunk.len();
            if is_end {
                if received_len != total_len {
                    anyhow::bail!(
                        "Chunked binary frame ended early: received {} of {} bytes",
                        received_len,
                        total_len
                    );
                }
                return Ok(Some(data));
            }

            self.pending = Some(PendingBinaryFrame {
                total_len,
                received_len,
                data,
            });
            return Ok(None);
        }

        let pending = self.pending.as_mut().ok_or_else(|| {
            anyhow::anyhow!("Received chunk continuation without a pending chunked message")
        })?;

        if pending.total_len != total_len {
            anyhow::bail!(
                "Chunked binary frame total length changed mid-stream: {} != {}",
                total_len,
                pending.total_len
            );
        }
        if pending.received_len != offset {
            anyhow::bail!(
                "Chunked binary frame offset mismatch: expected {}, got {}",
                pending.received_len,
                offset
            );
        }
        if pending.received_len + chunk.len() > pending.total_len {
            anyhow::bail!(
                "Chunked binary frame exceeds declared length: {} + {} > {}",
                pending.received_len,
                chunk.len(),
                pending.total_len
            );
        }

        pending.data.extend_from_slice(chunk);
        pending.received_len += chunk.len();

        if !is_end {
            return Ok(None);
        }

        let pending = self.pending.take().expect("pending chunk must exist");
        if pending.received_len != pending.total_len {
            anyhow::bail!(
                "Chunked binary frame ended early: received {} of {} bytes",
                pending.received_len,
                pending.total_len
            );
        }
        Ok(Some(pending.data))
    }
}

pub fn chunk_transport_binary_payload(payload: &[u8]) -> Vec<Vec<u8>> {
    if payload.len() <= MAX_BINARY_TRANSPORT_CHUNK_SIZE {
        return vec![payload.to_vec()];
    }

    let total_len = payload.len();
    let mut frames = Vec::new();
    let mut offset = 0usize;
    while offset < total_len {
        let end = (offset + MAX_BINARY_TRANSPORT_CHUNK_SIZE).min(total_len);
        let mut frame = Vec::with_capacity(CHUNKED_BINARY_FRAME_HEADER_LEN + end - offset);
        frame.extend_from_slice(CHUNKED_BINARY_FRAME_MAGIC);

        let mut flags = 0u8;
        if offset == 0 {
            flags |= CHUNK_FLAG_START;
        }
        if end == total_len {
            flags |= CHUNK_FLAG_END;
        }
        frame.push(flags);
        frame.extend_from_slice(&(total_len as u64).to_le_bytes());
        frame.extend_from_slice(&(offset as u64).to_le_bytes());
        frame.extend_from_slice(&payload[offset..end]);
        frames.push(frame);
        offset = end;
    }
    frames
}

fn append_u32(frame: &mut Vec<u8>, value: usize) -> Option<()> {
    let value = u32::try_from(value).ok()?;
    frame.extend_from_slice(&value.to_le_bytes());
    Some(())
}

fn append_string(frame: &mut Vec<u8>, value: &str) -> Option<()> {
    append_u32(frame, value.len())?;
    frame.extend_from_slice(value.as_bytes());
    Some(())
}

fn append_optional_string(frame: &mut Vec<u8>, value: Option<&str>) -> Option<()> {
    match value {
        Some(value) => {
            frame.push(1);
            append_string(frame, value)?;
        }
        None => frame.push(0),
    }
    Some(())
}

fn append_lines(frame: &mut Vec<u8>, lines: &[String]) -> Option<()> {
    append_u32(frame, lines.len())?;
    for line in lines {
        append_string(frame, line)?;
    }
    Some(())
}

fn git_path_op_binary_frame(
    magic: &[u8; 4],
    session_id: SessionId,
    paths: &[String],
    repo_root: Option<&str>,
) -> Option<Vec<u8>> {
    let mut frame = Vec::new();
    frame.extend_from_slice(magic);
    frame.extend_from_slice(session_id.as_bytes());
    append_u32(&mut frame, paths.len())?;
    for path in paths {
        append_string(&mut frame, path)?;
    }
    // Trailing optional field: old decoders stop after the paths and ignore
    // it; new decoders read it when present (absent on old-client frames).
    append_optional_string(&mut frame, repo_root)?;
    Some(frame)
}

fn read_binary_session_header(data: &[u8], magic: &[u8; 4]) -> anyhow::Result<(SessionId, usize)> {
    let header_len = magic.len() + 16;
    if data.len() < header_len {
        anyhow::bail!(
            "Binary frame too short: expected at least {} bytes, got {}",
            header_len,
            data.len()
        );
    }
    if !data.starts_with(magic) {
        anyhow::bail!("Invalid binary frame magic");
    }

    let mut session_id_bytes = [0u8; 16];
    session_id_bytes.copy_from_slice(&data[4..20]);
    Ok((SessionId::from_bytes(session_id_bytes), 20))
}

fn read_u8(data: &[u8], offset: &mut usize) -> anyhow::Result<u8> {
    if *offset >= data.len() {
        anyhow::bail!("Binary frame ended while reading byte");
    }
    let value = data[*offset];
    *offset += 1;
    Ok(value)
}

fn read_u32(data: &[u8], offset: &mut usize) -> anyhow::Result<u32> {
    if data.len().saturating_sub(*offset) < 4 {
        anyhow::bail!("Binary frame ended while reading u32");
    }
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&data[*offset..*offset + 4]);
    *offset += 4;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32(data: &[u8], offset: &mut usize) -> anyhow::Result<i32> {
    if data.len().saturating_sub(*offset) < 4 {
        anyhow::bail!("Binary frame ended while reading i32");
    }
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&data[*offset..*offset + 4]);
    *offset += 4;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u64(data: &[u8], offset: &mut usize) -> anyhow::Result<u64> {
    if data.len().saturating_sub(*offset) < 8 {
        anyhow::bail!("Binary frame ended while reading u64");
    }
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&data[*offset..*offset + 8]);
    *offset += 8;
    Ok(u64::from_le_bytes(bytes))
}

fn read_string(data: &[u8], offset: &mut usize) -> anyhow::Result<String> {
    let len = read_u32(data, offset)? as usize;
    if data.len().saturating_sub(*offset) < len {
        anyhow::bail!("Binary frame ended while reading string");
    }
    let value = std::str::from_utf8(&data[*offset..*offset + len])?.to_string();
    *offset += len;
    Ok(value)
}

fn read_optional_string(data: &[u8], offset: &mut usize) -> anyhow::Result<Option<String>> {
    match read_u8(data, offset)? {
        0 => Ok(None),
        1 => read_string(data, offset).map(Some),
        other => anyhow::bail!("Invalid optional string marker {}", other),
    }
}

fn read_f32(data: &[u8], offset: &mut usize) -> anyhow::Result<f32> {
    if data.len().saturating_sub(*offset) < 4 {
        anyhow::bail!("Binary frame ended while reading f32");
    }
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&data[*offset..*offset + 4]);
    *offset += 4;
    Ok(f32::from_le_bytes(bytes))
}

fn read_lines(data: &[u8], offset: &mut usize) -> anyhow::Result<Vec<String>> {
    let count = read_u32(data, offset)? as usize;
    let mut lines = Vec::with_capacity(count);
    for _ in 0..count {
        lines.push(read_string(data, offset)?);
    }
    Ok(lines)
}

fn git_status_to_u8(status: &GitFileStatus) -> u8 {
    match status {
        GitFileStatus::Modified => 1,
        GitFileStatus::Added => 2,
        GitFileStatus::Deleted => 3,
        GitFileStatus::Untracked => 4,
        GitFileStatus::Conflict => 5,
    }
}

fn git_status_from_u8(status: u8) -> anyhow::Result<GitFileStatus> {
    match status {
        1 => Ok(GitFileStatus::Modified),
        2 => Ok(GitFileStatus::Added),
        3 => Ok(GitFileStatus::Deleted),
        4 => Ok(GitFileStatus::Untracked),
        5 => Ok(GitFileStatus::Conflict),
        other => anyhow::bail!("Unknown git file status {}", other),
    }
}

fn ai_role_to_u8(role: &AiConversationRole) -> u8 {
    match role {
        AiConversationRole::User => 1,
        AiConversationRole::Model => 2,
    }
}

fn ai_role_from_u8(role: u8) -> anyhow::Result<AiConversationRole> {
    match role {
        1 => Ok(AiConversationRole::User),
        2 => Ok(AiConversationRole::Model),
        other => anyhow::bail!("Unknown AI conversation role {}", other),
    }
}

fn ai_message_kind_to_u8(kind: &AiConversationMessageKind) -> u8 {
    match kind {
        AiConversationMessageKind::Text => 1,
        AiConversationMessageKind::Thinking => 2,
    }
}

fn ai_message_kind_from_u8(kind: u8) -> anyhow::Result<AiConversationMessageKind> {
    match kind {
        1 => Ok(AiConversationMessageKind::Text),
        2 => Ok(AiConversationMessageKind::Thinking),
        other => anyhow::bail!("Unknown AI conversation message kind {}", other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_http_forward_flags_are_backward_compatible() {
        let session_id = SessionId::new();
        let legacy_request = serde_json::json!({
            "StartStaticHttpServer": {
                "session_id": session_id,
                "path": "/tmp/site"
            }
        })
        .to_string();
        let request = Message::from_text(&legacy_request).unwrap();
        assert!(matches!(
            &request,
            Message::StartStaticHttpServer {
                request_port_forward: false,
                ..
            }
        ));

        let legacy_response = serde_json::json!({
            "StartStaticHttpServerResult": {
                "session_id": session_id,
                "path": "/tmp/site",
                "port": 8080
            }
        })
        .to_string();
        let response = Message::from_text(&legacy_response).unwrap();
        assert!(matches!(
            &response,
            Message::StartStaticHttpServerResult {
                port_forwarded: false,
                ..
            }
        ));
        assert!(request.expected_request_response(&response));

        let forwarded_response = Message::StartStaticHttpServerResult {
            session_id,
            path: "/tmp/site".to_string(),
            port: Some(62000),
            url: None,
            share_token: Some("share-secret".to_string()),
            port_forwarded: true,
            error: None,
        };
        let encoded = forwarded_response.to_text().unwrap();
        assert!(matches!(
            Message::from_text(&encoded).unwrap(),
            Message::StartStaticHttpServerResult {
                port: Some(62000),
                share_token: Some(ref token),
                port_forwarded: true,
                ..
            } if token == "share-secret"
        ));
    }

    #[test]
    fn reverse_proxy_settings_fields_round_trip() {
        let request = Message::RequestReverseProxy {
            target_port: 8780,
            auth_token: Some("primary".to_string()),
            purpose: Some(ReverseProxyPurpose::ServerSettings),
        };
        let encoded = request.to_text().unwrap();
        assert!(matches!(
            Message::from_text(&encoded).unwrap(),
            Message::RequestReverseProxy {
                purpose: Some(ReverseProxyPurpose::ServerSettings),
                ..
            }
        ));

        let response = Message::ReverseProxyReady {
            target_port: 8780,
            proxy_port: 62000,
            reused_existing: false,
            settings_ticket: Some("ticket".to_string()),
            error: None,
        };
        let encoded = response.to_text().unwrap();
        assert!(matches!(
            Message::from_text(&encoded).unwrap(),
            Message::ReverseProxyReady {
                settings_ticket: Some(ref ticket),
                ..
            } if ticket == "ticket"
        ));
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::CreateSession {
            shell: "/bin/bash".to_string(),
            profile_id: Some("default-shell".to_string()),
            size: TerminalSize::new(24, 80),
            env: HashMap::from([(String::from("TERM"), String::from("xterm-256color"))]),
            cwd: Some("/tmp".to_string()),
            title: None,
            sibling_session_id: None,
            server_id: None,
            tab_group_id: None,
            duplicate_session_id: None,
            app: None,
            background: Some([0xf5, 0xf5, 0xf7]),
            foreground: Some([0x1d, 0x1d, 0x1f]),
        };

        let serialized = msg.serialize().unwrap();
        let deserialized = Message::deserialize(&serialized).unwrap();

        match deserialized {
            Message::CreateSession {
                shell,
                profile_id,
                size,
                env,
                cwd,
                title,
                sibling_session_id,
                server_id,
                tab_group_id,
                duplicate_session_id,
                app,
                background,
                foreground,
            } => {
                assert_eq!(shell, "/bin/bash");
                assert_eq!(profile_id.as_deref(), Some("default-shell"));
                assert_eq!(size.rows, 24);
                assert_eq!(size.cols, 80);
                assert_eq!(env.get("TERM").map(String::as_str), Some("xterm-256color"));
                assert_eq!(cwd.as_deref(), Some("/tmp"));
                assert!(title.is_none());
                assert!(sibling_session_id.is_none());
                assert!(server_id.is_none());
                assert!(tab_group_id.is_none());
                assert!(duplicate_session_id.is_none());
                assert!(app.is_none());
                assert_eq!(background, Some([0xf5, 0xf5, 0xf7]));
                assert_eq!(foreground, Some([0x1d, 0x1d, 0x1f]));
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_session_id_extraction() {
        let session_id = SessionId::new();
        let msg = Message::Output {
            session_id,
            start_offset: 12,
            data: vec![1, 2, 3],
        };

        assert_eq!(msg.session_id(), Some(session_id));

        let ping = Message::Ping;
        assert_eq!(ping.session_id(), None);
    }

    #[test]
    fn test_screen_streaming_messages_roundtrip() {
        let messages = vec![
            Message::ListScreenSources { server_id: None },
            Message::ScreenSourcesResult {
                sources: vec![
                    ScreenSourceInfo {
                        spec: "main".to_string(),
                        kind: ScreenSourceKind::Screen,
                        title: "Main display".to_string(),
                        status: ScreenSourceStatus::Available,
                        width: Some(2560),
                        height: Some(1440),
                        primary: true,
                    },
                    ScreenSourceInfo {
                        spec: "sim:ABCD-1234".to_string(),
                        kind: ScreenSourceKind::IosSimulator,
                        title: "iPhone 16".to_string(),
                        status: ScreenSourceStatus::Unavailable {
                            reason: "simulator shut down".to_string(),
                        },
                        width: None,
                        height: None,
                        primary: false,
                    },
                ],
                error: None,
                server_id: Some("shared-1".to_string()),
                revision: Some(1),
            },
            Message::ScreenSourcesResult {
                sources: vec![],
                error: Some("screen-server unreachable".to_string()),
                server_id: None,
                revision: None,
            },
            Message::ScreenSourcesChanged {
                sources: vec![],
                server_id: Some("shared-1".to_string()),
                revision: 2,
            },
            Message::StartScreenStream {
                source: "main".to_string(),
                server_id: None,
            },
            Message::ScreenStreamReady {
                source: "main".to_string(),
                server_id: None,
                endpoint: StreamEndpoint::Direct {
                    host: String::new(),
                    port: 36459,
                },
                token: "0123456789abcdef0123456789abcdef".to_string(),
                codec: "h264".to_string(),
                format: "nv12".to_string(),
                error: None,
            },
            Message::ScreenStreamReady {
                source: "android:emulator-5554".to_string(),
                server_id: Some("shared-1".to_string()),
                endpoint: StreamEndpoint::Relay {
                    host: "relay.example.com".to_string(),
                    port: 40000,
                    pairing: "pairing-token".to_string(),
                },
                token: String::new(),
                codec: "hevc".to_string(),
                format: "p010".to_string(),
                error: Some("encoder busy".to_string()),
            },
            Message::StopScreenStream {
                source: "main".to_string(),
                server_id: None,
            },
            Message::ScreenStreamStopped {
                source: "main".to_string(),
                server_id: None,
                error: None,
            },
            Message::ScreenStreamStopped {
                source: "main".to_string(),
                server_id: Some("shared-1".to_string()),
                error: Some("teardown failed".to_string()),
            },
            Message::ScreenStreamNotify {
                source: "main".to_string(),
                event: ScreenStreamEvent::StreamDied {
                    message: "encoder crashed".to_string(),
                },
            },
            Message::ScreenStreamNotify {
                source: "display:1".to_string(),
                event: ScreenStreamEvent::SourceUnavailable {
                    reason: "monitor unplugged".to_string(),
                },
            },
            Message::ScreenStreamStats {
                frames: 300,
                avg_pickup_us: 1200,
                avg_encode_ms: 2.6,
                delivered: 298,
                dropped: 2,
                avg_net_ms: 4.1,
            },
            Message::BootSimulator {
                spec: "sim:ABCD-1234".to_string(),
                server_id: Some("shared-1".to_string()),
            },
            Message::BootSimulatorResult {
                spec: "sim:ABCD-1234".to_string(),
                server_id: Some("shared-1".to_string()),
                error: None,
            },
            Message::ShutdownSimulator {
                spec: "sim:ABCD-1234".to_string(),
                server_id: None,
            },
            Message::ShutdownSimulatorResult {
                spec: "sim:ABCD-1234".to_string(),
                server_id: None,
                error: None,
            },
            Message::BootAndroidEmulator {
                avd: "Medium_Phone_API_36.0".to_string(),
                server_id: Some("shared-1".to_string()),
            },
            Message::BootAndroidEmulatorResult {
                avd: "Medium_Phone_API_36.0".to_string(),
                server_id: Some("shared-1".to_string()),
                operation_id: Some(7),
                error: None,
            },
            Message::AndroidEmulatorBooted {
                avd: "Medium_Phone_API_36.0".to_string(),
                serial: "emulator-5554".to_string(),
                server_id: Some("shared-1".to_string()),
                operation_id: Some(7),
            },
            Message::AndroidEmulatorOperationFailed {
                operation: AndroidOperationKind::Shutdown,
                target: "emulator-5554".to_string(),
                server_id: Some("shared-1".to_string()),
                operation_id: Some(8),
                error: "device did not exit".to_string(),
            },
        ];

        // Message does not derive PartialEq; a serialize → deserialize →
        // serialize byte-equality check verifies the full roundtrip.
        for msg in messages {
            let bytes = msg.serialize().unwrap();
            let decoded = Message::deserialize(&bytes).unwrap();
            let reencoded = decoded.serialize().unwrap();
            assert_eq!(
                bytes,
                reencoded,
                "roundtrip mismatch for {}",
                msg.msg_name()
            );
        }
    }

    #[test]
    fn test_relay_udp_pairing_messages_roundtrip() {
        let messages = vec![
            Message::RelayAllocateUdpPairing {
                request_id: 42,
                target_port: 36459,
            },
            Message::RelayUdpPairingAllocated {
                request_id: 42,
                pairing: "7c9e6679-7425-40de-944b-e07fc1f90ae7".to_string(),
                host: "relay.example.com".to_string(),
                port: 36500,
                error: None,
            },
            Message::RelayUdpPairingAllocated {
                request_id: 43,
                pairing: String::new(),
                host: String::new(),
                port: 0,
                error: Some("no free UDP port in relay range".to_string()),
            },
            Message::RelayReleaseUdpPairing {
                pairing: "7c9e6679-7425-40de-944b-e07fc1f90ae7".to_string(),
            },
        ];

        for msg in messages {
            let bytes = msg.serialize().unwrap();
            let decoded = Message::deserialize(&bytes).unwrap();
            let reencoded = decoded.serialize().unwrap();
            assert_eq!(
                bytes,
                reencoded,
                "roundtrip mismatch for {}",
                msg.msg_name()
            );
            match &msg {
                Message::RelayAllocateUdpPairing { .. } => {
                    assert_eq!(msg.msg_name(), "RelayAllocateUdpPairing")
                }
                Message::RelayUdpPairingAllocated { .. } => {
                    assert_eq!(msg.msg_name(), "RelayUdpPairingAllocated")
                }
                Message::RelayReleaseUdpPairing { .. } => {
                    assert_eq!(msg.msg_name(), "RelayReleaseUdpPairing")
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_screen_stream_stats_roundtrip() {
        let msg = Message::ScreenStreamStats {
            frames: 300,
            avg_pickup_us: 1200,
            avg_encode_ms: 2.6,
            delivered: 298,
            dropped: 2,
            avg_net_ms: 4.1,
        };
        let bytes = msg.serialize().unwrap();
        let decoded = Message::deserialize(&bytes).unwrap();
        match decoded {
            Message::ScreenStreamStats {
                frames,
                avg_pickup_us,
                avg_encode_ms,
                delivered,
                dropped,
                avg_net_ms,
            } => {
                assert_eq!(frames, 300);
                assert_eq!(avg_pickup_us, 1200);
                assert_eq!(avg_encode_ms, 2.6);
                assert_eq!(delivered, 298);
                assert_eq!(dropped, 2);
                assert_eq!(avg_net_ms, 4.1);
            }
            other => panic!("expected ScreenStreamStats, got {}", other.msg_name()),
        }
        assert_eq!(msg.msg_name(), "ScreenStreamStats");
        assert!(msg.is_control());
    }

    #[test]
    fn test_screen_stream_ready_flattens_endpoint() {
        let msg = Message::ScreenStreamReady {
            source: "main".to_string(),
            server_id: None,
            endpoint: StreamEndpoint::Direct {
                host: String::new(),
                port: 36459,
            },
            token: "0123456789abcdef0123456789abcdef".to_string(),
            codec: "h264".to_string(),
            format: "nv12".to_string(),
            error: None,
        };
        let json = String::from_utf8(msg.serialize().unwrap()).unwrap();
        assert!(json.contains("\"endpoint\":\"direct\""), "{}", json);
        assert!(json.contains("\"port\":36459"), "{}", json);
    }

    #[test]
    fn test_screen_stream_response_matching_is_server_scoped() {
        let request = Message::StartScreenStream {
            source: "sim:device-1".to_string(),
            server_id: Some("shared-1".to_string()),
        };
        let response = |server_id| Message::ScreenStreamReady {
            source: "sim:device-1".to_string(),
            server_id,
            endpoint: StreamEndpoint::Direct {
                host: String::new(),
                port: 36459,
            },
            token: String::new(),
            codec: "h264".to_string(),
            format: "nv12".to_string(),
            error: None,
        };

        assert!(request.expected_request_response(&response(Some("shared-1".to_string()))));
        assert!(!request.expected_request_response(&response(None)));
        assert!(!request.expected_request_response(&response(Some("shared-2".to_string()))));
    }

    #[test]
    fn test_simulator_power_response_matching_is_server_scoped() {
        let request = Message::ShutdownSimulator {
            spec: "sim:device-1".to_string(),
            server_id: Some("shared-1".to_string()),
        };
        let response = |server_id| Message::ShutdownSimulatorResult {
            spec: "sim:device-1".to_string(),
            server_id,
            error: None,
        };

        assert!(request.expected_request_response(&response(Some("shared-1".to_string()))));
        assert!(!request.expected_request_response(&response(None)));
        assert!(!request.expected_request_response(&response(Some("shared-2".to_string()))));
    }

    #[test]
    fn test_android_power_response_matching_and_relay_routing_are_server_scoped() {
        let request = Message::BootAndroidEmulator {
            avd: "Pixel_Fold_API_36".to_string(),
            server_id: Some("shared-1".to_string()),
        };
        let response = |server_id| Message::BootAndroidEmulatorResult {
            avd: "Pixel_Fold_API_36".to_string(),
            server_id,
            operation_id: None,
            error: None,
        };
        assert_eq!(request.relay_route_server_id().as_deref(), Some("shared-1"));
        assert!(request.expected_request_response(&response(Some("shared-1".to_string()))));
        assert!(!request.expected_request_response(&response(None)));
        assert!(!request.expected_request_response(&response(Some("shared-2".to_string()))));

        let request = Message::ShutdownAndroidEmulator {
            serial: "emulator-5554".to_string(),
            server_id: Some("shared-1".to_string()),
        };
        let response = |server_id| Message::ShutdownAndroidEmulatorResult {
            serial: "emulator-5554".to_string(),
            server_id,
            operation_id: None,
            error: None,
        };
        assert_eq!(request.relay_route_server_id().as_deref(), Some("shared-1"));
        assert!(request.expected_request_response(&response(Some("shared-1".to_string()))));
        assert!(!request.expected_request_response(&response(None)));
        assert!(!request.expected_request_response(&response(Some("shared-2".to_string()))));
    }

    #[test]
    fn test_android_power_messages_accept_legacy_payloads() {
        let boot =
            Message::from_text(r#"{"BootAndroidEmulator":{"avd":"Pixel_Fold_API_36"}}"#).unwrap();
        assert!(matches!(
            boot,
            Message::BootAndroidEmulator {
                server_id: None,
                ..
            }
        ));
        let result = Message::from_text(
            r#"{"BootAndroidEmulatorResult":{"avd":"Pixel_Fold_API_36","error":null}}"#,
        )
        .unwrap();
        assert!(matches!(
            result,
            Message::BootAndroidEmulatorResult {
                server_id: None,
                ..
            }
        ));
    }

    #[test]
    fn test_screen_source_info_flattens_status() {
        let info = ScreenSourceInfo {
            spec: "sim:ABCD-1234".to_string(),
            kind: ScreenSourceKind::IosSimulator,
            title: "iPhone 16".to_string(),
            status: ScreenSourceStatus::Unavailable {
                reason: "simulator shut down".to_string(),
            },
            width: None,
            height: None,
            primary: false,
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(
            json.contains("\"status\":\"unavailable\",\"reason\":\"simulator shut down\""),
            "{}",
            json
        );
        // Option fields are skipped when unset.
        assert!(!json.contains("\"width\""), "{}", json);
    }

    #[test]
    fn test_quick_command_requests_expose_session_id_for_relay_routing() {
        // Quick command requests must expose their session so the client can
        // route them to the owning server's relay request tunnel; a `None`
        // here silently reroutes them to the local relay (Direct) connection.
        let session_id = SessionId::new();

        let list = Message::QuickCommandList { session_id };
        assert_eq!(list.session_id(), Some(session_id));

        let run = Message::RunQuickCommand {
            session_id,
            name: "build".to_string(),
        };
        assert_eq!(run.session_id(), Some(session_id));

        let setup_mcp_app = Message::SetupMcpApp {
            session_id,
            executable: "kimi".to_string(),
        };
        assert_eq!(setup_mcp_app.session_id(), Some(session_id));
    }

    #[test]
    fn test_setup_mcp_app_result_is_request_response_and_roundtrips() {
        // SetupMcpApp rides the request connection; the handler MUST answer
        // with SetupMcpAppResult (or an Error), so the response must be
        // classified as one and carry the session for relay routing.
        let session_id = SessionId::new();
        let result = Message::SetupMcpAppResult {
            session_id,
            error: None,
        };
        assert!(result.is_request_response());
        assert_eq!(result.session_id(), Some(session_id));
        assert_eq!(result.msg_name(), "SetupMcpAppResult");

        let failed = Message::SetupMcpAppResult {
            session_id,
            error: Some("session not found".to_string()),
        };
        let request = Message::SetupMcpApp {
            session_id,
            executable: "qwen".to_string(),
        };
        assert!(request.expected_request_response(&failed));

        let encoded = serde_json::to_string(&failed).unwrap();
        let decoded: Message = serde_json::from_str(&encoded).unwrap();
        assert!(matches!(
            decoded,
            Message::SetupMcpAppResult {
                error: Some(ref message),
                ..
            } if message == "session not found"
        ));
    }

    #[test]
    fn test_rename_session_is_control_and_exposes_session_id() {
        let session_id = SessionId::new();
        let message = Message::RenameSession {
            session_id,
            session_name: Some("Renamed".to_string()),
        };

        assert_eq!(message.session_id(), Some(session_id));
        assert!(message.is_control());
    }

    #[test]
    fn test_hello_serialization_with_auth_fields() {
        let target_session_id = SessionId::new();
        let message = Message::Hello {
            protocol_version: 2,
            client_version: "1.0.0".to_string(),
            secret_key_hash: Some("abc123".to_string()),
            supports_binary_output: true,
            supports_message_encryption: true,
            connection_kind: Some(ConnectionKind::Terminal),
            auth_token: Some("token-123".to_string()),
            target_session_id: Some(target_session_id),
            relay_terminal_capability: Some("rtc1:test".to_string()),
            target_server_id: Some("server-1".to_string()),
            relay_request_capability: Some("rrc1:test".to_string()),
        };

        let serialized = message.to_text().unwrap();
        let deserialized = Message::from_text(&serialized).unwrap();

        match deserialized {
            Message::Hello {
                protocol_version,
                client_version,
                secret_key_hash,
                supports_binary_output,
                supports_message_encryption,
                connection_kind,
                auth_token,
                target_session_id: decoded_target_session_id,
                relay_terminal_capability,
                target_server_id,
                relay_request_capability,
            } => {
                assert_eq!(protocol_version, 2);
                assert_eq!(client_version, "1.0.0");
                assert_eq!(secret_key_hash.as_deref(), Some("abc123"));
                assert!(supports_binary_output);
                assert!(supports_message_encryption);
                assert_eq!(connection_kind, Some(ConnectionKind::Terminal));
                assert_eq!(auth_token.as_deref(), Some("token-123"));
                assert_eq!(decoded_target_session_id, Some(target_session_id));
                assert_eq!(relay_terminal_capability.as_deref(), Some("rtc1:test"));
                assert_eq!(target_server_id.as_deref(), Some("server-1"));
                assert_eq!(relay_request_capability.as_deref(), Some("rrc1:test"));
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_message_role_classifiers() {
        let session_id = SessionId::new();
        let request = Message::GitDiff {
            session_id,
            path: "src/main.rs".to_string(),
            commit: None,
            cached: false,
            compare_with_worktree: false,
        };
        let response = Message::GitDiffResult {
            session_id,
            path: "src/main.rs".to_string(),
            diff: b"diff --git".to_vec(),
        };
        let terminal = Message::Input {
            session_id,
            data: b"ls\r".to_vec(),
            cursor_from_end: None,
        };

        assert!(request.is_request_message());
        assert!(response.is_request_response());
        assert!(request.expected_request_response(&response));
        assert!(terminal.is_terminal_stream_message());
        assert!(terminal.is_terminal_client_message());
        assert!(!terminal.is_request_message());

        let stop_watching = Message::StopWatchingAiConversation {
            session_id,
            conversation_id: "conversation-1".to_string(),
        };
        assert!(!stop_watching.is_request_message());

        let history_request = Message::FetchHistoryRaw {
            session_id,
            before_offset: 1024,
            max_bytes: 4096,
        };
        let history_response = Message::HistoryRawChunk {
            session_id,
            start_offset: 0,
            end_offset: 1024,
            data: vec![1, 2, 3],
        };
        let export_history = Message::ExportSessionRawHistory { session_id };
        assert!(history_request.is_request_message());
        assert!(!history_request.is_terminal_client_message());
        assert!(history_response.is_request_response());
        assert!(!history_response.is_terminal_stream_message());
        assert!(history_request.expected_request_response(&history_response));
        assert!(export_history.is_request_message());
        assert_eq!(export_history.session_id(), Some(session_id));

        let checkout = Message::GitCheckout {
            session_id,
            ref_: "feature/request-routing".to_string(),
            create_branch: false,
            branch_name: String::new(),
            track: false,
            force: false,
            merge: false,
        };
        let delete_branch = Message::GitDeleteBranch {
            session_id,
            branch: "feature/obsolete".to_string(),
            is_worktree: false,
        };
        let merge = Message::GitMerge {
            session_id,
            branch: "feature/ready".to_string(),
        };
        assert!(checkout.is_request_message());
        assert!(delete_branch.is_request_message());
        assert!(merge.is_request_message());
    }

    #[test]
    fn commit_message_generation_uses_acceptance_as_request_response() {
        let session_id = SessionId::new();
        let accepted = Message::GitGenerateCommitMessageAccepted { session_id };
        let agent_request = Message::GitGenerateCommitMessageAgent { session_id };
        let completed = Message::GitGenerateCommitMessageResult {
            session_id,
            message: Some("fix: keep async result".to_string()),
            error: None,
        };

        assert!(accepted.is_request_response());
        assert!(agent_request.expected_request_response(&accepted));
        assert!(!agent_request.expected_request_response(&completed));

        let encoded = accepted.to_text().unwrap();
        assert!(matches!(
            Message::from_text(&encoded).unwrap(),
            Message::GitGenerateCommitMessageAccepted {
                session_id: decoded_session_id,
            } if decoded_session_id == session_id
        ));
    }

    #[test]
    fn test_tab_info_messages_are_request_response_pair() {
        let server_id = Some("server-a".to_string());
        let session_id = SessionId::new();
        let request = Message::GetTabInfoList {
            server_id: server_id.clone(),
        };
        let response = Message::TabInfoList {
            server_id,
            tabs: vec![TabInfo {
                server_id: Some("server-a".to_string()),
                tab_id: session_id,
                sessions: Vec::new(),
                layout: vterm_common::TerminalLayout::Terminal { session_id },
            }],
        };

        assert!(request.is_request_message());
        assert!(response.is_request_response());
        assert!(request.expected_request_response(&response));
    }

    #[test]
    fn test_git_stash_and_submodule_messages_are_request_response_pairs() {
        let session_id = SessionId::new();

        let stash_list = Message::GitStashList { session_id };
        let stash_list_response = Message::GitStashListResult {
            session_id,
            repo_root: Some("D:/repo".to_string()),
            stashes: vec![GitStashEntry {
                index: 0,
                message: "wip".to_string(),
                branch: "main".to_string(),
                date: "just now".to_string(),
                files: vec![],
            }],
        };
        assert!(stash_list.is_request_message());
        assert!(stash_list_response.is_request_response());
        assert!(stash_list.expected_request_response(&stash_list_response));
        // Relay routing relies on session_id(); a missing arm silently
        // misroutes the request to the directly connected server.
        assert_eq!(stash_list.session_id(), Some(session_id));
        assert_eq!(stash_list_response.session_id(), Some(session_id));

        for request in [
            Message::GitStashPush {
                session_id,
                message: Some("wip".to_string()),
                paths: vec![],
            },
            Message::GitStashApply {
                session_id,
                index: 0,
            },
            Message::GitStashPop {
                session_id,
                index: 0,
            },
            Message::GitStashDrop {
                session_id,
                index: 0,
            },
        ] {
            let response = Message::GitStashResult {
                session_id,
                error: None,
            };
            assert!(request.is_request_message());
            assert!(response.is_request_response());
            assert!(request.expected_request_response(&response));
            assert_eq!(request.session_id(), Some(session_id));
            assert_eq!(response.session_id(), Some(session_id));
        }

        let submodule_list = Message::GitSubmoduleList { session_id };
        let submodule_list_response = Message::GitSubmoduleListResult {
            session_id,
            repo_root: None,
            submodules: vec![GitSubmoduleInfo {
                path: "vendor/lib".to_string(),
                head: Some("main".to_string()),
                head_detached: None,
            }],
        };
        assert!(submodule_list.is_request_message());
        assert!(submodule_list_response.is_request_response());
        assert!(submodule_list.expected_request_response(&submodule_list_response));
        assert_eq!(submodule_list.session_id(), Some(session_id));
        assert_eq!(submodule_list_response.session_id(), Some(session_id));
    }

    #[test]
    fn test_polish_voice_input_text_serialization_and_pairing() {
        let session_id = SessionId::new();
        let request_id = "voice-polish-1".to_string();
        let window_id = 42;
        let component_id = 7;
        let request = Message::PolishVoiceInputText {
            window_id,
            component_id,
            request_id: request_id.clone(),
            text: "fix teh typo".to_string(),
            context: VoiceInputOptimizationContext::git(
                session_id,
                "src/main.rs",
                "abc123",
                10,
                20,
            ),
        };
        let response = Message::PolishVoiceInputTextResult {
            window_id,
            component_id,
            session_id,
            request_id,
            text: Some("fix the typo".to_string()),
            error: None,
        };

        let serialized = request.serialize().unwrap();
        let decoded = Message::deserialize(&serialized).unwrap();

        match decoded {
            Message::PolishVoiceInputText {
                window_id: decoded_window_id,
                component_id: decoded_component_id,
                text,
                context,
                ..
            } => {
                assert_eq!(decoded_window_id, window_id);
                assert_eq!(decoded_component_id, component_id);
                assert_eq!(text, "fix teh typo");
                assert_eq!(
                    context,
                    VoiceInputOptimizationContext::git(session_id, "src/main.rs", "abc123", 10, 20,)
                );
            }
            other => panic!("unexpected message: {other:?}"),
        }

        assert!(request.is_request_message());
        assert!(response.is_request_response());
        assert!(request.expected_request_response(&response));
    }

    #[test]
    fn test_binary_transcribe_audio_round_trip_with_context() {
        let session_id = SessionId::new();
        let context = VoiceInputOptimizationContext::file(session_id, "src/main.rs", 10, 20);
        let message = Message::TranscribeAudio {
            window_id: 42,
            event_id: "voice_input_dialog".to_string(),
            context: context.clone(),
            data: vec![1, 2, 3, 4].into(),
            compressed: true,
        };

        let frame = message.to_binary_frame().unwrap();
        let decoded = Message::from_transport_binary(&frame).unwrap();

        match decoded {
            Message::TranscribeAudio {
                window_id,
                event_id,
                context: decoded_context,
                data,
                compressed,
            } => {
                assert_eq!(window_id, 42);
                assert_eq!(event_id, "voice_input_dialog");
                assert_eq!(decoded_context, context);
                assert_eq!(data.0, vec![1, 2, 3, 4]);
                assert!(compressed);
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }

    #[test]
    fn test_split_terminal_serialization() {
        let target_session_id = SessionId::new();
        let message = Message::SplitTerminal {
            target_session_id,
            placement: vterm_common::TabSplitPlacement::Bottom,
            size: TerminalSize::new(24, 80),
            profile_id: None,
        };

        let serialized = message.serialize().unwrap();
        let decoded = Message::deserialize(&serialized).unwrap();

        match decoded {
            Message::SplitTerminal {
                target_session_id: decoded_target,
                placement,
                size,
                profile_id,
            } => {
                assert_eq!(decoded_target, target_session_id);
                assert_eq!(placement, vterm_common::TabSplitPlacement::Bottom);
                assert_eq!(size, TerminalSize::new(24, 80));
                assert_eq!(profile_id, None);
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }

    #[test]
    fn test_split_terminal_serialization_with_profile() {
        let target_session_id = SessionId::new();
        let message = Message::SplitTerminal {
            target_session_id,
            placement: vterm_common::TabSplitPlacement::Right,
            size: TerminalSize::new(30, 100),
            profile_id: Some("powershell".to_string()),
        };

        let serialized = message.serialize().unwrap();
        let decoded = Message::deserialize(&serialized).unwrap();

        match decoded {
            Message::SplitTerminal {
                target_session_id: decoded_target,
                placement,
                size,
                profile_id,
            } => {
                assert_eq!(decoded_target, target_session_id);
                assert_eq!(placement, vterm_common::TabSplitPlacement::Right);
                assert_eq!(size, TerminalSize::new(30, 100));
                assert_eq!(profile_id.as_deref(), Some("powershell"));
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }

    #[test]
    fn test_binary_output_transport_round_trip() {
        let session_id = SessionId::new();
        let message = Message::Output {
            session_id,
            start_offset: 42,
            data: vec![27, 91, 50, 74],
        };

        let frame = message.to_binary_frame().unwrap();
        let decoded = Message::from_transport_binary(&frame).unwrap();

        match decoded {
            Message::Output {
                session_id: decoded_session_id,
                start_offset,
                data,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(start_offset, 42);
                assert_eq!(data, vec![27, 91, 50, 74]);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_binary_input_transport_round_trip_with_cursor() {
        let session_id = SessionId::new();
        let message = Message::Input {
            session_id,
            data: vec![0, 255, 10, 13],
            cursor_from_end: Some(CursorFromEnd {
                lines_from_end: 7,
                column: 12,
            }),
        };

        let frame = message.to_binary_frame().unwrap();
        let decoded = Message::from_transport_binary(&frame).unwrap();

        match decoded {
            Message::Input {
                session_id: decoded_session_id,
                data,
                cursor_from_end,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(data, vec![0, 255, 10, 13]);
                let cursor = cursor_from_end.expect("cursor metadata should round-trip");
                assert_eq!(cursor.lines_from_end, 7);
                assert_eq!(cursor.column, 12);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_chunked_binary_file_content_transport_round_trip() {
        let session_id = SessionId::new();
        let message = Message::FileContent {
            session_id,
            path: "Cargo.lock".to_string(),
            data: vec![0x5a; MAX_BINARY_TRANSPORT_CHUNK_SIZE * 2 + 17],
            diff_ranges: Vec::new(),
            error: None,
        };

        let frames = message.to_transport_binary_frames().unwrap();
        assert!(frames.len() > 1);

        let mut decoder = TransportBinaryFrameDecoder::default();
        let mut decoded = None;
        for frame in frames {
            let message = decoder.decode_message_frame(&frame).unwrap();
            if message.is_some() {
                decoded = message;
            }
        }

        match decoded.expect("chunked message should complete") {
            Message::FileContent {
                session_id: decoded_session_id,
                path,
                data,
                diff_ranges,
                error,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(path, "Cargo.lock");
                assert_eq!(data.len(), MAX_BINARY_TRANSPORT_CHUNK_SIZE * 2 + 17);
                assert!(diff_ranges.is_empty());
                assert!(error.is_none());
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_chunked_binary_save_file_transport_round_trip() {
        let session_id = SessionId::new();
        let data: Vec<u8> = (0..MAX_BINARY_TRANSPORT_CHUNK_SIZE + 257)
            .map(|i| (i % 251) as u8)
            .collect();
        let message = Message::SaveFile {
            session_id,
            path: "src/main.rs".to_string(),
            data: data.clone(),
        };

        let frames = message.to_transport_binary_frames().unwrap();
        assert!(frames.len() > 1);

        let mut decoder = TransportBinaryFrameDecoder::default();
        let mut decoded = None;
        for frame in frames {
            let message = decoder.decode_message_frame(&frame).unwrap();
            if message.is_some() {
                decoded = message;
            }
        }

        match decoded.expect("chunked save file message should complete") {
            Message::SaveFile {
                session_id: decoded_session_id,
                path,
                data: decoded_data,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(path, "src/main.rs");
                assert_eq!(decoded_data, data);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_chunked_binary_ai_conversation_messages_transport_round_trip() {
        let session_id = SessionId::new();
        let message = Message::AiConversationMessages {
            session_id,
            conversation_id: "48cfd0e8-65da-438e-981a-bf0ce482c144".to_string(),
            messages: vec![
                AiConversationMessage {
                    id: "user-1".to_string(),
                    role: AiConversationRole::User,
                    kind: AiConversationMessageKind::Text,
                    time: "1778293323".to_string(),
                    model: String::new(),
                    markdown: "Summarize this repo.".to_string(),
                },
                AiConversationMessage {
                    id: "assistant-1".to_string(),
                    role: AiConversationRole::Model,
                    kind: AiConversationMessageKind::Thinking,
                    time: "1778293324".to_string(),
                    model: "glm-5.1".to_string(),
                    markdown: "Inspecting files.".repeat(MAX_BINARY_TRANSPORT_CHUNK_SIZE / 8),
                },
            ],
        };

        let frames = message.to_transport_binary_frames().unwrap();
        assert!(frames.len() > 1);

        let mut decoder = TransportBinaryFrameDecoder::default();
        let mut decoded = None;
        for frame in frames {
            let message = decoder.decode_message_frame(&frame).unwrap();
            if message.is_some() {
                decoded = message;
            }
        }

        match decoded.expect("chunked message should complete") {
            Message::AiConversationMessages {
                session_id: decoded_session_id,
                conversation_id,
                messages,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(conversation_id, "48cfd0e8-65da-438e-981a-bf0ce482c144");
                assert_eq!(messages.len(), 2);
                assert_eq!(messages[0].role, AiConversationRole::User);
                assert_eq!(messages[0].kind, AiConversationMessageKind::Text);
                assert_eq!(messages[1].role, AiConversationRole::Model);
                assert_eq!(messages[1].kind, AiConversationMessageKind::Thinking);
                assert_eq!(messages[1].model, "glm-5.1");
                assert!(messages[1].markdown.starts_with("Inspecting files."));
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_binary_ai_conversation_messages_update_transport_round_trip() {
        let session_id = SessionId::new();
        let message = Message::AiConversationMessagesUpdate {
            session_id,
            conversation_id: "conversation-1".to_string(),
            start_index: 3,
            messages: vec![AiConversationMessage {
                id: "assistant-4".to_string(),
                role: AiConversationRole::Model,
                kind: AiConversationMessageKind::Text,
                time: "2026-05-09 12:00".to_string(),
                model: "glm-5.1".to_string(),
                markdown: "New message".to_string(),
            }],
        };

        let frame = message.to_binary_frame().unwrap();
        let decoded = Message::from_transport_binary(&frame).unwrap();

        match decoded {
            Message::AiConversationMessagesUpdate {
                session_id: decoded_session_id,
                conversation_id,
                start_index,
                messages,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(conversation_id, "conversation-1");
                assert_eq!(start_index, 3);
                assert_eq!(messages.len(), 1);
                assert_eq!(messages[0].id, "assistant-4");
                assert_eq!(messages[0].model, "glm-5.1");
                assert_eq!(messages[0].markdown, "New message");
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_binary_path_content_transport_round_trip() {
        let session_id = SessionId::new();
        let message = Message::PathContentList {
            session_id,
            path: "C:\\work".to_string(),
            items: vec![
                FileListItem::directory("src".to_string()),
                FileListItem::file("Cargo.toml".to_string()),
            ],
            listing_generation: 42,
        };

        let frame = message.to_binary_frame().unwrap();
        let decoded = Message::from_transport_binary(&frame).unwrap();

        match decoded {
            Message::PathContentList {
                session_id: decoded_session_id,
                path,
                items,
                listing_generation,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(path, "C:\\work");
                assert_eq!(listing_generation, 42);
                assert_eq!(items.len(), 2);
                assert_eq!(items[0], FileListItem::directory("src".to_string()));
                assert_eq!(items[1], FileListItem::file("Cargo.toml".to_string()));
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_binary_git_changed_files_transport_round_trip() {
        let session_id = SessionId::new();
        let message = Message::GitChangedFilesList {
            session_id,
            commit: None,
            repo_root: Some("/repo".to_string()),
            items: vec![GitChangedFile {
                file: "src/main.rs".to_string(),
                status: GitFileStatus::Modified,
                staged: true,
                is_submodule: true,
                submodule_dirty: false,
            }],
        };

        let frame = message.to_binary_frame().unwrap();
        let decoded = Message::from_transport_binary(&frame).unwrap();

        match decoded {
            Message::GitChangedFilesList {
                session_id: decoded_session_id,
                commit,
                repo_root,
                items,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(commit, None);
                assert_eq!(repo_root.as_deref(), Some("/repo"));
                assert_eq!(items.len(), 1);
                assert_eq!(items[0].file, "src/main.rs");
                assert_eq!(items[0].status, GitFileStatus::Modified);
                assert!(items[0].staged);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_binary_find_files_transport_round_trip() {
        let session_id = SessionId::new();
        let message = Message::FindFilesResult {
            session_id,
            request_id: "search-1".to_string(),
            query: "needle".to_string(),
            chunk_index: 2,
            is_final: true,
            results: vec![FileSearchResult {
                file: "main.rs".to_string(),
                path: "src/main.rs".to_string(),
                matches: vec![
                    FileSearchMatch {
                        line: 7,
                        text: "let needle = true;".to_string(),
                    },
                    FileSearchMatch {
                        line: 12,
                        text: "println!(\"needle\");".to_string(),
                    },
                ],
            }],
            error: None,
        };

        let frame = message.to_binary_frame().unwrap();
        let decoded = Message::from_transport_binary(&frame).unwrap();

        match decoded {
            Message::FindFilesResult {
                session_id: decoded_session_id,
                request_id,
                query,
                chunk_index,
                is_final,
                results,
                error,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(request_id, "search-1");
                assert_eq!(query, "needle");
                assert_eq!(chunk_index, 2);
                assert!(is_final);
                assert!(error.is_none());
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].path, "src/main.rs");
                assert_eq!(results[0].matches[0].line, 7);
                assert_eq!(results[0].matches[1].text, "println!(\"needle\");");
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_chunked_binary_search_path_names_transport_round_trip() {
        let session_id = SessionId::new();
        let results: Vec<PathNameMatch> = (0..1000)
            .map(|i| PathNameMatch {
                path: format!("/repo/crates/generated/deep/path/component_{i:04}.rs"),
                name: format!("component_{i:04}.rs"),
                is_dir: i % 7 == 0,
            })
            .collect();
        let message = Message::SearchPathNamesResult {
            session_id,
            request_id: "name-search-1".to_string(),
            root_path: "/repo".to_string(),
            query: "c".to_string(),
            is_final: true,
            results: results.clone(),
            error: None,
        };

        let frames = message.to_transport_binary_frames().unwrap();
        assert!(frames.len() > 1);

        let mut decoder = TransportBinaryFrameDecoder::default();
        let mut decoded = None;
        for frame in frames {
            let message = decoder.decode_message_frame(&frame).unwrap();
            if message.is_some() {
                decoded = message;
            }
        }

        match decoded.expect("path name search message should decode") {
            Message::SearchPathNamesResult {
                session_id: decoded_session_id,
                request_id,
                root_path,
                query,
                is_final,
                results: decoded_results,
                error,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(request_id, "name-search-1");
                assert_eq!(root_path, "/repo");
                assert_eq!(query, "c");
                assert!(is_final);
                assert!(error.is_none());
                assert_eq!(decoded_results, results);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_chunked_binary_git_changed_files_transport_round_trip() {
        let session_id = SessionId::new();
        let items: Vec<GitChangedFile> = (0..1500)
            .map(|i| GitChangedFile {
                file: format!("crates/vterm/assets/queries/lang_{i:04}/highlights.scm"),
                status: GitFileStatus::Added,
                staged: i % 2 == 0,
                is_submodule: false,
                submodule_dirty: false,
            })
            .collect();
        let message = Message::GitChangedFilesList {
            session_id,
            commit: Some("abc123".to_string()),
            repo_root: Some("/repo".to_string()),
            items: items.clone(),
        };

        let frames = message.to_transport_binary_frames().unwrap();
        assert!(frames.len() > 1);

        let mut decoder = TransportBinaryFrameDecoder::default();
        let mut decoded = None;
        for frame in frames {
            let message = decoder.decode_message_frame(&frame).unwrap();
            if message.is_some() {
                decoded = message;
            }
        }

        match decoded.expect("git changed files message should decode") {
            Message::GitChangedFilesList {
                session_id: decoded_session_id,
                commit,
                repo_root,
                items: decoded_items,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(commit.as_deref(), Some("abc123"));
                assert_eq!(repo_root.as_deref(), Some("/repo"));
                assert_eq!(decoded_items, items);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_chunked_binary_git_branches_transport_round_trip() {
        let session_id = SessionId::new();
        let branches: Vec<GitBranchInfo> = (0..1530)
            .map(|i| GitBranchInfo {
                name: format!("remotes/origin/feature/deep/path/branch_{i:04}"),
                is_current: i == 17,
                is_remote: i % 2 == 0,
                worktree_path: None,
            })
            .collect();
        let tags: Vec<String> = (0..5).map(|i| format!("v1.{i}.0")).collect();
        let message = Message::GitBranchesResult {
            session_id,
            current: Some("feature/deep/path/branch_0017".to_string()),
            detached_head: None,
            branches: branches.clone(),
            remote_tracking_hash: Some("abc123def456".to_string()),
            tags: tags.clone(),
        };

        let frames = message.to_transport_binary_frames().unwrap();
        assert!(frames.len() > 1);

        let mut decoder = TransportBinaryFrameDecoder::default();
        let mut decoded = None;
        for frame in frames {
            let message = decoder.decode_message_frame(&frame).unwrap();
            if message.is_some() {
                decoded = message;
            }
        }

        match decoded.expect("git branches message should decode") {
            Message::GitBranchesResult {
                session_id: decoded_session_id,
                current,
                detached_head,
                branches: decoded_branches,
                remote_tracking_hash,
                tags: decoded_tags,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(current.as_deref(), Some("feature/deep/path/branch_0017"));
                assert_eq!(detached_head, None);
                assert_eq!(decoded_branches, branches);
                assert_eq!(remote_tracking_hash.as_deref(), Some("abc123def456"));
                assert_eq!(decoded_tags, tags);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_binary_git_branches_worktree_paths_round_trip() {
        // Verifies per-branch worktree_path is preserved across the binary
        // transport — including a None entry (main, primary) interleaved with
        // Some entries (linked worktrees), which is the real-world shape.
        let session_id = SessionId::new();
        let branches: Vec<GitBranchInfo> = vec![
            GitBranchInfo {
                name: "main".to_string(),
                is_current: true,
                is_remote: false,
                worktree_path: None,
            },
            GitBranchInfo {
                name: "feature".to_string(),
                is_current: false,
                is_remote: false,
                worktree_path: Some("/repo/.worktrees/feature".to_string()),
            },
            GitBranchInfo {
                name: "bugfix".to_string(),
                is_current: false,
                is_remote: false,
                worktree_path: Some("/repo/.worktrees/bugfix".to_string()),
            },
            GitBranchInfo {
                name: "remotes/origin/main".to_string(),
                is_current: false,
                is_remote: true,
                worktree_path: None,
            },
        ];
        let message = Message::GitBranchesResult {
            session_id,
            current: None,
            detached_head: Some("a1b2c3d".to_string()),
            branches: branches.clone(),
            remote_tracking_hash: None,
            tags: Vec::new(),
        };

        let frames = message.to_transport_binary_frames().unwrap();
        let mut decoder = TransportBinaryFrameDecoder::default();
        let mut decoded = None;
        for frame in frames {
            if let Some(m) = decoder.decode_message_frame(&frame).unwrap() {
                decoded = Some(m);
            }
        }

        match decoded.expect("git branches message should decode") {
            Message::GitBranchesResult {
                branches: decoded_branches,
                current,
                detached_head,
                ..
            } => {
                assert_eq!(decoded_branches, branches);
                assert_eq!(current, None);
                assert_eq!(detached_head.as_deref(), Some("a1b2c3d"));
                // Spot-check the worktree paths survived in order.
                assert_eq!(decoded_branches[0].worktree_path, None);
                assert_eq!(
                    decoded_branches[1].worktree_path.as_deref(),
                    Some("/repo/.worktrees/feature")
                );
                assert_eq!(
                    decoded_branches[2].worktree_path.as_deref(),
                    Some("/repo/.worktrees/bugfix")
                );
                assert_eq!(decoded_branches[3].worktree_path, None);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_binary_git_stage_transport_round_trip() {
        let session_id = SessionId::new();
        let paths: Vec<String> = (0..1200)
            .map(|i| format!("src/generated/deep/path/file_{i:04}.rs"))
            .collect();
        let message = Message::GitStage {
            session_id,
            paths: paths.clone(),
            repo_root: Some("/repo".to_string()),
        };

        let frames = message.to_transport_binary_frames().unwrap();
        let mut decoder = TransportBinaryFrameDecoder::default();
        let mut decoded = None;
        for frame in frames {
            let message = decoder.decode_message_frame(&frame).unwrap();
            if message.is_some() {
                decoded = message;
            }
        }

        match decoded.expect("git stage message should decode") {
            Message::GitStage {
                session_id: decoded_session_id,
                paths: decoded_paths,
                repo_root: decoded_repo_root,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(decoded_paths, paths);
                assert_eq!(decoded_repo_root.as_deref(), Some("/repo"));
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_binary_git_stage_decodes_legacy_frame_without_repo_root() {
        let session_id = SessionId::new();
        let message = Message::GitStage {
            session_id,
            paths: vec!["a.rs".to_string()],
            repo_root: None,
        };
        let mut frame = message.to_binary_frame().unwrap();
        // A pre-extension client emits no trailing repo_root flag at all.
        frame.truncate(frame.len() - 1);

        match Message::from_transport_binary(&frame).unwrap() {
            Message::GitStage {
                session_id: decoded_session_id,
                paths,
                repo_root,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(paths, vec!["a.rs".to_string()]);
                assert_eq!(repo_root, None);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_binary_git_unstage_and_discard_transport_round_trip() {
        let session_id = SessionId::new();
        let paths = vec![
            "README.md".to_string(),
            "crates/vterm-core/src/protocol.rs".to_string(),
        ];

        for (message, expected_repo_root) in [
            (
                Message::GitUnstage {
                    session_id,
                    paths: paths.clone(),
                    repo_root: None,
                },
                None,
            ),
            (
                Message::GitDiscard {
                    session_id,
                    paths: paths.clone(),
                    repo_root: Some("/repo".to_string()),
                },
                Some("/repo"),
            ),
        ] {
            let frame = message.to_binary_frame().unwrap();
            let decoded = Message::from_transport_binary(&frame).unwrap();
            match decoded {
                Message::GitUnstage {
                    session_id: decoded_session_id,
                    paths: decoded_paths,
                    repo_root: decoded_repo_root,
                }
                | Message::GitDiscard {
                    session_id: decoded_session_id,
                    paths: decoded_paths,
                    repo_root: decoded_repo_root,
                } => {
                    assert_eq!(decoded_session_id, session_id);
                    assert_eq!(decoded_paths, paths);
                    assert_eq!(decoded_repo_root.as_deref(), expected_repo_root);
                }
                _ => panic!("Wrong message type"),
            }
        }
    }

    #[test]
    fn test_binary_git_log_transport_round_trip() {
        let session_id = SessionId::new();
        let message = Message::GitLogList {
            session_id,
            commit: Some("abc123".to_string()),
            branch: Some("main".to_string()),
            repo_root: Some("/repo".to_string()),
            commits: vec![GitCommitInfo {
                hash: "abc123".to_string(),
                message: "subject\n\nbody line 1\nbody line 2".to_string(),
                author: "Ada".to_string(),
                date: "2 hours ago".to_string(),
                exact_time: "2026-04-22 12:34:56 +0800".to_string(),
                parents: vec!["def456".to_string(), "789abc".to_string()],
            }],
        };

        let frame = message.to_binary_frame().unwrap();
        let decoded = Message::from_transport_binary(&frame).unwrap();

        match decoded {
            Message::GitLogList {
                session_id: decoded_session_id,
                commit,
                branch,
                repo_root,
                commits,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(commit.as_deref(), Some("abc123"));
                assert_eq!(branch.as_deref(), Some("main"));
                assert_eq!(repo_root.as_deref(), Some("/repo"));
                assert_eq!(commits.len(), 1);
                assert_eq!(commits[0].hash, "abc123");
                assert_eq!(commits[0].message, "subject\n\nbody line 1\nbody line 2");
                assert_eq!(commits[0].author, "Ada");
                assert_eq!(commits[0].date, "2 hours ago");
                assert_eq!(commits[0].exact_time, "2026-04-22 12:34:56 +0800");
                assert_eq!(commits[0].parents, vec!["def456", "789abc"]);
            }
            _ => panic!("Wrong message type"),
        }
    }
    #[test]
    fn test_set_fs_watchers_round_trip() {
        let session_id = SessionId::new();
        let message = Message::SetFsWatchers {
            session_id,
            watches: FsWatchConfig {
                file_content: Some(FsWatchFileContent {
                    path: "src/main.rs".to_string(),
                    max_size: 4096,
                }),
                git: Some(FsWatchGit { commit: None }),
                directories: vec!["src".to_string(), "tests".to_string()],
            },
        };

        assert_eq!(message.msg_name(), "SetFsWatchers");
        assert_eq!(message.session_id(), Some(session_id));
        assert!(message.is_control());
        assert!(!message.is_request_message());
        assert!(message.is_terminal_client_message());

        let serialized = message.serialize().unwrap();
        let decoded = Message::deserialize(&serialized).unwrap();
        match decoded {
            Message::SetFsWatchers {
                session_id: decoded_session_id,
                watches,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(watches.file_content.unwrap().path, "src/main.rs");
                assert_eq!(watches.git.unwrap().commit, None);
                assert_eq!(
                    watches.directories,
                    vec!["src".to_string(), "tests".to_string()]
                );
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }

    #[test]
    fn test_stream_start_terminal_theme_colors_round_trip() {
        let session_id = SessionId::new();
        let message = Message::StreamStart {
            session_id,
            initial_size: Some(TerminalSize::new(24, 80)),
            background: Some([0xf5, 0xf5, 0xf7]),
            foreground: Some([0x1d, 0x1d, 0x1f]),
        };

        let serialized = message.serialize().unwrap();
        match Message::deserialize(&serialized).unwrap() {
            Message::StreamStart {
                session_id: decoded_session_id,
                background,
                foreground,
                ..
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(background, Some([0xf5, 0xf5, 0xf7]));
                assert_eq!(foreground, Some([0x1d, 0x1d, 0x1f]));
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }

    #[test]
    fn test_set_terminal_theme_colors_round_trip() {
        let session_id = SessionId::new();
        let message = Message::SetTerminalThemeColors {
            session_id,
            background: Some([0xf5, 0xf5, 0xf7]),
            foreground: Some([0x1d, 0x1d, 0x1f]),
        };

        assert_eq!(message.msg_name(), "SetTerminalThemeColors");
        assert_eq!(message.session_id(), Some(session_id));
        assert!(message.is_control());
        assert!(!message.is_request_message());
        assert!(message.is_terminal_client_message());

        let serialized = message.serialize().unwrap();
        let decoded = Message::deserialize(&serialized).unwrap();
        match decoded {
            Message::SetTerminalThemeColors {
                session_id: decoded_session_id,
                background,
                foreground,
            } => {
                assert_eq!(decoded_session_id, session_id);
                assert_eq!(background, Some([0xf5, 0xf5, 0xf7]));
                assert_eq!(foreground, Some([0x1d, 0x1d, 0x1f]));
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }
}

#[cfg(test)]
mod rebase_binary_tests {
    use super::*;

    fn roundtrip(message: Message) -> Message {
        // Full transport path: binary frame → chunked frames → decoder.
        let frames = message.to_transport_binary_frames().expect("binary frame");
        let mut decoder = TransportBinaryFrameDecoder::default();
        let mut out = None;
        for frame in frames {
            out = decoder.decode_message_frame(&frame).unwrap();
        }
        out.expect("complete message")
    }

    #[test]
    fn test_git_rebase_conflict_content_binary_roundtrip() {
        let session_id = SessionId::new();
        // Larger than one Noise frame (65_535) to exercise chunking.
        let big = "x".repeat(200_000);
        let ours_meta = GitCommitMeta {
            hash: "a1b2c3d4e5f6".to_string(),
            message: "base commit".to_string(),
            author: "Bob".to_string(),
            date: "2 days ago".to_string(),
        };
        let theirs_meta = GitCommitMeta {
            hash: "f6e5d4c3b2a1".to_string(),
            message: "incoming commit".to_string(),
            author: "Alice".to_string(),
            date: "3 hours ago".to_string(),
        };
        let message = Message::GitRebaseConflictContentResult {
            session_id,
            path: "src/Foo.cs".to_string(),
            base: Some(big.clone()),
            ours: Some(big.clone()),
            theirs: None,
            error: None,
            ours_commit: Some(ours_meta.clone()),
            theirs_commit: Some(theirs_meta.clone()),
        };
        match roundtrip(message) {
            Message::GitRebaseConflictContentResult {
                session_id: decoded,
                path,
                base,
                ours,
                theirs,
                error,
                ours_commit,
                theirs_commit,
            } => {
                assert_eq!(decoded, session_id);
                assert_eq!(path, "src/Foo.cs");
                assert_eq!(base.as_deref().map(str::len), Some(200_000));
                assert_eq!(ours.as_deref().map(str::len), Some(200_000));
                assert_eq!(theirs, None);
                assert_eq!(error, None);
                assert_eq!(ours_commit, Some(ours_meta));
                assert_eq!(theirs_commit, Some(theirs_meta));
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }

    #[test]
    fn test_git_rebase_resolve_binary_roundtrip() {
        let session_id = SessionId::new();
        let message = Message::GitRebaseResolve {
            session_id,
            path: "a/b.cs".to_string(),
            content: "line1\nline2\n".repeat(50_000),
        };
        match roundtrip(message) {
            Message::GitRebaseResolve {
                session_id: decoded,
                path,
                content,
            } => {
                assert_eq!(decoded, session_id);
                assert_eq!(path, "a/b.cs");
                assert_eq!(content.len(), 600_000);
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }

    #[test]
    fn test_git_rebase_start_result_binary_roundtrip() {
        let session_id = SessionId::new();
        let commits: Vec<GitRebaseTodoCommit> = (0..50)
            .map(|i| GitRebaseTodoCommit {
                hash: format!("hash{i}"),
                message: format!("commit message {i}"),
                author: "Alice".to_string(),
                date: "1 day ago".to_string(),
                action: "pick".to_string(),
            })
            .collect();
        let message = Message::GitRebaseStartResult {
            session_id,
            error: None,
            source_branch: Some("feature/x".to_string()),
            commits,
        };
        match roundtrip(message) {
            Message::GitRebaseStartResult {
                session_id: decoded,
                error,
                source_branch,
                commits,
            } => {
                assert_eq!(decoded, session_id);
                assert_eq!(error, None);
                assert_eq!(source_branch.as_deref(), Some("feature/x"));
                assert_eq!(commits.len(), 50);
                assert_eq!(commits[49].message, "commit message 49");
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }

    #[test]
    fn test_git_rebase_ai_resolve_binary_roundtrip() {
        let session_id = SessionId::new();
        let big_line = "let value = compute(); // ".repeat(8000);
        let message = Message::GitRebaseAiResolve {
            session_id,
            run_id: 7,
            instruction: "Keep both changes".to_string(),
            ours_label: Some("main (target)".to_string()),
            theirs_label: None,
            targets: vec![
                vterm_common::GitRebaseAiTarget {
                    path: "src/a.rs".to_string(),
                    hunks: vec![
                        vterm_common::GitRebaseAiHunk {
                            idx: 2,
                            base: vec!["base".to_string()],
                            ours: vec!["ours1".to_string(), big_line.clone()],
                            theirs: vec![],
                        },
                        vterm_common::GitRebaseAiHunk {
                            idx: 5,
                            base: vec![],
                            ours: vec!["o".to_string()],
                            theirs: vec!["t".to_string()],
                        },
                    ],
                },
                vterm_common::GitRebaseAiTarget {
                    path: "src/b.rs".to_string(),
                    hunks: vec![],
                },
            ],
        };
        match roundtrip(message) {
            Message::GitRebaseAiResolve {
                session_id: decoded,
                run_id,
                instruction,
                ours_label,
                theirs_label,
                targets,
            } => {
                assert_eq!(decoded, session_id);
                assert_eq!(run_id, 7);
                assert_eq!(instruction, "Keep both changes");
                assert_eq!(ours_label.as_deref(), Some("main (target)"));
                assert_eq!(theirs_label, None);
                assert_eq!(targets.len(), 2);
                assert_eq!(targets[0].path, "src/a.rs");
                assert_eq!(targets[0].hunks.len(), 2);
                assert_eq!(targets[0].hunks[0].idx, 2);
                assert_eq!(targets[0].hunks[0].ours[1].len(), big_line.len());
                assert!(targets[0].hunks[0].theirs.is_empty());
                assert_eq!(targets[0].hunks[1].idx, 5);
                assert_eq!(targets[1].path, "src/b.rs");
                assert!(targets[1].hunks.is_empty());
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }

    #[test]
    fn test_git_rebase_ai_resolve_result_binary_roundtrip() {
        let session_id = SessionId::new();
        let hunk = vterm_common::GitRebaseAiHunkEvent {
            path: "src/a.rs".to_string(),
            idx: 2,
            status: "done".to_string(),
            lines: Some(vec!["merged1".to_string(), "merged2".to_string()]),
            strategy: Some("union".to_string()),
            explanation: Some("kept both sides".to_string()),
            confidence: Some(0.92),
            error: None,
        };
        let message = Message::GitRebaseAiResolveResult {
            session_id,
            run_id: 7,
            hunk: Some(hunk),
            error: None,
            done: false,
        };
        match roundtrip(message) {
            Message::GitRebaseAiResolveResult {
                session_id: decoded,
                run_id,
                hunk,
                error,
                done,
            } => {
                assert_eq!(decoded, session_id);
                assert_eq!(run_id, 7);
                let hunk = hunk.expect("hunk event");
                assert_eq!(hunk.path, "src/a.rs");
                assert_eq!(hunk.idx, 2);
                assert_eq!(hunk.status, "done");
                assert_eq!(hunk.lines.as_ref().map(Vec::len), Some(2));
                assert_eq!(hunk.strategy.as_deref(), Some("union"));
                assert_eq!(hunk.explanation.as_deref(), Some("kept both sides"));
                assert_eq!(hunk.confidence, Some(0.92));
                assert_eq!(hunk.error, None);
                assert_eq!(error, None);
                assert!(!done);
            }
            other => panic!("unexpected message: {other:?}"),
        }

        // Final run-level error message without a hunk event.
        let message = Message::GitRebaseAiResolveResult {
            session_id,
            run_id: 8,
            hunk: None,
            error: Some("git LLM is not configured".to_string()),
            done: true,
        };
        match roundtrip(message) {
            Message::GitRebaseAiResolveResult {
                session_id: decoded,
                run_id,
                hunk,
                error,
                done,
            } => {
                assert_eq!(decoded, session_id);
                assert_eq!(run_id, 8);
                assert!(hunk.is_none());
                assert_eq!(error.as_deref(), Some("git LLM is not configured"));
                assert!(done);
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }
}
