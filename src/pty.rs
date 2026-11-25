use async_trait::async_trait;
use portable_pty::{ChildKiller, CommandBuilder, NativePtySystem, PtySize, PtySystem};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar};
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

use vterm_common::TerminalSize;

/// Error type for PTY operations
#[derive(thiserror::Error, Debug)]
pub enum PtyError {
    #[error("Failed to create PTY: {0}")]
    CreationError(String),
    #[error("Failed to spawn process: {0}")]
    SpawnError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("PTY not running")]
    NotRunning,
    #[error("Send error")]
    SendError,
}

/// Trait for PTY implementations
#[async_trait]
pub trait Pty: Send + Sync {
    /// Resize the PTY
    fn resize(&self, size: TerminalSize) -> Result<(), PtyError>;

    /// Write data to the PTY
    async fn write(&mut self, data: &[u8]) -> Result<(), PtyError>;

    /// Read data from the PTY (non-blocking)
    async fn read(&mut self, buf: &mut [u8]) -> Result<usize, PtyError>;

    /// Check if the PTY is still running
    fn is_running(&self) -> bool;

    /// Get the shell process ID if available
    fn process_id(&self) -> Option<u32>;
}

/// Cross-platform PTY implementation using portable-pty
pub struct PortablePty {
    master: Arc<std::sync::Mutex<Box<dyn portable_pty::MasterPty + Send>>>,
    child: Arc<std::sync::Mutex<Option<Box<dyn portable_pty::Child + Send + Sync>>>>,
    child_killer: Arc<std::sync::Mutex<Box<dyn ChildKiller + Send + Sync>>>,
    running: Arc<AtomicBool>,
    exit_completed: Arc<(std::sync::Mutex<bool>, Condvar)>,
    writer: Arc<std::sync::Mutex<Box<dyn Write + Send>>>,
    reader: Arc<std::sync::Mutex<Box<dyn Read + Send>>>,
    process_id: Option<u32>,
}

impl PortablePty {
    pub fn new_command(
        size: TerminalSize,
        program: &str,
        args: &[String],
        env: &HashMap<String, String>,
        cwd: Option<&str>,
    ) -> Result<Self, PtyError> {
        let pty_system = NativePtySystem::default();

        let pty_size = PtySize {
            rows: size.rows,
            cols: size.cols,
            pixel_width: 0,
            pixel_height: 0,
        };

        let pair = pty_system
            .openpty(pty_size)
            .map_err(|e| PtyError::CreationError(e.to_string()))?;

        let mut cmd = CommandBuilder::new(program);
        for arg in args {
            cmd.arg(arg);
        }
        for (key, value) in env {
            cmd.env(key, value);
        }
        if let Some(cwd) = cwd.filter(|cwd| !cwd.is_empty()) {
            cmd.cwd(Path::new(cwd));
        }

        let child = pair
            .slave
            .spawn_command(cmd)
            .map_err(|e| PtyError::SpawnError(e.to_string()))?;

        let process_id = child.process_id();
        let child_killer = child.clone_killer();

        let reader = pair
            .master
            .try_clone_reader()
            .map_err(|e| PtyError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        let writer = pair
            .master
            .take_writer()
            .map_err(|e| PtyError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        info!(
            "Spawned PTY with shell: {} (PID: {:?})",
            program, process_id
        );

        Ok(Self {
            master: Arc::new(std::sync::Mutex::new(pair.master)),
            child: Arc::new(std::sync::Mutex::new(Some(child))),
            child_killer: Arc::new(std::sync::Mutex::new(child_killer)),
            running: Arc::new(AtomicBool::new(true)),
            exit_completed: Arc::new((std::sync::Mutex::new(false), Condvar::new())),
            writer: Arc::new(std::sync::Mutex::new(writer)),
            reader: Arc::new(std::sync::Mutex::new(reader)),
            process_id,
        })
    }

    /// Create a new PTY with the given size and shell command
    pub fn new(size: TerminalSize, shell: Option<&str>) -> Result<Self, PtyError> {
        let shell_cmd = shell.map(|s| s.to_string()).unwrap_or_else(default_shell);
        Self::new_command(size, &shell_cmd, &[], &HashMap::new(), None)
    }

    /// Create a PTY with the default shell
    pub fn with_default_shell(size: TerminalSize) -> Result<Self, PtyError> {
        Self::new(size, None)
    }

    pub fn writer(&self) -> Arc<std::sync::Mutex<Box<dyn Write + Send>>> {
        self.writer.clone()
    }

    pub fn reader(&self) -> Arc<std::sync::Mutex<Box<dyn Read + Send>>> {
        self.reader.clone()
    }

    /// Spawn a dedicated blocking writer thread and return a channel sender.
    ///
    /// Writes sent to the channel are forwarded to the PTY without requiring
    /// the caller to hold the session lock across the blocking PTY write.
    ///
    /// `latency_session`: when set (and `VTERM_LATENCY_LOG` is on), each
    /// batch logs how much of the keystroke-echo budget was already spent by
    /// the time the bytes hit the ConPTY pipe — the difference to the
    /// reader thread's `server:pty-read-returned` is ConPTY/app time.
    pub fn spawn_writer(
        &self,
        latency_session: Option<vterm_common::SessionId>,
    ) -> mpsc::Sender<Vec<u8>> {
        let writer = self.writer.clone();
        let (tx, mut rx) = mpsc::channel::<Vec<u8>>(1024);

        std::thread::Builder::new()
            .name("vterm-pty-writer".to_string())
            .spawn(move || {
                const MAX_BATCH_BYTES: usize = 64 * 1024;
                while let Some(mut data) = rx.blocking_recv() {
                    // Channel + thread-wake probe: how much of the echo
                    // budget was already spent when the writer thread picked
                    // the batch up. Also fires for server-initiated writes
                    // (query responses) that closely follow a keystroke.
                    if let Some(session_id) = latency_session {
                        if let Some(echo_ms) =
                            vterm_common::latency::echo_ms_since_input(session_id)
                        {
                            if vterm_common::latency::sample(
                                "server:pty-write-recv",
                                session_id,
                                echo_ms as f64,
                                5.0,
                                3.0,
                            ) {
                                vterm_common::latency_log!(
                                    stage = "server:pty-write-recv",
                                    session = %session_id,
                                    echo_ms,
                                    bytes = data.len(),
                                    "[latency] writer thread picked up input late (channel/wake)"
                                );
                            }
                        }
                    }
                    while data.len() < MAX_BATCH_BYTES {
                        match rx.try_recv() {
                            Ok(next) if data.len() + next.len() <= MAX_BATCH_BYTES => {
                                data.extend_from_slice(&next);
                            }
                            Ok(next) => {
                                if let Err(error) = write_pty_data(&writer, &data) {
                                    error!("Failed to write to PTY: {error}");
                                    return;
                                }
                                data = next;
                            }
                            Err(_) => break,
                        }
                    }

                    let write_started = std::time::Instant::now();
                    if let Err(error) = write_pty_data(&writer, &data) {
                        error!("Failed to write to PTY: {error}");
                        break;
                    }
                    if let Some(session_id) = latency_session {
                        let write_ms = write_started.elapsed().as_millis() as u64;
                        if let Some(echo_ms) =
                            vterm_common::latency::echo_ms_since_input(session_id)
                        {
                            if vterm_common::latency::sample(
                                "server:pty-write-flushed",
                                session_id,
                                echo_ms as f64,
                                20.0,
                                3.0,
                            ) {
                                vterm_common::latency_log!(
                                    stage = "server:pty-write-flushed",
                                    session = %session_id,
                                    echo_ms,
                                    write_ms,
                                    bytes = data.len(),
                                    "[latency] PTY write flushed; remaining echo time is ConPTY/app"
                                );
                            }
                        }
                    }
                }
                debug!("PTY writer thread ended");
            })
            .expect("failed to spawn PTY writer thread");

        tx
    }

    pub fn spawn_exit_watcher<F>(&self, on_exit: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let child = self.child.lock().ok().and_then(|mut child| child.take());
        let running = Arc::clone(&self.running);
        let exit_completed = Arc::clone(&self.exit_completed);
        std::thread::Builder::new()
            .name("vterm-pty-wait".to_string())
            .spawn(move || {
                if let Some(mut child) = child {
                    let _ = child.wait();
                }
                running.store(false, Ordering::Release);
                mark_exit_completed(&exit_completed);
                on_exit();
            })
            .expect("failed to spawn PTY exit watcher");
    }

    pub fn terminate(&mut self) -> Result<(), PtyError> {
        if !self.running.load(Ordering::Acquire) {
            return Ok(());
        }

        let mut child = self.child.lock().map_err(|_| PtyError::NotRunning)?.take();
        if let Some(child) = child.as_mut() {
            #[cfg(unix)]
            if let Some(pid) = self.process_id {
                signal_process_group(pid, libc::SIGHUP);
            }
            let kill_result = child.kill();
            let wait_result = child.wait();
            #[cfg(unix)]
            if let Some(pid) = self.process_id {
                // Mop up group members that ignored the hangup.
                signal_process_group(pid, libc::SIGKILL);
            }
            self.running.store(false, Ordering::Release);
            mark_exit_completed(&self.exit_completed);
            kill_result.map_err(|e| {
                PtyError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e))
            })?;
            wait_result.map_err(|e| {
                PtyError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e))
            })?;
        } else {
            // The exit watcher owns the Child and is blocked in wait(). Signal
            // it through the cloned killer, then wait for confirmation that it
            // actually reaped the process. portable-pty implementations differ
            // in how they report killing an already-exited process, so watcher
            // completion is the authoritative success signal.
            #[cfg(unix)]
            if let Some(pid) = self.process_id {
                signal_process_group(pid, libc::SIGHUP);
            }
            let kill_result = self
                .child_killer
                .lock()
                .map_err(|_| PtyError::NotRunning)?
                .kill();
            if wait_for_exit_completion(&self.exit_completed, Duration::from_secs(2)) {
                #[cfg(unix)]
                if let Some(pid) = self.process_id {
                    signal_process_group(pid, libc::SIGKILL);
                }
                self.running.store(false, Ordering::Release);
                return Ok(());
            }
            if let Err(error) = kill_result {
                return Err(PtyError::IoError(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    error,
                )));
            }
            return Err(PtyError::IoError(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "timed out waiting for PTY process exit",
            )));
        }

        Ok(())
    }

    fn terminate_without_waiting(&mut self) {
        if !self.running.load(Ordering::Acquire) {
            return;
        }

        let child = self.child.lock().ok().and_then(|mut child| child.take());
        if let Some(mut child) = child {
            let running = Arc::clone(&self.running);
            let exit_completed = Arc::clone(&self.exit_completed);
            #[cfg(unix)]
            let process_id = self.process_id;
            #[cfg(unix)]
            if let Some(pid) = process_id {
                signal_process_group(pid, libc::SIGHUP);
            }
            if let Err(error) = std::thread::Builder::new()
                .name("vterm-pty-drop-wait".to_string())
                .spawn(move || {
                    let _ = child.kill();
                    let _ = child.wait();
                    #[cfg(unix)]
                    if let Some(pid) = process_id {
                        signal_process_group(pid, libc::SIGKILL);
                    }
                    running.store(false, Ordering::Release);
                    mark_exit_completed(&exit_completed);
                })
            {
                error!("Failed to spawn PTY drop cleanup thread: {error}");
            }
            return;
        }

        // The exit watcher owns the child and will reap it. Drop only needs to
        // signal termination; waiting here could block a Tokio runtime worker.
        if let Ok(mut killer) = self.child_killer.lock() {
            #[cfg(unix)]
            if let Some(pid) = self.process_id {
                signal_process_group(pid, libc::SIGHUP);
            }
            let _ = killer.kill();
        }
    }
}

impl Drop for PortablePty {
    fn drop(&mut self) {
        self.terminate_without_waiting();
    }
}

#[async_trait]
impl Pty for PortablePty {
    fn resize(&self, size: TerminalSize) -> Result<(), PtyError> {
        let pty_size = PtySize {
            rows: size.rows,
            cols: size.cols,
            pixel_width: 0,
            pixel_height: 0,
        };

        self.master
            .lock()
            .map_err(|_| PtyError::NotRunning)?
            .resize(pty_size)
            .map_err(|e| PtyError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        Ok(())
    }

    async fn write(&mut self, data: &[u8]) -> Result<(), PtyError> {
        let writer = self.writer.clone();

        // Use blocking write in a spawn_blocking context
        let data = data.to_vec();
        tokio::task::spawn_blocking(move || {
            let mut writer = writer.lock().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::Other, "Failed to lock writer")
            })?;
            writer.write_all(&data)?;
            writer.flush()?;
            Ok::<_, std::io::Error>(())
        })
        .await
        .map_err(|e| PtyError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e)))??;

        Ok(())
    }

    async fn read(&mut self, buf: &mut [u8]) -> Result<usize, PtyError> {
        let reader = self.reader.clone();
        let buf_len = buf.len();

        // Read into a local buffer first
        let result = tokio::task::spawn_blocking(move || {
            let mut reader = reader.lock().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::Other, "Failed to lock reader")
            })?;
            let mut local_buf = vec![0u8; buf_len];
            let n = reader.read(&mut local_buf)?;
            Ok::<_, std::io::Error>((local_buf, n))
        })
        .await
        .map_err(|e| PtyError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        match result {
            Ok((local_buf, n)) => {
                buf[..n].copy_from_slice(&local_buf[..n]);
                Ok(n)
            }
            Err(e) => Err(PtyError::IoError(e)),
        }
    }

    fn is_running(&self) -> bool {
        if !self.running.load(Ordering::Acquire) {
            return false;
        }

        let mut child = match self.child.lock() {
            Ok(child) => child,
            Err(_) => return false,
        };
        let Some(child) = child.as_mut() else {
            return self.running.load(Ordering::Acquire);
        };
        match child.try_wait() {
            Ok(None) => true,
            Ok(Some(_)) | Err(_) => {
                self.running.store(false, Ordering::Release);
                mark_exit_completed(&self.exit_completed);
                false
            }
        }
    }

    fn process_id(&self) -> Option<u32> {
        self.process_id
    }
}

/// Send `signal` to the shell's whole process group. The pty child is a
/// session leader, so its pid is also its process-group id and `kill(-pid)`
/// reaches every job the shell spawned. portable-pty's own `Child::kill`
/// only signals the shell pid, which lets SIGHUP-ignoring jobs (dev
/// servers, `nohup`, ssh ControlMaster) outlive the closed tab as orphans.
#[cfg(unix)]
fn signal_process_group(pid: u32, signal: i32) {
    // SAFETY: plain signal send to a process group; no pointers involved.
    // ESRCH (group already gone) is expected and ignored.
    unsafe {
        libc::kill(-(pid as i32), signal);
    }
}

fn mark_exit_completed(exit_completed: &(std::sync::Mutex<bool>, Condvar)) {
    let (completed, completed_cv) = exit_completed;
    if let Ok(mut completed) = completed.lock() {
        *completed = true;
        completed_cv.notify_all();
    }
}

fn wait_for_exit_completion(
    exit_completed: &(std::sync::Mutex<bool>, Condvar),
    timeout: Duration,
) -> bool {
    let (completed, completed_cv) = exit_completed;
    let Ok(completed) = completed.lock() else {
        return false;
    };
    if *completed {
        return true;
    }
    completed_cv
        .wait_timeout_while(completed, timeout, |completed| !*completed)
        .map(|(completed, _)| *completed)
        .unwrap_or(false)
}

fn write_pty_data(
    writer: &std::sync::Mutex<Box<dyn Write + Send>>,
    data: &[u8],
) -> std::io::Result<()> {
    let mut writer = writer
        .lock()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "PTY writer lock poisoned"))?;
    writer.write_all(data)?;
    writer.flush()
}

/// Get the default shell for the current platform
fn default_shell() -> String {
    #[cfg(target_os = "windows")]
    {
        preferred_windows_shell()
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string())
    }
}

/// Resolve the most reliable shell executable on Windows.
#[cfg(target_os = "windows")]
fn preferred_windows_shell() -> String {
    if let Ok(comspec) = std::env::var("COMSPEC") {
        if std::path::Path::new(&comspec).exists() {
            return comspec;
        }
    }

    for candidate in ["pwsh.exe", "powershell.exe", "cmd.exe"] {
        if command_exists(candidate) {
            return candidate.to_string();
        }
    }

    "cmd.exe".to_string()
}

/// Check whether an executable is available on the current PATH.
#[cfg(target_os = "windows")]
fn command_exists(command: &str) -> bool {
    let path_var = match std::env::var_os("PATH") {
        Some(path) => path,
        None => return false,
    };

    std::env::split_paths(&path_var).any(|dir| dir.join(command).exists())
}

/// PTY reader task that forwards output to a channel
pub fn spawn_pty_reader(mut pty: Box<dyn Pty>) -> mpsc::Receiver<Result<Vec<u8>, PtyError>> {
    let (tx, rx) = mpsc::channel(1024);

    tokio::spawn(async move {
        let mut buffer = vec![0u8; 4096];

        while pty.is_running() {
            match pty.read(&mut buffer).await {
                Ok(0) => {
                    // EOF
                    break;
                }
                Ok(n) => {
                    let data = buffer[..n].to_vec();
                    if tx.send(Ok(data)).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    break;
                }
            }
        }

        debug!("PTY reader task ended");
    });

    rx
}

/// PTY writer that receives input from a channel
pub fn spawn_pty_writer(mut pty: Box<dyn Pty>) -> mpsc::Sender<Vec<u8>> {
    let (tx, mut rx) = mpsc::channel::<Vec<u8>>(1024);

    tokio::spawn(async move {
        while let Some(data) = rx.recv().await {
            if let Err(e) = pty.write(&data).await {
                error!("Failed to write to PTY: {}", e);
                break;
            }
        }

        debug!("PTY writer task ended");
    });

    tx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_shell() {
        let shell = default_shell();
        assert!(!shell.is_empty());
    }

    #[test]
    fn test_pty_creation() {
        let size = TerminalSize::new(24, 80);
        let pty = PortablePty::with_default_shell(size);
        #[cfg(target_os = "windows")]
        {
            if let Err(error) = pty {
                eprintln!("Skipping PTY creation assertion on Windows: {}", error);
                return;
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            assert!(pty.is_ok());
        }
    }

    #[test]
    fn test_is_running_detects_exit_without_watcher() {
        let size = TerminalSize::new(24, 80);
        #[cfg(target_os = "windows")]
        let (program, args) = (
            std::env::var("COMSPEC").unwrap_or_else(|_| "cmd.exe".to_string()),
            vec!["/C".to_string(), "exit 0".to_string()],
        );
        #[cfg(not(target_os = "windows"))]
        let (program, args) = (
            "/bin/sh".to_string(),
            vec!["-c".to_string(), "exit 0".to_string()],
        );

        let pty = PortablePty::new_command(size, &program, &args, &HashMap::new(), None)
            .expect("short-lived PTY should start");
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            if !pty.is_running() {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("is_running stayed true after an unwatched child exited");
    }

    #[cfg(unix)]
    #[test]
    fn test_terminate_kills_sighup_ignoring_group_members() {
        // A SIGHUP-ignoring member of the shell's process group must not
        // survive `terminate`: portable-pty only signals the shell pid, so
        // terminate's process-group SIGKILL is what takes it down.
        let pid_file =
            std::env::temp_dir().join(format!("vterm-pty-test-{}", std::process::id()));
        let script = format!(
            "(trap '' HUP; exec sleep 300) </dev/null >/dev/null 2>&1 & echo $! > {}; wait",
            pid_file.display()
        );
        let mut pty = PortablePty::new_command(
            TerminalSize::new(24, 80),
            "/bin/sh",
            &["-c".to_string(), script],
            &HashMap::new(),
            None,
        )
        .expect("failed to create PTY");

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let sleep_pid = loop {
            if let Ok(contents) = std::fs::read_to_string(&pid_file) {
                if let Ok(pid) = contents.trim().parse::<i32>() {
                    break pid;
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "sleep pid file never appeared"
            );
            std::thread::sleep(Duration::from_millis(20));
        };

        pty.terminate().expect("terminate should succeed");

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let alive = loop {
            // SAFETY: signal 0 probes existence without delivering a signal.
            if unsafe { libc::kill(sleep_pid, 0) } != 0 {
                break false;
            }
            if std::time::Instant::now() >= deadline {
                break true;
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let _ = std::fs::remove_file(&pid_file);
        if alive {
            // SAFETY: best-effort cleanup so a failing test leaks nothing.
            unsafe { libc::kill(sleep_pid, libc::SIGKILL) };
            panic!("SIGHUP-ignoring process-group member survived terminate()");
        }
    }

    #[test]
    fn test_terminate_waits_for_exit_watcher() {
        let size = TerminalSize::new(24, 80);
        let mut pty = match PortablePty::with_default_shell(size) {
            Ok(pty) => pty,
            Err(error) if cfg!(target_os = "windows") => {
                eprintln!("Skipping PTY termination assertion on Windows: {error}");
                return;
            }
            Err(error) => panic!("failed to create PTY: {error}"),
        };
        let (exit_tx, exit_rx) = std::sync::mpsc::channel();
        pty.spawn_exit_watcher(move || {
            let _ = exit_tx.send(());
        });

        pty.terminate()
            .expect("terminate should confirm that the watcher reaped the child");
        exit_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("exit watcher callback should complete");
        assert!(!pty.is_running());
    }
}
