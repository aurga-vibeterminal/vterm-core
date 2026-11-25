//! Keystroke-echo latency probe for the raw PTY layer (no client, no
//! network). Spawns a program inside a ConPTY and measures:
//!   t_first — input handed over → first output bytes readable
//!   t_last  — input handed over → output settled (100ms quiet window)
//!
//! Two input paths:
//!   (default)      write the PTY master directly (write_all + flush)
//!   --via-server   vterm-core PortablePty + spawn_writer: mpsc channel →
//!                  dedicated writer thread → write_all+flush, and a reader
//!                  thread with the same Mutex+4096B pattern as the server's
//!                  session_io — an exact replica of vterm-server's hot path.
//!
//! Usage:
//!   cargo run -p vterm-core --example pty_echo_latency -- [options] <program> [args...]
//! Options:
//!   --cols N        (default 200)
//!   --rows N        (default 50)
//!   --repeats N     (default 8)
//!   --key S         (default "1"; escapes: \r \n \t \e \xNN)
//!   --pre S         repeatable: keys sent before measuring, with a settle
//!                   drain after each (800ms quiet, 60s cap)
//!   --cwd DIR       working directory for the child
//!   --env K=V       repeatable
//!   --fill-lines N  fill scrollback via cmd.exe for-loop (or --fill-cmd)
//!   --fill-cmd S    custom fill command (Enter appended)
//!   --via-server    use the server's writer/reader thread structure
//!
//! Examples:
//!   ... -- cmd.exe
//!   ... -- --via-server --pre "hi\r" C:\Users\user\.kimi-code\bin\kimi.EXE

use portable_pty::{CommandBuilder, NativePtySystem, PtySize, PtySystem};
use std::io::{Read, Write};
use std::time::{Duration, Instant};

/// Minimal escape decoding for --key/--pre: \r \n \t \e \xNN
fn unescape(s: &str) -> Vec<u8> {
    let mut out = Vec::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut b = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut b).as_bytes());
            continue;
        }
        match chars.next() {
            Some('r') => out.push(b'\r'),
            Some('n') => out.push(b'\n'),
            Some('t') => out.push(b'\t'),
            Some('e') => out.push(0x1b),
            Some('x') => {
                let h: String = chars.by_ref().take(2).collect();
                out.push(u8::from_str_radix(&h, 16).unwrap_or(b'?'));
            }
            Some(other) => out.push(other as u8),
            None => {}
        }
    }
    out
}

fn main() {
    let mut cols: u16 = 200;
    let mut rows: u16 = 50;
    let mut repeats: usize = 8;
    let mut key = b"1".to_vec();
    let mut cwd: Option<String> = None;
    let mut envs: Vec<(String, String)> = Vec::new();
    let mut pre: Vec<Vec<u8>> = Vec::new();
    let mut fill_lines: usize = 0;
    let mut fill_cmd: Option<String> = None;
    /// Sleep this long right before each measured keystroke, simulating the
    /// user returning to an idle session.
    let mut idle_ms: u64 = 0;
    /// Echo marker to look for in the output stream (for apps that flood
    /// output continuously, where "first chunk after write" is meaningless).
    let mut marker: Option<Vec<u8>> = None;
    let mut via_server = false;
    let mut positional: Vec<String> = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--cols" => cols = args.next().expect("--cols value").parse().unwrap(),
            "--rows" => rows = args.next().expect("--rows value").parse().unwrap(),
            "--repeats" => repeats = args.next().expect("--repeats value").parse().unwrap(),
            "--key" => key = unescape(&args.next().expect("--key value")),
            "--pre" => pre.push(unescape(&args.next().expect("--pre value"))),
            "--cwd" => cwd = Some(args.next().expect("--cwd value")),
            "--env" => {
                let kv = args.next().expect("--env KEY=VALUE");
                let (k, v) = kv.split_once('=').expect("--env KEY=VALUE");
                envs.push((k.to_string(), v.to_string()));
            }
            "--fill-lines" => {
                fill_lines = args.next().expect("--fill-lines value").parse().unwrap()
            }
            "--fill-cmd" => fill_cmd = Some(args.next().expect("--fill-cmd value")),
            "--idle-ms" => idle_ms = args.next().expect("--idle-ms value").parse().unwrap(),
            "--via-server" => via_server = true,
            "--marker" => marker = Some(args.next().expect("--marker value").into_bytes()),
            "--" => {} // conventional separator before the program
            _ => positional.push(arg),
        }
    }
    let Some(program) = positional.first().cloned() else {
        eprintln!("usage: pty_echo_latency [--cols N] [--rows N] [--repeats N] [--key S] [--pre S]... [--cwd DIR] [--env K=V]... [--fill-lines N | --fill-cmd S] [--via-server] <program> [args...]");
        std::process::exit(2);
    };
    let child_args: Vec<String> = positional[1..].to_vec();

    // Output chunks from the reader thread.
    let (out_tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
    // Input path: direct master write, or the server's spawn_writer channel.
    let mut write_input: Box<dyn FnMut(&[u8])>;
    let mut kill_child: Box<dyn FnMut()>;

    if via_server {
        let size = vterm_common::TerminalSize::new(rows, cols);
        let env_map: std::collections::HashMap<String, String> = envs.iter().cloned().collect();
        let pty = vterm_core::pty::PortablePty::new_command(
            size,
            &program,
            &child_args,
            &env_map,
            cwd.as_deref(),
        )
        .expect("PortablePty::new_command");
        let writer_tx = pty.spawn_writer(None);
        let reader = pty.reader();
        // Reader thread identical to session_io.rs: Mutex + blocking read
        // into a 4096-byte buffer.
        std::thread::spawn(move || {
            let mut buffer = [0u8; 4096];
            loop {
                let read_result = {
                    let mut reader = match reader.lock() {
                        Ok(r) => r,
                        Err(_) => break,
                    };
                    reader.read(&mut buffer)
                };
                match read_result {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if out_tx.send(buffer[..n].to_vec()).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        write_input = Box::new(move |data: &[u8]| {
            writer_tx
                .blocking_send(data.to_vec())
                .expect("writer channel closed");
        });
        kill_child = Box::new(move || {
            let _ = &pty; // keep alive until here
        });
    } else {
        let pty_system = NativePtySystem::default();
        let pair = pty_system
            .openpty(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .expect("openpty");

        let mut cmd = CommandBuilder::new(&program);
        for a in &child_args {
            cmd.arg(a);
        }
        if let Some(dir) = &cwd {
            cmd.cwd(dir);
        }
        for (k, v) in &envs {
            cmd.env(k, v);
        }
        let mut child = pair.slave.spawn_command(cmd).expect("spawn");

        let mut reader = pair.master.try_clone_reader().expect("reader");
        let mut writer = pair.master.take_writer().expect("writer");

        std::thread::spawn(move || loop {
            let mut b = [0u8; 8192];
            match reader.read(&mut b) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if out_tx.send(b[..n].to_vec()).is_err() {
                        break;
                    }
                }
            }
        });
        write_input = Box::new(move |data: &[u8]| {
            writer.write_all(data).expect("write");
            writer.flush().expect("flush");
        });
        kill_child = Box::new(move || {
            let _ = child.kill();
        });
    }

    // Drain the startup banner: read until 800ms of silence (15s cap).
    let dump_banner = std::env::var("DUMP_BANNER").is_ok();
    let start = Instant::now();
    let mut last_byte = Instant::now();
    let mut banner_bytes = 0usize;
    let mut banner = Vec::new();
    while last_byte.elapsed() < Duration::from_millis(800)
        && start.elapsed() < Duration::from_secs(15)
    {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => {
                banner_bytes += chunk.len();
                if dump_banner {
                    banner.extend_from_slice(&chunk);
                }
                last_byte = Instant::now();
            }
            Err(_) => {}
        }
    }
    println!("startup drained: {} bytes", banner_bytes);
    if dump_banner {
        let tail: Vec<u8> = banner.iter().rev().take(3000).rev().copied().collect();
        println!(
            "--- banner tail (escaped) ---\n{}\n---",
            String::from_utf8_lossy(&tail).escape_debug()
        );
    }

    for pre_keys in &pre {
        write_input(pre_keys);
        // Wait for the pre-keys' output to settle (800ms quiet, 60s cap —
        // LLM responses stream slowly).
        let pre_start = Instant::now();
        let mut pre_last = Instant::now();
        let mut pre_bytes = 0usize;
        while pre_last.elapsed() < Duration::from_millis(800)
            && pre_start.elapsed() < Duration::from_secs(60)
        {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(chunk) => {
                    pre_bytes += chunk.len();
                    pre_last = Instant::now();
                }
                Err(_) => {}
            }
        }
        println!("pre sent {:?}: drained {} bytes", pre_keys, pre_bytes);
    }

    // Fill the screen/scrollback to simulate a long-running TUI session.
    let fill = match (fill_cmd, fill_lines) {
        (Some(cmd), _) => Some(cmd),
        (None, n) if n > 0 => Some(format!("for /l %i in (1,1,{}) do @echo fill-line-%i", n)),
        _ => None,
    };
    if let Some(cmd) = fill {
        let t_fill = Instant::now();
        write_input(cmd.as_bytes());
        write_input(b"\r");
        let mut last = Instant::now();
        let mut bytes = 0usize;
        while last.elapsed() < Duration::from_millis(1500)
            && t_fill.elapsed() < Duration::from_secs(120)
        {
            match rx.recv_timeout(Duration::from_millis(200)) {
                Ok(chunk) => {
                    bytes += chunk.len();
                    last = Instant::now();
                }
                Err(_) => {}
            }
        }
        println!(
            "fill drained: {} bytes in {:?} (scrollback filled)",
            bytes,
            t_fill.elapsed()
        );
    }

    let mut t_firsts = Vec::new();
    let mut t_lasts = Vec::new();
    for i in 0..repeats {
        // Drain leftovers.
        while rx.try_recv().is_ok() {}
        if idle_ms > 0 {
            std::thread::sleep(Duration::from_millis(idle_ms));
            while rx.try_recv().is_ok() {}
        }

        let t0 = Instant::now();
        write_input(&key);

        let mut t_first = None;
        let mut bytes = 0usize;
        let mut chunks = 0usize;
        // In marker mode the output stream never goes quiet (the app is
        // flooding), so stop as soon as the echo marker has been seen.
        let mut seen: Vec<u8> = Vec::new();
        loop {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(chunk) => {
                    bytes += chunk.len();
                    chunks += 1;
                    if let Some(m) = &marker {
                        seen.extend_from_slice(&chunk);
                        if seen.windows(m.len()).any(|w| w == m.as_slice()) {
                            t_first = Some(t0.elapsed());
                            break;
                        }
                        // Keep only the tail needed for cross-chunk matches.
                        let keep = m.len().saturating_sub(1);
                        if seen.len() > keep.max(4096) {
                            let split = seen.len() - keep.max(4096);
                            seen.drain(..split);
                        }
                    } else if t_first.is_none() {
                        t_first = Some(t0.elapsed());
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if marker.is_none() {
                        break;
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
            if t0.elapsed() > Duration::from_secs(5) {
                break;
            }
        }
        let t_last = t0.elapsed();
        match t_first {
            Some(tf) => {
                t_firsts.push(tf);
                let settled = if marker.is_some() {
                    tf
                } else {
                    t_last - Duration::from_millis(100) // subtract quiet window
                };
                t_lasts.push(settled);
                println!(
                    "iter {}: t_first={:>7.1?}  t_last={:>7.1?}  bytes={} chunks={}",
                    i, tf, settled, bytes, chunks
                );
            }
            None => println!("iter {}: NO OUTPUT within 5s", i),
        }
        // Let the app settle between iterations.
        if marker.is_none() {
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    if !t_firsts.is_empty() {
        let avg_first = t_firsts.iter().sum::<Duration>() / t_firsts.len() as u32;
        let avg_last = t_lasts.iter().sum::<Duration>() / t_lasts.len() as u32;
        let max_first = t_firsts.iter().max().copied().unwrap();
        println!(
            "== {}x{} {} (via_server={}): avg t_first={:?} (max {:?}), avg t_last={:?} over {} iters",
            cols,
            rows,
            program,
            via_server,
            avg_first,
            max_first,
            avg_last,
            t_firsts.len()
        );
    }

    kill_child();
}
