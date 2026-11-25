//! Synthetic heavy TUI for the echo-latency harness: floods full-screen
//! truecolor VT frames continuously and echoes each keystroke with an
//! `echo:<char>` marker. Simulates a code-agent TUI while it is streaming
//! long output.
//!
//! usage: flood [frame_interval_ms] [cols] [rows]

use std::io::{Read, Write};

fn main() {
    let interval: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let cols: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let rows: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(80);

    // Flood thread: write one full-screen truecolor frame per interval.
    // The blocking write applies natural backpressure when the pipe is full.
    std::thread::spawn(move || {
        let mut n: u32 = 0;
        let mut frame = String::new();
        loop {
            frame.clear();
            frame.push_str("\x1b[H");
            for r in 0..rows {
                for c in 0..cols {
                    let rr = ((r * 3) as u32 + n) % 256;
                    let gg = ((c * 5) as u32 + n) % 256;
                    let bb = (r as u32 + c as u32 + n) % 256;
                    frame.push_str(&format!("\x1b[48;2;{rr};{gg};{bb}m "));
                }
                frame.push_str("\r\n");
            }
            n = n.wrapping_add(1);
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            if lock.write_all(frame.as_bytes()).is_err() {
                return;
            }
            let _ = lock.flush();
            drop(lock);
            std::thread::sleep(std::time::Duration::from_millis(interval));
        }
    });

    // Echo thread (main): one marker per keystroke read.
    let mut stdin = std::io::stdin();
    let mut buf = [0u8; 64];
    loop {
        match stdin.read(&mut buf) {
            Ok(0) | Err(_) => return,
            Ok(n) => {
                for &b in &buf[..n] {
                    let stdout = std::io::stdout();
                    let mut lock = stdout.lock();
                    if lock
                        .write_all(format!("\x1b[2K\necho:{}\n", b as char).as_bytes())
                        .is_err()
                    {
                        return;
                    }
                    let _ = lock.flush();
                }
            }
        }
    }
}
