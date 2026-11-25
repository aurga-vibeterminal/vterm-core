//! Terminal Core - Core terminal functionality
//!
//! This crate provides:
//! - Protocol message definitions and serialization
//! - Noise Protocol encryption for secure communication
//! - PTY abstraction for cross-platform terminal support
//! - Terminal emulator (re-exported from the standalone `vterm-emulator`
//!   crate, which can also be embedded directly in mobile apps)

pub mod crypto;
pub mod folder_stream;
pub mod protocol;
#[cfg(feature = "pty")]
pub mod pty;

// The terminal emulator lives in its own crate (`vterm-emulator`) so it can
// be embedded in mobile applications without pulling in the rest of
// vterm-core. Re-exported here under the historical `emulator` path.
pub use vterm_emulator as emulator;

// Re-export commonly used types
pub use crypto::{
    generate_keypair, EncryptedStream, MessageCrypto, MessageCryptoHandshake, TRANSPORT_TAG_BINARY,
    TRANSPORT_TAG_TEXT,
};
pub use emulator::Terminal;
pub use protocol::Message;
#[cfg(feature = "pty")]
pub use pty::{PortablePty, Pty, PtyError};
