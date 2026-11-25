use snow::{Builder, HandshakeState, TransportState};
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf};
use tracing::{debug, trace};

/// Noise protocol pattern and parameters
const NOISE_PATTERN: &str = "Noise_XX_25519_ChaChaPoly_BLAKE2s";

/// Size of the Noise protocol header (2 bytes for length)
const NOISE_HEADER_LEN: usize = 2;

/// Maximum payload size for a single Noise message
const MAX_NOISE_PAYLOAD_SIZE: usize = 65535;
const MESSAGE_ENCRYPTION_HANDSHAKE_MAGIC: &[u8; 4] = b"VNH1";
const MESSAGE_ENCRYPTION_FRAME_MAGIC: &[u8; 4] = b"VNE1";
/// Fragmented-message magic: a message larger than one Noise frame is
/// split into chunks, each encrypted as its own Noise message and
/// reassembled after decryption.
const MESSAGE_ENCRYPTION_FRAG_MAGIC: &[u8; 4] = b"VNF1";
/// Fragment header after the magic: msg_id u32 BE + idx u16 BE + count u16 BE.
const FRAG_HEADER_LEN: usize = 8;
/// Plaintext bytes per fragment (leaves comfortable room below the
/// Noise frame limit after the 16-byte auth tag + headers).
const FRAG_CHUNK_SIZE: usize = 65_000;

pub const TRANSPORT_TAG_TEXT: u8 = 1;
pub const TRANSPORT_TAG_BINARY: u8 = 2;
/// Multiplexed v10 request frame.
pub const TRANSPORT_TAG_REQUEST: u8 = 3;

/// WebSocket-message Noise handshake used after the plaintext protocol
/// Hello/HelloAck negotiation.
pub struct MessageCryptoHandshake {
    state: HandshakeState,
}

impl MessageCryptoHandshake {
    pub fn new_initiator() -> anyhow::Result<Self> {
        let builder = Builder::new(NOISE_PATTERN.parse()?);
        let static_key = builder.generate_keypair()?;
        Ok(Self {
            state: builder
                .local_private_key(&static_key.private)
                .build_initiator()?,
        })
    }

    pub fn new_responder() -> anyhow::Result<Self> {
        let builder = Builder::new(NOISE_PATTERN.parse()?);
        let static_key = builder.generate_keypair()?;
        Ok(Self {
            state: builder
                .local_private_key(&static_key.private)
                .build_responder()?,
        })
    }

    pub fn write_frame(&mut self) -> anyhow::Result<Vec<u8>> {
        let mut msg = vec![0u8; 1024];
        let len = self.state.write_message(&[], &mut msg)?;
        let mut frame = Vec::with_capacity(MESSAGE_ENCRYPTION_HANDSHAKE_MAGIC.len() + len);
        frame.extend_from_slice(MESSAGE_ENCRYPTION_HANDSHAKE_MAGIC);
        frame.extend_from_slice(&msg[..len]);
        Ok(frame)
    }

    pub fn read_frame(&mut self, frame: &[u8]) -> anyhow::Result<()> {
        let payload = frame
            .strip_prefix(MESSAGE_ENCRYPTION_HANDSHAKE_MAGIC)
            .ok_or_else(|| anyhow::anyhow!("Invalid message-encryption handshake frame"))?;
        let mut out = vec![0u8; 1024];
        self.state.read_message(payload, &mut out)?;
        Ok(())
    }

    pub fn into_transport(self) -> anyhow::Result<MessageCrypto> {
        Ok(MessageCrypto {
            transport: self.state.into_transport_mode()?,
            next_frag_id: 0,
            frag_buffers: std::collections::HashMap::new(),
        })
    }
}

/// Reassembly state for one fragmented message.
struct FragBuffer {
    count: u16,
    received: u16,
    chunks: Vec<Option<Vec<u8>>>,
}

/// Encrypts/decrypts individual WebSocket message payloads after the Noise
/// transport state has been established.
pub struct MessageCrypto {
    transport: TransportState,
    next_frag_id: u32,
    frag_buffers: std::collections::HashMap<u32, FragBuffer>,
}

impl MessageCrypto {
    /// Encrypt a message into one or more frames. Messages above the
    /// Noise frame limit are fragmented (`VNF1` chunks); each fragment
    /// is its own Noise message (ordered nonces require in-order
    /// delivery, which the transport guarantees).
    pub fn encrypt_frame(&mut self, plaintext: &[u8]) -> anyhow::Result<Vec<Vec<u8>>> {
        if plaintext.len() <= MAX_NOISE_PAYLOAD_SIZE {
            return Ok(vec![self.encrypt_single(plaintext)?]);
        }
        let id = self.next_frag_id;
        self.next_frag_id = self.next_frag_id.wrapping_add(1);
        let count = plaintext.len().div_ceil(FRAG_CHUNK_SIZE) as u16;
        let mut frames = Vec::with_capacity(count as usize);
        for (idx, chunk) in plaintext.chunks(FRAG_CHUNK_SIZE).enumerate() {
            let mut frame = Vec::with_capacity(4 + FRAG_HEADER_LEN + chunk.len() + 16);
            frame.extend_from_slice(MESSAGE_ENCRYPTION_FRAG_MAGIC);
            frame.extend_from_slice(&id.to_be_bytes());
            frame.extend_from_slice(&(idx as u16).to_be_bytes());
            frame.extend_from_slice(&count.to_be_bytes());
            frame.extend_from_slice(&self.encrypt_single(chunk)?);
            frames.push(frame);
        }
        Ok(frames)
    }

    /// Decrypt a frame. Returns `None` while a fragmented message is
    /// still incomplete (more chunks to come).
    pub fn decrypt_frame(&mut self, frame: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
        if let Some(payload) = frame.strip_prefix(MESSAGE_ENCRYPTION_FRAME_MAGIC) {
            return self.decrypt_single(payload).map(Some);
        }
        let Some(rest) = frame.strip_prefix(MESSAGE_ENCRYPTION_FRAG_MAGIC) else {
            anyhow::bail!("Invalid encrypted message frame");
        };
        if rest.len() < FRAG_HEADER_LEN {
            anyhow::bail!("Fragmented frame too short for header");
        }
        let id = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]);
        let idx = u16::from_be_bytes([rest[4], rest[5]]) as usize;
        let count = u16::from_be_bytes([rest[6], rest[7]]);
        let chunk = self.decrypt_single(&rest[FRAG_HEADER_LEN..])?;

        let buf = self.frag_buffers.entry(id).or_insert_with(|| FragBuffer {
            count,
            received: 0,
            chunks: vec![None; count as usize],
        });
        if buf.count != count || idx >= count as usize {
            anyhow::bail!("Invalid fragment header (id={id} idx={idx} count={count})");
        }
        if buf.chunks[idx].is_none() {
            buf.chunks[idx] = Some(chunk);
            buf.received += 1;
        }
        if buf.received == buf.count {
            let buf = self
                .frag_buffers
                .remove(&id)
                .expect("frag buffer just used");
            let mut out = Vec::new();
            for chunk in buf.chunks.into_iter().flatten() {
                out.extend_from_slice(&chunk);
            }
            return Ok(Some(out));
        }
        Ok(None)
    }

    fn encrypt_single(&mut self, plaintext: &[u8]) -> anyhow::Result<Vec<u8>> {
        let mut ciphertext = vec![0u8; plaintext.len() + 16];
        let len = self.transport.write_message(plaintext, &mut ciphertext)?;
        let mut frame = Vec::with_capacity(MESSAGE_ENCRYPTION_FRAME_MAGIC.len() + len);
        frame.extend_from_slice(MESSAGE_ENCRYPTION_FRAME_MAGIC);
        frame.extend_from_slice(&ciphertext[..len]);
        Ok(frame)
    }

    fn decrypt_single(&mut self, payload: &[u8]) -> anyhow::Result<Vec<u8>> {
        let mut plaintext = vec![0u8; payload.len()];
        let len = self.transport.read_message(payload, &mut plaintext)?;
        plaintext.truncate(len);
        Ok(plaintext)
    }
}

pub fn is_message_encryption_handshake_frame(frame: &[u8]) -> bool {
    frame.starts_with(MESSAGE_ENCRYPTION_HANDSHAKE_MAGIC)
}

pub fn is_encrypted_message_frame(frame: &[u8]) -> bool {
    frame.starts_with(MESSAGE_ENCRYPTION_FRAME_MAGIC)
        || frame.starts_with(MESSAGE_ENCRYPTION_FRAG_MAGIC)
}

/// Encrypted stream wrapper that handles Noise Protocol encryption
pub struct EncryptedStream<T> {
    inner: T,
    transport: TransportState,
    read_buffer: Vec<u8>,
    write_buffer: Vec<u8>,
    is_initiator: bool,
    read_buffer_pos: usize,
    read_buffer_len: usize,
    encrypted_read_buffer: Vec<u8>,
    encrypted_read_len: usize,
    encrypted_read_pos: usize,
    read_header: [u8; NOISE_HEADER_LEN],
    read_header_pos: usize,
    write_buffer_pos: usize,
}

impl<T> EncryptedStream<T>
where
    T: AsyncReadExt + AsyncWriteExt + Unpin,
{
    /// Create a new encrypted stream as the initiator (client)
    pub async fn new_initiator(mut inner: T) -> anyhow::Result<Self> {
        debug!("Starting Noise handshake as initiator");

        let builder = Builder::new(NOISE_PATTERN.parse()?);
        let static_key = builder.generate_keypair()?;

        let mut handshake = builder
            .local_private_key(&static_key.private)
            .build_initiator()?;

        // -> e
        let mut msg = [0u8; 1024];
        let len = handshake.write_message(&[], &mut msg)?;
        inner.write_all(&(len as u16).to_be_bytes()).await?;
        inner.write_all(&msg[..len]).await?;
        debug!("Sent first handshake message ({} bytes)", len);

        // <- e, ee, s, es
        let mut len_buf = [0u8; 2];
        inner.read_exact(&mut len_buf).await?;
        let len = u16::from_be_bytes(len_buf) as usize;
        let mut response = vec![0u8; len];
        inner.read_exact(&mut response).await?;

        let mut payload = [0u8; 1024];
        handshake.read_message(&response, &mut payload)?;
        debug!("Received and processed second handshake message");

        // -> s, se
        let len = handshake.write_message(&[], &mut msg)?;
        inner.write_all(&(len as u16).to_be_bytes()).await?;
        inner.write_all(&msg[..len]).await?;
        // debug!("Sent third handshake message ({} bytes)", len);

        let transport = handshake.into_transport_mode()?;
        // debug!("Noise handshake completed successfully");

        Ok(Self {
            inner,
            transport,
            read_buffer: Vec::with_capacity(MAX_NOISE_PAYLOAD_SIZE),
            write_buffer: Vec::with_capacity(MAX_NOISE_PAYLOAD_SIZE),
            is_initiator: true,
            read_buffer_pos: 0,
            read_buffer_len: 0,
            encrypted_read_buffer: Vec::new(),
            encrypted_read_len: 0,
            encrypted_read_pos: 0,
            read_header: [0; NOISE_HEADER_LEN],
            read_header_pos: 0,
            write_buffer_pos: 0,
        })
    }

    /// Create a new encrypted stream as the responder (server)
    pub async fn new_responder(mut inner: T) -> anyhow::Result<Self> {
        debug!("Starting Noise handshake as responder");

        let builder = Builder::new(NOISE_PATTERN.parse()?);
        let static_key = builder.generate_keypair()?;

        let mut handshake = builder
            .local_private_key(&static_key.private)
            .build_responder()?;

        // <- e
        let mut len_buf = [0u8; 2];
        inner.read_exact(&mut len_buf).await?;
        let len = u16::from_be_bytes(len_buf) as usize;
        let mut request = vec![0u8; len];
        inner.read_exact(&mut request).await?;

        let mut payload = [0u8; 1024];
        handshake.read_message(&request, &mut payload)?;
        debug!("Received and processed first handshake message");

        // -> e, ee, s, es
        let mut msg = [0u8; 1024];
        let len = handshake.write_message(&[], &mut msg)?;
        inner.write_all(&(len as u16).to_be_bytes()).await?;
        inner.write_all(&msg[..len]).await?;
        debug!("Sent second handshake message ({} bytes)", len);

        // <- s, se
        inner.read_exact(&mut len_buf).await?;
        let len = u16::from_be_bytes(len_buf) as usize;
        let mut final_msg = vec![0u8; len];
        inner.read_exact(&mut final_msg).await?;

        handshake.read_message(&final_msg, &mut payload)?;
        // debug!("Received and processed third handshake message");

        let transport = handshake.into_transport_mode()?;
        // debug!("Noise handshake completed successfully");

        Ok(Self {
            inner,
            transport,
            read_buffer: Vec::with_capacity(MAX_NOISE_PAYLOAD_SIZE),
            write_buffer: Vec::with_capacity(MAX_NOISE_PAYLOAD_SIZE),
            is_initiator: false,
            read_buffer_pos: 0,
            read_buffer_len: 0,
            encrypted_read_buffer: Vec::new(),
            encrypted_read_len: 0,
            encrypted_read_pos: 0,
            read_header: [0; NOISE_HEADER_LEN],
            read_header_pos: 0,
            write_buffer_pos: 0,
        })
    }

    /// Get the underlying stream (for advanced use)
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Check if this stream is the initiator
    pub fn is_initiator(&self) -> bool {
        self.is_initiator
    }

    fn reset_read_frame(&mut self) {
        self.read_header_pos = 0;
        self.encrypted_read_len = 0;
        self.encrypted_read_pos = 0;
        self.encrypted_read_buffer.clear();
    }
}

impl<T> AsyncRead for EncryptedStream<T>
where
    T: AsyncRead + AsyncWrite + Unpin,
{
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if self.read_buffer_pos < self.read_buffer_len {
            let available = self.read_buffer_len - self.read_buffer_pos;
            let to_copy = available.min(buf.remaining());
            buf.put_slice(&self.read_buffer[self.read_buffer_pos..self.read_buffer_pos + to_copy]);
            self.read_buffer_pos += to_copy;
            return Poll::Ready(Ok(()));
        }

        while self.read_header_pos < NOISE_HEADER_LEN {
            let header_start = self.read_header_pos;
            let mut header_chunk = [0u8; NOISE_HEADER_LEN];
            header_chunk[..header_start].copy_from_slice(&self.read_header[..header_start]);
            let mut header_buf = ReadBuf::new(&mut header_chunk[header_start..]);

            match Pin::new(&mut self.inner).poll_read(cx, &mut header_buf) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Ready(Ok(())) => {
                    let bytes_read = header_buf.filled().len();
                    self.read_header[header_start..header_start + bytes_read]
                        .copy_from_slice(header_buf.filled());
                    if bytes_read == 0 {
                        if self.read_header_pos == 0 {
                            return Poll::Ready(Ok(()));
                        }

                        self.reset_read_frame();
                        return Poll::Ready(Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "Could not read length prefix",
                        )));
                    }

                    self.read_header_pos += bytes_read;
                }
            }
        }

        if self.encrypted_read_len == 0 {
            let encrypted_len = u16::from_be_bytes(self.read_header) as usize;
            if encrypted_len > MAX_NOISE_PAYLOAD_SIZE {
                self.reset_read_frame();
                return Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Noise message too large: {} bytes", encrypted_len),
                )));
            }

            self.encrypted_read_len = encrypted_len;
            self.encrypted_read_pos = 0;
            self.encrypted_read_buffer.resize(encrypted_len, 0);
        }

        while self.encrypted_read_pos < self.encrypted_read_len {
            let data_start = self.encrypted_read_pos;
            let data_end = self.encrypted_read_len;
            let mut encrypted_read_buffer = std::mem::take(&mut self.encrypted_read_buffer);
            let mut encrypted_buf = ReadBuf::new(&mut encrypted_read_buffer[data_start..data_end]);

            match Pin::new(&mut self.inner).poll_read(cx, &mut encrypted_buf) {
                Poll::Pending => {
                    self.encrypted_read_buffer = encrypted_read_buffer;
                    return Poll::Pending;
                }
                Poll::Ready(Err(error)) => {
                    self.encrypted_read_buffer = encrypted_read_buffer;
                    return Poll::Ready(Err(error));
                }
                Poll::Ready(Ok(())) => {
                    let bytes_read = encrypted_buf.filled().len();
                    self.encrypted_read_buffer = encrypted_read_buffer;
                    if bytes_read == 0 {
                        self.reset_read_frame();
                        return Poll::Ready(Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "Could not read full encrypted message",
                        )));
                    }

                    self.encrypted_read_pos += bytes_read;
                }
            }
        }

        let encrypted_read_len = self.encrypted_read_len;
        let encrypted_read_buffer = std::mem::take(&mut self.encrypted_read_buffer);
        let mut decrypted_buffer = vec![0u8; self.encrypted_read_len];
        let decrypted_len = match self.transport.read_message(
            &encrypted_read_buffer[..encrypted_read_len],
            &mut decrypted_buffer,
        ) {
            Ok(length) => length,
            Err(error) => {
                self.reset_read_frame();
                return Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Decryption failed: {}", error),
                )));
            }
        };

        self.read_buffer = decrypted_buffer;
        self.read_buffer_len = decrypted_len;
        self.read_buffer_pos = 0;
        self.reset_read_frame();

        let to_copy = decrypted_len.min(buf.remaining());
        buf.put_slice(&self.read_buffer[..to_copy]);
        self.read_buffer_pos = to_copy;

        trace!("Decrypted {} bytes", decrypted_len);
        Poll::Ready(Ok(()))
    }
}

impl<T> AsyncWrite for EncryptedStream<T>
where
    T: AsyncRead + AsyncWrite + Unpin,
{
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        while self.write_buffer_pos < self.write_buffer.len() {
            let pending_data = self.write_buffer[self.write_buffer_pos..].to_vec();
            match Pin::new(&mut self.inner).poll_write(cx, &pending_data) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Ready(Ok(0)) => {
                    return Poll::Ready(Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "Failed to flush encrypted message",
                    )));
                }
                Poll::Ready(Ok(bytes_written)) => {
                    self.write_buffer_pos += bytes_written;
                }
            }
        }

        self.write_buffer.clear();
        self.write_buffer_pos = 0;

        let max_encrypted_len = buf.len() + 16;
        let mut write_buffer = vec![0u8; NOISE_HEADER_LEN + max_encrypted_len];

        let encrypted_len = match self
            .transport
            .write_message(buf, &mut write_buffer[NOISE_HEADER_LEN..])
        {
            Ok(len) => len,
            Err(error) => {
                return Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Encryption failed: {}", error),
                )));
            }
        };

        write_buffer[..NOISE_HEADER_LEN].copy_from_slice(&(encrypted_len as u16).to_be_bytes());
        self.write_buffer = write_buffer[..NOISE_HEADER_LEN + encrypted_len].to_vec();
        self.write_buffer_pos = 0;

        while self.write_buffer_pos < self.write_buffer.len() {
            let pending_data = self.write_buffer[self.write_buffer_pos..].to_vec();
            match Pin::new(&mut self.inner).poll_write(cx, &pending_data) {
                Poll::Pending => {
                    trace!("Buffered encrypted write of {} bytes", buf.len());
                    return Poll::Ready(Ok(buf.len()));
                }
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Ready(Ok(0)) => {
                    return Poll::Ready(Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "Failed to write encrypted message",
                    )));
                }
                Poll::Ready(Ok(bytes_written)) => {
                    self.write_buffer_pos += bytes_written;
                }
            }
        }

        self.write_buffer.clear();
        self.write_buffer_pos = 0;
        trace!("Encrypted and wrote {} bytes", buf.len());
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        while self.write_buffer_pos < self.write_buffer.len() {
            let pending_data = self.write_buffer[self.write_buffer_pos..].to_vec();
            match Pin::new(&mut self.inner).poll_write(cx, &pending_data) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Ready(Ok(0)) => {
                    return Poll::Ready(Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "Failed to flush encrypted message",
                    )));
                }
                Poll::Ready(Ok(bytes_written)) => {
                    self.write_buffer_pos += bytes_written;
                }
            }
        }

        self.write_buffer.clear();
        self.write_buffer_pos = 0;
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

/// Generate a new static keypair for Noise protocol
pub fn generate_keypair() -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
    let builder = Builder::new(NOISE_PATTERN.parse()?);
    let keypair = builder.generate_keypair()?;
    Ok((keypair.public.to_vec(), keypair.private.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_keypair_generation() {
        let (public, private) = generate_keypair().unwrap();
        assert_eq!(public.len(), 32);
        assert_eq!(private.len(), 32);
    }

    // Note: Full handshake test would require actual TCP connection
    // This is tested in integration tests
}
