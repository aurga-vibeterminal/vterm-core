use snow::params::NoiseParams;
/// Noise_XX test vector generator for iOS interop testing.
///
/// Runs a complete Noise_XX handshake with FIXED static + ephemeral keys
/// (via snow's testing-only API), so the Swift client can verify its hand-rolled
/// implementation byte for byte.
///
/// Run: cargo test --test noise_test_vectors -- --nocapture
use snow::Builder;

fn to_hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{:02x}", b)).collect()
}

#[test]
fn generate_noise_xx_test_vectors() -> anyhow::Result<()> {
    let pattern = "Noise_XX_25519_ChaChaPoly_BLAKE2s";
    let params: NoiseParams = pattern.parse()?;

    // Fixed static keypairs.
    let init_static_priv: [u8; 32] = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
        0x1f, 0x20,
    ];
    let resp_static_priv: [u8; 32] = [
        0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
        0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e,
        0x3f, 0x40,
    ];
    // Fixed ephemeral private keys (for deterministic handshake messages).
    let init_ephem_priv: [u8; 32] = [
        0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f,
        0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e,
        0x5f, 0x60,
    ];
    let resp_ephem_priv: [u8; 32] = [
        0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f,
        0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e,
        0x7f, 0x80,
    ];

    // Derive public keys from private.
    let init_kp = Builder::new(params.clone())
        .fixed_ephemeral_key_for_testing_only(&init_ephem_priv)
        .local_private_key(&init_static_priv)
        .build_initiator()?;
    let init_static_pub = init_kp.get_remote_static(); // not useful here

    // For public key derivation: use generate_keypair with fixed ephemeral.
    // Actually, snow derives the static public from the private during build.
    // To get the public, we do a throwaway handshake and extract.
    let mut init = Builder::new(params.clone())
        .fixed_ephemeral_key_for_testing_only(&init_ephem_priv)
        .local_private_key(&init_static_priv)
        .build_initiator()?;
    let mut resp = Builder::new(params.clone())
        .fixed_ephemeral_key_for_testing_only(&resp_ephem_priv)
        .local_private_key(&resp_static_priv)
        .build_responder()?;

    // Message 1: -> e
    let mut buf = vec![0u8; 1024];
    let n1 = init.write_message(&[], &mut buf)?;
    let msg1 = buf[..n1].to_vec();
    println!("DEBUG msg1 ({} bytes): {}", msg1.len(), to_hex(&msg1));

    // Responder reads msg1
    let mut rbuf = vec![0u8; 1024];
    resp.read_message(&msg1, &mut rbuf)?;

    // Message 2: <- e, ee, s, es
    let n2 = resp.write_message(&[], &mut buf)?;
    let msg2 = buf[..n2].to_vec();

    // Initiator reads msg2
    init.read_message(&msg2, &mut rbuf)?;

    // Print the handshake hash after reading msg2 — this is the AD that
    // the initiator's NEXT AEAD operation uses (for msg3's Enc(s)).
    // But more importantly, print it BEFORE reading msg2, to get the h
    // used for Enc(s) decryption.
    // We already consumed msg2, so let's re-build a fresh initiator to get
    // the intermediate hash. Actually, let's just print it at each step.
    {
        let mut init2 = Builder::new(params.clone())
            .fixed_ephemeral_key_for_testing_only(&init_ephem_priv)
            .local_private_key(&init_static_priv)
            .build_initiator()?;

        let hh0 = init2.get_handshake_hash().to_vec();
        println!("DEBUG hh0 (before msg1): {}", to_hex(&hh0));

        let n = init2.write_message(&[], &mut buf)?;
        let hh1 = init2.get_handshake_hash().to_vec();
        println!(
            "DEBUG hh1 (after write msg1, includes payload): {}",
            to_hex(&hh1)
        );

        init2.read_message(&msg2, &mut rbuf)?;
        let hh2 = init2.get_handshake_hash().to_vec();
        println!("DEBUG hh2 (after read msg2): {}", to_hex(&hh2));
    }

    // Message 3: -> s, se
    let n3 = init.write_message(&[], &mut buf)?;
    let msg3 = buf[..n3].to_vec();

    // Responder reads msg3
    resp.read_message(&msg3, &mut rbuf)?;

    // Extract public keys (after handshake, each side has the other's remote static).
    let resp_static_pub = init
        .get_remote_static()
        .map(|s| s.to_vec())
        .unwrap_or_default();
    let init_static_pub_vec = resp
        .get_remote_static()
        .map(|s| s.to_vec())
        .unwrap_or_default();

    // Derive the ephemeral public keys from msg1/msg2 (first 32 bytes).
    let init_ephem_pub = msg1[..32].to_vec();
    let resp_ephem_pub = msg2[..32].to_vec();

    // Transport mode.
    let mut init_t = init.into_transport_mode()?;
    let mut resp_t = resp.into_transport_mode()?;

    // Test payload I→R.
    let pt_i2r = b"Hello from iOS!";
    let enc_n = init_t.write_message(pt_i2r, &mut buf)?;
    let transport_i2r = buf[..enc_n].to_vec();

    // Verify decrypt.
    let dec_n = resp_t.read_message(&transport_i2r, &mut rbuf)?;
    assert_eq!(&rbuf[..dec_n], pt_i2r);

    // Test payload R→I.
    let pt_r2i = b"Hello from server!";
    let enc_n2 = resp_t.write_message(pt_r2i, &mut buf)?;
    let transport_r2i = buf[..enc_n2].to_vec();

    // Print test vector.
    println!("===NOISE_TEST_VECTOR_BEGIN===");
    println!("init_static_priv:{}", to_hex(&init_static_priv));
    println!("init_static_pub:{}", to_hex(&init_static_pub_vec));
    println!("init_ephem_priv:{}", to_hex(&init_ephem_priv));
    println!("init_ephem_pub:{}", to_hex(&init_ephem_pub));
    println!("resp_static_priv:{}", to_hex(&resp_static_priv));
    println!("resp_static_pub:{}", to_hex(&resp_static_pub));
    println!("resp_ephem_priv:{}", to_hex(&resp_ephem_priv));
    println!("resp_ephem_pub:{}", to_hex(&resp_ephem_pub));
    println!("msg1:{}", to_hex(&msg1));
    println!("msg2:{}", to_hex(&msg2));
    println!("msg3:{}", to_hex(&msg3));
    println!("transport_pt_i2r:{}", to_hex(pt_i2r));
    println!("transport_i2r:{}", to_hex(&transport_i2r));
    println!("transport_pt_r2i:{}", to_hex(pt_r2i));
    println!("transport_r2i:{}", to_hex(&transport_r2i));
    println!("===NOISE_TEST_VECTOR_END===");

    // === The initiator already successfully read msg2 (it's a full snow handshake). ===
    // The initiator's snow internal state has the CORRECT h and cipher key.
    // Since snow's HandshakeState fields are private, we can't extract them directly.
    // But the full handshake SUCCEEDED (no error), which means snow internally
    // computed the correct keys. The iOS implementation must match.

    Ok(())
}
