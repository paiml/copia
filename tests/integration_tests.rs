//! Integration tests for copia.

use std::io::Cursor;

use copia::{
    Codec, CopiaSync, FrameBuilder, FrameHeader, Message, MessageType, StrongHash, Sync,
    SyncBuilder, PROTOCOL_MAGIC, PROTOCOL_VERSION,
};

// =============================================================================
// END-TO-END SYNC TESTS
// =============================================================================

#[test]
fn sync_identical_files() {
    let sync = SyncBuilder::new().block_size(512).build();

    // Data must be larger than block size for good compression ratio
    let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();

    // Generate signature
    let sig = sync.signature(Cursor::new(&data)).unwrap();

    // Compute delta (should be all copies)
    let delta = sync.delta(Cursor::new(&data), &sig).unwrap();

    // Verify high match ratio for block-aligned identical data
    assert!(
        delta.compression_ratio(data.len() as u64) > 0.9,
        "Expected high ratio, got {}",
        delta.compression_ratio(data.len() as u64)
    );

    // Apply patch
    let mut output = Vec::new();
    sync.patch(Cursor::new(&data), &delta, &mut output).unwrap();

    assert_eq!(output, data);
}

#[test]
fn sync_modified_file() {
    let sync = SyncBuilder::new()
        .block_size(512)
        .verify_checksum(true)
        .build();

    let basis = b"Hello, World! This is a test file with some content.";
    let source = b"Hello, Universe! This is a test file with some content.";

    // Generate signature from basis
    let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();

    // Compute delta from modified source
    let delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

    // Apply patch to reconstruct source
    let mut output = Vec::new();
    sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)
        .unwrap();

    assert_eq!(output, source);
}

#[test]
fn sync_appended_content() {
    let sync = CopiaSync::with_block_size(512);

    let basis = b"Original content";
    let source = b"Original content with appended data at the end";

    let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
    let delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

    let mut output = Vec::new();
    sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)
        .unwrap();

    assert_eq!(output, source);
}

#[test]
fn sync_prepended_content() {
    let sync = CopiaSync::with_block_size(512);

    let basis = b"Original content here";
    let source = b"Prepended data followed by Original content here";

    let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
    let delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

    let mut output = Vec::new();
    sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)
        .unwrap();

    assert_eq!(output, source);
}

#[test]
fn sync_large_file() {
    let sync = SyncBuilder::new()
        .block_size(1024)
        .verify_checksum(true)
        .build();

    // Create a large basis file (100KB)
    let basis: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

    // Create source with 5% modification
    let mut source = basis.clone();
    for i in (0..source.len()).step_by(20) {
        source[i] = 0xFF;
    }

    let sig = sync.signature(Cursor::new(&basis)).unwrap();
    let delta = sync.delta(Cursor::new(&source), &sig).unwrap();

    let mut output = Vec::new();
    sync.patch(Cursor::new(&basis), &delta, &mut output).unwrap();

    assert_eq!(output, source);
}

#[test]
fn sync_empty_to_content() {
    let sync = CopiaSync::with_block_size(512);

    let basis: &[u8] = b"";
    let source = b"Brand new content created from nothing";

    let sig = sync.signature(Cursor::new(basis)).unwrap();
    let delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

    // All should be literal since basis is empty
    assert_eq!(delta.bytes_matched(), 0);
    assert_eq!(delta.bytes_literal(), source.len() as u64);

    let mut output = Vec::new();
    sync.patch(Cursor::new(basis), &delta, &mut output).unwrap();

    assert_eq!(output, source);
}

#[test]
fn sync_content_to_empty() {
    let sync = CopiaSync::with_block_size(512);

    let basis = b"Content that will be completely removed";
    let source: &[u8] = b"";

    let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
    let delta = sync.delta(Cursor::new(source), &sig).unwrap();

    let mut output = Vec::new();
    sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)
        .unwrap();

    assert!(output.is_empty());
}

#[test]
fn sync_binary_data() {
    let sync = CopiaSync::with_block_size(512);

    // Binary data with all byte values
    let basis: Vec<u8> = (0..=255).cycle().take(5000).collect();

    // Modify some bytes
    let mut source = basis.clone();
    source[100] = 0x00;
    source[500] = 0xFF;
    source[2000] = 0xAB;

    let sig = sync.signature(Cursor::new(&basis)).unwrap();
    let delta = sync.delta(Cursor::new(&source), &sig).unwrap();

    let mut output = Vec::new();
    sync.patch(Cursor::new(&basis), &delta, &mut output).unwrap();

    assert_eq!(output, source);
}

// =============================================================================
// PROTOCOL INTEGRATION TESTS
// =============================================================================

#[test]
fn protocol_signature_request_response_flow() {
    let codec = Codec::new();
    let sync = CopiaSync::with_block_size(1024);

    // Client sends signature request
    let request = FrameBuilder::new()
        .file_id(1)
        .block_size(1024)
        .signature_request();

    let mut buf = Vec::new();
    codec.write_message(&mut buf, &request).unwrap();

    // Server receives and processes
    let mut codec2 = Codec::new();
    let received = codec2.read_message(&mut Cursor::new(&buf)).unwrap();

    if let Message::SignatureRequest { file_id, block_size } = received {
        assert_eq!(file_id, 1);
        assert_eq!(block_size, 1024);

        // Server generates signature
        let data = b"file content to sign";
        let sig = sync.signature(Cursor::new(data.as_slice())).unwrap();

        // Server sends response
        let response = FrameBuilder::new().file_id(1).signature_response(sig);
        let mut response_buf = Vec::new();
        codec.write_message(&mut response_buf, &response).unwrap();

        // Client receives response
        let mut codec3 = Codec::new();
        let response_msg = codec3
            .read_message(&mut Cursor::new(&response_buf))
            .unwrap();

        if let Message::SignatureResponse { file_id, signature } = response_msg {
            assert_eq!(file_id, 1);
            assert!(!signature.is_empty());
        } else {
            panic!("Expected SignatureResponse");
        }
    } else {
        panic!("Expected SignatureRequest");
    }
}

#[test]
fn protocol_full_sync_flow() {
    let codec = Codec::new();
    let sync = CopiaSync::with_block_size(512);

    let basis = b"Original file content on receiver side";
    let source = b"Modified file content on sender side";

    // 1. Receiver generates and sends signature
    let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
    let sig_msg = FrameBuilder::new().file_id(1).signature_response(sig);

    let mut sig_buf = Vec::new();
    codec.write_message(&mut sig_buf, &sig_msg).unwrap();

    // 2. Sender receives signature, computes delta
    let mut codec2 = Codec::new();
    let received_sig = codec2.read_message(&mut Cursor::new(&sig_buf)).unwrap();

    let delta = if let Message::SignatureResponse { signature, .. } = received_sig {
        sync.delta(Cursor::new(source.as_slice()), &signature)
            .unwrap()
    } else {
        panic!("Expected SignatureResponse");
    };

    // 3. Sender sends delta
    let delta_msg = FrameBuilder::new().file_id(1).delta_data(delta);
    let mut delta_buf = Vec::new();
    codec.write_message(&mut delta_buf, &delta_msg).unwrap();

    // 4. Receiver applies delta
    let mut codec3 = Codec::new();
    let received_delta = codec3.read_message(&mut Cursor::new(&delta_buf)).unwrap();

    let mut output = Vec::new();
    if let Message::DeltaData { delta, .. } = received_delta {
        sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)
            .unwrap();
    } else {
        panic!("Expected DeltaData");
    }

    // 5. Verify reconstruction
    assert_eq!(output, source);

    // 6. Receiver sends ack
    let ack = FrameBuilder::new()
        .file_id(1)
        .ack(true, Some("Sync complete".to_string()));
    let mut ack_buf = Vec::new();
    codec.write_message(&mut ack_buf, &ack).unwrap();

    let mut codec4 = Codec::new();
    let received_ack = codec4.read_message(&mut Cursor::new(&ack_buf)).unwrap();

    if let Message::Ack { success, message, .. } = received_ack {
        assert!(success);
        assert_eq!(message, Some("Sync complete".to_string()));
    } else {
        panic!("Expected Ack");
    }
}

#[test]
fn protocol_ping_pong() {
    let codec = Codec::new();

    // Send ping
    let ping = FrameBuilder::ping(42);
    let mut buf = Vec::new();
    codec.write_message(&mut buf, &ping).unwrap();

    // Receive and respond
    let mut codec2 = Codec::new();
    let received = codec2.read_message(&mut Cursor::new(&buf)).unwrap();

    if let Message::Ping { seq } = received {
        assert_eq!(seq, 42);

        // Send pong
        let pong = FrameBuilder::pong(seq);
        let mut pong_buf = Vec::new();
        codec.write_message(&mut pong_buf, &pong).unwrap();

        let mut codec3 = Codec::new();
        let received_pong = codec3.read_message(&mut Cursor::new(&pong_buf)).unwrap();

        if let Message::Pong { seq } = received_pong {
            assert_eq!(seq, 42);
        } else {
            panic!("Expected Pong");
        }
    } else {
        panic!("Expected Ping");
    }
}

#[test]
fn protocol_error_handling() {
    let codec = Codec::new();

    let error = FrameBuilder::error(404, "File not found".to_string());
    let mut buf = Vec::new();
    codec.write_message(&mut buf, &error).unwrap();

    let mut codec2 = Codec::new();
    let received = codec2.read_message(&mut Cursor::new(&buf)).unwrap();

    if let Message::Error { code, message } = received {
        assert_eq!(code, 404);
        assert_eq!(message, "File not found");
    } else {
        panic!("Expected Error");
    }
}

// =============================================================================
// MULTI-FILE SYNC TESTS
// =============================================================================

#[test]
fn sync_multiple_files() {
    let sync = CopiaSync::with_block_size(512);

    let files = vec![
        (b"File 1 content".to_vec(), b"File 1 modified".to_vec()),
        (b"File 2 original".to_vec(), b"File 2 updated version".to_vec()),
        (b"Third file here".to_vec(), b"Third file with changes".to_vec()),
    ];

    for (basis, source) in &files {
        let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
        let delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

        let mut output = Vec::new();
        sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)
            .unwrap();

        assert_eq!(&output, source);
    }
}

// =============================================================================
// CHECKSUM VERIFICATION TESTS
// =============================================================================

#[test]
fn checksum_detects_corruption() {
    let sync = SyncBuilder::new()
        .block_size(512)
        .verify_checksum(true)
        .build();

    let basis = b"Original content";
    let source = b"New content";

    let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
    let mut delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

    // Corrupt the checksum
    delta.checksum = StrongHash::zero();

    let mut output = Vec::new();
    let result = sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output);

    assert!(result.is_err());
}

#[test]
fn checksum_verification_disabled() {
    let sync = SyncBuilder::new()
        .block_size(512)
        .verify_checksum(false)
        .build();

    let basis = b"Original content";
    let source = b"New content";

    let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
    let mut delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

    // Corrupt the checksum - should still work
    delta.checksum = StrongHash::zero();

    let mut output = Vec::new();
    sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)
        .unwrap();

    assert_eq!(output, source);
}

// =============================================================================
// BLOCK SIZE VARIATION TESTS
// =============================================================================

#[test]
fn various_block_sizes() {
    let block_sizes = [512, 1024, 2048, 4096, 8192];

    let basis: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
    let mut source = basis.clone();
    source[5000] = 0xFF;

    for block_size in block_sizes {
        let sync = CopiaSync::with_block_size(block_size);

        let sig = sync.signature(Cursor::new(&basis)).unwrap();
        let delta = sync.delta(Cursor::new(&source), &sig).unwrap();

        let mut output = Vec::new();
        sync.patch(Cursor::new(&basis), &delta, &mut output).unwrap();

        assert_eq!(output, source, "Failed for block_size={block_size}");
    }
}

// =============================================================================
// PROTOCOL CONSTANTS
// =============================================================================

#[test]
fn protocol_constants() {
    assert_eq!(PROTOCOL_MAGIC, *b"COPA");
    assert_eq!(PROTOCOL_VERSION, 1);
    assert_eq!(FrameHeader::SIZE, 12);
}

#[test]
fn message_types_exhaustive() {
    let types = [
        MessageType::SignatureRequest,
        MessageType::SignatureResponse,
        MessageType::DeltaData,
        MessageType::Ack,
        MessageType::Error,
        MessageType::Ping,
        MessageType::Pong,
    ];

    for (i, msg_type) in types.iter().enumerate() {
        let from_u8 = MessageType::from_u8((i + 1) as u8).unwrap();
        assert_eq!(*msg_type, from_u8);
    }
}
