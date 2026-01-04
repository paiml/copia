//! Wire protocol for network transmission of copia operations.
//!
//! This module implements the COPIA wire protocol for transmitting
//! signatures, deltas, and control messages over network connections.

use std::io::{Read, Write};

use serde::{Deserialize, Serialize};

use crate::delta::Delta;
use crate::error::{CopiaError, Result};
use crate::signature::Signature;

/// Protocol magic bytes: "COPA"
pub const PROTOCOL_MAGIC: [u8; 4] = *b"COPA";

/// Current protocol version.
pub const PROTOCOL_VERSION: u8 = 1;

/// Maximum payload size (16 MB).
pub const MAX_PAYLOAD_SIZE: u32 = 16 * 1024 * 1024;

/// Protocol message types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum MessageType {
    /// Signature request.
    SignatureRequest = 0x01,
    /// Signature response.
    SignatureResponse = 0x02,
    /// Delta data.
    DeltaData = 0x03,
    /// Acknowledgment.
    Ack = 0x04,
    /// Error message.
    Error = 0x05,
    /// Heartbeat ping.
    Ping = 0x06,
    /// Heartbeat pong.
    Pong = 0x07,
}

impl MessageType {
    /// Convert from u8.
    ///
    /// # Errors
    ///
    /// Returns `ProtocolError` if the value is invalid.
    pub fn from_u8(value: u8) -> Result<Self> {
        match value {
            0x01 => Ok(Self::SignatureRequest),
            0x02 => Ok(Self::SignatureResponse),
            0x03 => Ok(Self::DeltaData),
            0x04 => Ok(Self::Ack),
            0x05 => Ok(Self::Error),
            0x06 => Ok(Self::Ping),
            0x07 => Ok(Self::Pong),
            _ => Err(CopiaError::ProtocolError(format!(
                "Invalid message type: {value:#x}"
            ))),
        }
    }
}

/// Protocol frame header.
///
/// ```text
/// ┌─────────┬─────────┬─────────┬─────────┐
/// │  MAGIC  │ LENGTH  │  TYPE   │ VERSION │
/// │ 4 bytes │ 4 bytes │ 1 byte  │ 1 byte  │
/// └─────────┴─────────┴─────────┴─────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    /// Magic bytes: "COPA".
    pub magic: [u8; 4],
    /// Payload length (little-endian).
    pub length: u32,
    /// Message type.
    pub msg_type: MessageType,
    /// Protocol version.
    pub version: u8,
    /// Reserved flags.
    pub flags: u16,
}

impl FrameHeader {
    /// Header size in bytes.
    pub const SIZE: usize = 12;

    /// Create a new frame header.
    #[must_use]
    pub const fn new(msg_type: MessageType, payload_len: u32) -> Self {
        Self {
            magic: PROTOCOL_MAGIC,
            length: payload_len,
            msg_type,
            version: PROTOCOL_VERSION,
            flags: 0,
        }
    }

    /// Validate the header.
    ///
    /// # Errors
    ///
    /// Returns `ProtocolError` if validation fails.
    pub fn validate(&self) -> Result<()> {
        if self.magic != PROTOCOL_MAGIC {
            return Err(CopiaError::ProtocolError(format!(
                "Invalid magic: expected {:?}, got {:?}",
                PROTOCOL_MAGIC, self.magic
            )));
        }
        if self.version != PROTOCOL_VERSION {
            return Err(CopiaError::ProtocolError(format!(
                "Unsupported version: expected {PROTOCOL_VERSION}, got {}",
                self.version
            )));
        }
        if self.length > MAX_PAYLOAD_SIZE {
            return Err(CopiaError::ProtocolError(format!(
                "Payload too large: {} > {MAX_PAYLOAD_SIZE}",
                self.length
            )));
        }
        Ok(())
    }

    /// Encode header to bytes.
    #[must_use]
    pub fn encode(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..8].copy_from_slice(&self.length.to_le_bytes());
        buf[8] = self.msg_type as u8;
        buf[9] = self.version;
        buf[10..12].copy_from_slice(&self.flags.to_le_bytes());
        buf
    }

    /// Decode header from bytes.
    ///
    /// # Errors
    ///
    /// Returns `ProtocolError` if decoding fails.
    pub fn decode(buf: &[u8; Self::SIZE]) -> Result<Self> {
        let magic: [u8; 4] = buf[0..4].try_into().map_err(|_| {
            CopiaError::ProtocolError("Failed to decode magic".to_string())
        })?;

        let length = u32::from_le_bytes(buf[4..8].try_into().map_err(|_| {
            CopiaError::ProtocolError("Failed to decode length".to_string())
        })?);

        let msg_type = MessageType::from_u8(buf[8])?;
        let version = buf[9];

        let flags = u16::from_le_bytes(buf[10..12].try_into().map_err(|_| {
            CopiaError::ProtocolError("Failed to decode flags".to_string())
        })?);

        let header = Self {
            magic,
            length,
            msg_type,
            version,
            flags,
        };

        header.validate()?;
        Ok(header)
    }

    /// Read header from a reader.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or decoding fails.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let mut buf = [0u8; Self::SIZE];
        reader.read_exact(&mut buf)?;
        Self::decode(&buf)
    }

    /// Write header to a writer.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.encode())?;
        Ok(())
    }
}

/// Protocol messages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Message {
    /// Request signature generation.
    SignatureRequest {
        /// File identifier.
        file_id: u64,
        /// Block size for signature.
        block_size: u32,
    },
    /// Signature response.
    SignatureResponse {
        /// File identifier.
        file_id: u64,
        /// Generated signature.
        signature: Signature,
    },
    /// Delta transmission.
    DeltaData {
        /// File identifier.
        file_id: u64,
        /// Computed delta.
        delta: Delta,
    },
    /// Acknowledgment.
    Ack {
        /// File identifier.
        file_id: u64,
        /// Success status.
        success: bool,
        /// Optional message.
        message: Option<String>,
    },
    /// Error response.
    Error {
        /// Error code.
        code: u32,
        /// Error message.
        message: String,
    },
    /// Heartbeat ping.
    Ping {
        /// Sequence number.
        seq: u64,
    },
    /// Heartbeat pong.
    Pong {
        /// Sequence number (echoed from ping).
        seq: u64,
    },
}

impl Message {
    /// Get the message type.
    #[must_use]
    pub const fn msg_type(&self) -> MessageType {
        match self {
            Self::SignatureRequest { .. } => MessageType::SignatureRequest,
            Self::SignatureResponse { .. } => MessageType::SignatureResponse,
            Self::DeltaData { .. } => MessageType::DeltaData,
            Self::Ack { .. } => MessageType::Ack,
            Self::Error { .. } => MessageType::Error,
            Self::Ping { .. } => MessageType::Ping,
            Self::Pong { .. } => MessageType::Pong,
        }
    }

    /// Encode message to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn encode(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| {
            CopiaError::ProtocolError(format!("Failed to encode message: {e}"))
        })
    }

    /// Decode message from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    pub fn decode(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(|e| {
            CopiaError::ProtocolError(format!("Failed to decode message: {e}"))
        })
    }
}

/// Protocol codec for reading/writing framed messages.
#[derive(Debug, Default)]
pub struct Codec {
    /// Buffer for partial reads.
    read_buf: Vec<u8>,
}

impl Codec {
    /// Create a new codec.
    #[must_use]
    pub fn new() -> Self {
        Self {
            read_buf: Vec::with_capacity(4096),
        }
    }

    /// Write a message to a writer.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn write_message<W: Write>(&self, writer: &mut W, message: &Message) -> Result<()> {
        let payload = message.encode()?;
        let payload_len = u32::try_from(payload.len()).map_err(|_| {
            CopiaError::ProtocolError("Payload too large for u32".to_string())
        })?;

        if payload_len > MAX_PAYLOAD_SIZE {
            return Err(CopiaError::ProtocolError(format!(
                "Payload exceeds maximum size: {payload_len} > {MAX_PAYLOAD_SIZE}"
            )));
        }

        let header = FrameHeader::new(message.msg_type(), payload_len);
        header.write_to(writer)?;
        writer.write_all(&payload)?;
        Ok(())
    }

    /// Read a message from a reader.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or decoding fails.
    pub fn read_message<R: Read>(&mut self, reader: &mut R) -> Result<Message> {
        let header = FrameHeader::read_from(reader)?;

        self.read_buf.resize(header.length as usize, 0);
        reader.read_exact(&mut self.read_buf)?;

        Message::decode(&self.read_buf)
    }
}

/// Frame builder for constructing protocol frames.
#[derive(Debug)]
pub struct FrameBuilder {
    file_id: u64,
    block_size: u32,
}

impl FrameBuilder {
    /// Create a new frame builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            file_id: 0,
            block_size: 2048,
        }
    }

    /// Set the file ID.
    #[must_use]
    pub const fn file_id(mut self, id: u64) -> Self {
        self.file_id = id;
        self
    }

    /// Set the block size.
    #[must_use]
    pub const fn block_size(mut self, size: u32) -> Self {
        self.block_size = size;
        self
    }

    /// Build a signature request message.
    #[must_use]
    pub const fn signature_request(&self) -> Message {
        Message::SignatureRequest {
            file_id: self.file_id,
            block_size: self.block_size,
        }
    }

    /// Build a signature response message.
    #[must_use]
    pub fn signature_response(&self, signature: Signature) -> Message {
        Message::SignatureResponse {
            file_id: self.file_id,
            signature,
        }
    }

    /// Build a delta data message.
    #[must_use]
    pub fn delta_data(&self, delta: Delta) -> Message {
        Message::DeltaData {
            file_id: self.file_id,
            delta,
        }
    }

    /// Build an acknowledgment message.
    #[must_use]
    pub fn ack(&self, success: bool, message: Option<String>) -> Message {
        Message::Ack {
            file_id: self.file_id,
            success,
            message,
        }
    }

    /// Build an error message.
    #[must_use]
    pub fn error(code: u32, message: String) -> Message {
        Message::Error { code, message }
    }

    /// Build a ping message.
    #[must_use]
    pub const fn ping(seq: u64) -> Message {
        Message::Ping { seq }
    }

    /// Build a pong message.
    #[must_use]
    pub const fn pong(seq: u64) -> Message {
        Message::Pong { seq }
    }
}

impl Default for FrameBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ==========================================================================
    // MESSAGE TYPE TESTS
    // ==========================================================================

    #[test]
    fn message_type_from_u8_valid() {
        assert_eq!(MessageType::from_u8(0x01).unwrap(), MessageType::SignatureRequest);
        assert_eq!(MessageType::from_u8(0x02).unwrap(), MessageType::SignatureResponse);
        assert_eq!(MessageType::from_u8(0x03).unwrap(), MessageType::DeltaData);
        assert_eq!(MessageType::from_u8(0x04).unwrap(), MessageType::Ack);
        assert_eq!(MessageType::from_u8(0x05).unwrap(), MessageType::Error);
        assert_eq!(MessageType::from_u8(0x06).unwrap(), MessageType::Ping);
        assert_eq!(MessageType::from_u8(0x07).unwrap(), MessageType::Pong);
    }

    #[test]
    fn message_type_from_u8_invalid() {
        assert!(MessageType::from_u8(0x00).is_err());
        assert!(MessageType::from_u8(0x08).is_err());
        assert!(MessageType::from_u8(0xFF).is_err());
    }

    // ==========================================================================
    // FRAME HEADER TESTS
    // ==========================================================================

    #[test]
    fn frame_header_new() {
        let header = FrameHeader::new(MessageType::Ping, 100);
        assert_eq!(header.magic, PROTOCOL_MAGIC);
        assert_eq!(header.length, 100);
        assert_eq!(header.msg_type, MessageType::Ping);
        assert_eq!(header.version, PROTOCOL_VERSION);
        assert_eq!(header.flags, 0);
    }

    #[test]
    fn frame_header_encode_decode() {
        let header = FrameHeader::new(MessageType::DeltaData, 12345);
        let encoded = header.encode();
        assert_eq!(encoded.len(), FrameHeader::SIZE);

        let decoded = FrameHeader::decode(&encoded).unwrap();
        assert_eq!(header, decoded);
    }

    #[test]
    fn frame_header_validate_valid() {
        let header = FrameHeader::new(MessageType::Ack, 1000);
        assert!(header.validate().is_ok());
    }

    #[test]
    fn frame_header_validate_invalid_magic() {
        let mut header = FrameHeader::new(MessageType::Ack, 100);
        header.magic = *b"XXXX";
        assert!(header.validate().is_err());
    }

    #[test]
    fn frame_header_validate_invalid_version() {
        let mut header = FrameHeader::new(MessageType::Ack, 100);
        header.version = 99;
        assert!(header.validate().is_err());
    }

    #[test]
    fn frame_header_validate_payload_too_large() {
        let header = FrameHeader::new(MessageType::Ack, MAX_PAYLOAD_SIZE + 1);
        assert!(header.validate().is_err());
    }

    #[test]
    fn frame_header_read_write() {
        let header = FrameHeader::new(MessageType::SignatureRequest, 500);
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(buf);
        let read_header = FrameHeader::read_from(&mut cursor).unwrap();
        assert_eq!(header, read_header);
    }

    // ==========================================================================
    // MESSAGE TESTS
    // ==========================================================================

    #[test]
    fn message_signature_request() {
        let msg = Message::SignatureRequest {
            file_id: 42,
            block_size: 1024,
        };
        assert_eq!(msg.msg_type(), MessageType::SignatureRequest);
    }

    #[test]
    fn message_signature_response() {
        let sig = Signature::new(1024, 0);
        let msg = Message::SignatureResponse {
            file_id: 1,
            signature: sig,
        };
        assert_eq!(msg.msg_type(), MessageType::SignatureResponse);
    }

    #[test]
    fn message_delta_data() {
        let delta = Delta::new(1024, 1000, 1000);
        let msg = Message::DeltaData {
            file_id: 1,
            delta,
        };
        assert_eq!(msg.msg_type(), MessageType::DeltaData);
    }

    #[test]
    fn message_ack() {
        let msg = Message::Ack {
            file_id: 1,
            success: true,
            message: Some("OK".to_string()),
        };
        assert_eq!(msg.msg_type(), MessageType::Ack);
    }

    #[test]
    fn message_error() {
        let msg = Message::Error {
            code: 404,
            message: "Not found".to_string(),
        };
        assert_eq!(msg.msg_type(), MessageType::Error);
    }

    #[test]
    fn message_ping_pong() {
        let ping = Message::Ping { seq: 123 };
        let pong = Message::Pong { seq: 123 };
        assert_eq!(ping.msg_type(), MessageType::Ping);
        assert_eq!(pong.msg_type(), MessageType::Pong);
    }

    #[test]
    fn message_encode_decode_signature_request() {
        let msg = Message::SignatureRequest {
            file_id: 42,
            block_size: 2048,
        };
        let encoded = msg.encode().unwrap();
        let decoded = Message::decode(&encoded).unwrap();
        assert_eq!(msg, decoded);
    }

    #[test]
    fn message_encode_decode_ack() {
        let msg = Message::Ack {
            file_id: 99,
            success: true,
            message: Some("Completed".to_string()),
        };
        let encoded = msg.encode().unwrap();
        let decoded = Message::decode(&encoded).unwrap();
        assert_eq!(msg, decoded);
    }

    #[test]
    fn message_encode_decode_error() {
        let msg = Message::Error {
            code: 500,
            message: "Internal error".to_string(),
        };
        let encoded = msg.encode().unwrap();
        let decoded = Message::decode(&encoded).unwrap();
        assert_eq!(msg, decoded);
    }

    #[test]
    fn message_encode_decode_delta() {
        let mut delta = Delta::new(1024, 500, 500);
        delta.push_copy(0, 200);
        delta.push_literal(b"new data");

        let msg = Message::DeltaData {
            file_id: 1,
            delta,
        };
        let encoded = msg.encode().unwrap();
        let decoded = Message::decode(&encoded).unwrap();
        assert_eq!(msg, decoded);
    }

    // ==========================================================================
    // CODEC TESTS
    // ==========================================================================

    #[test]
    fn codec_new() {
        let codec = Codec::new();
        assert!(codec.read_buf.is_empty());
    }

    #[test]
    fn codec_write_read_message() {
        let codec = Codec::new();
        let msg = Message::Ping { seq: 12345 };

        let mut buf = Vec::new();
        codec.write_message(&mut buf, &msg).unwrap();

        let mut codec2 = Codec::new();
        let mut cursor = Cursor::new(buf);
        let read_msg = codec2.read_message(&mut cursor).unwrap();

        assert_eq!(msg, read_msg);
    }

    #[test]
    fn codec_write_read_signature_request() {
        let codec = Codec::new();
        let msg = Message::SignatureRequest {
            file_id: 42,
            block_size: 4096,
        };

        let mut buf = Vec::new();
        codec.write_message(&mut buf, &msg).unwrap();

        let mut codec2 = Codec::new();
        let mut cursor = Cursor::new(buf);
        let read_msg = codec2.read_message(&mut cursor).unwrap();

        assert_eq!(msg, read_msg);
    }

    #[test]
    fn codec_write_read_complex_delta() {
        let codec = Codec::new();

        let mut delta = Delta::new(1024, 2000, 1500);
        delta.push_literal(b"header data");
        delta.push_copy(0, 500);
        delta.push_literal(b"middle section");
        delta.push_copy(800, 300);
        delta.push_literal(b"footer");

        let msg = Message::DeltaData {
            file_id: 999,
            delta,
        };

        let mut buf = Vec::new();
        codec.write_message(&mut buf, &msg).unwrap();

        let mut codec2 = Codec::new();
        let mut cursor = Cursor::new(buf);
        let read_msg = codec2.read_message(&mut cursor).unwrap();

        assert_eq!(msg, read_msg);
    }

    #[test]
    fn codec_multiple_messages() {
        let codec = Codec::new();

        let messages = vec![
            Message::Ping { seq: 1 },
            Message::Pong { seq: 1 },
            Message::SignatureRequest {
                file_id: 10,
                block_size: 2048,
            },
            Message::Ack {
                file_id: 10,
                success: true,
                message: None,
            },
        ];

        let mut buf = Vec::new();
        for msg in &messages {
            codec.write_message(&mut buf, msg).unwrap();
        }

        let mut codec2 = Codec::new();
        let mut cursor = Cursor::new(buf);

        for expected in &messages {
            let read_msg = codec2.read_message(&mut cursor).unwrap();
            assert_eq!(expected, &read_msg);
        }
    }

    // ==========================================================================
    // FRAME BUILDER TESTS
    // ==========================================================================

    #[test]
    fn frame_builder_new() {
        let builder = FrameBuilder::new();
        let msg = builder.signature_request();

        if let Message::SignatureRequest { file_id, block_size } = msg {
            assert_eq!(file_id, 0);
            assert_eq!(block_size, 2048);
        } else {
            panic!("Expected SignatureRequest");
        }
    }

    #[test]
    fn frame_builder_with_file_id() {
        let builder = FrameBuilder::new().file_id(42);
        let msg = builder.signature_request();

        if let Message::SignatureRequest { file_id, .. } = msg {
            assert_eq!(file_id, 42);
        } else {
            panic!("Expected SignatureRequest");
        }
    }

    #[test]
    fn frame_builder_with_block_size() {
        let builder = FrameBuilder::new().block_size(4096);
        let msg = builder.signature_request();

        if let Message::SignatureRequest { block_size, .. } = msg {
            assert_eq!(block_size, 4096);
        } else {
            panic!("Expected SignatureRequest");
        }
    }

    #[test]
    fn frame_builder_signature_response() {
        let sig = Signature::new(1024, 1000);
        let builder = FrameBuilder::new().file_id(1);
        let msg = builder.signature_response(sig.clone());

        if let Message::SignatureResponse { file_id, signature } = msg {
            assert_eq!(file_id, 1);
            assert_eq!(signature.block_size, sig.block_size);
        } else {
            panic!("Expected SignatureResponse");
        }
    }

    #[test]
    fn frame_builder_delta_data() {
        let delta = Delta::new(1024, 500, 500);
        let builder = FrameBuilder::new().file_id(99);
        let msg = builder.delta_data(delta);

        if let Message::DeltaData { file_id, .. } = msg {
            assert_eq!(file_id, 99);
        } else {
            panic!("Expected DeltaData");
        }
    }

    #[test]
    fn frame_builder_ack() {
        let builder = FrameBuilder::new().file_id(5);
        let msg = builder.ack(true, Some("Done".to_string()));

        if let Message::Ack { file_id, success, message } = msg {
            assert_eq!(file_id, 5);
            assert!(success);
            assert_eq!(message, Some("Done".to_string()));
        } else {
            panic!("Expected Ack");
        }
    }

    #[test]
    fn frame_builder_error() {
        let msg = FrameBuilder::error(404, "Not found".to_string());

        if let Message::Error { code, message } = msg {
            assert_eq!(code, 404);
            assert_eq!(message, "Not found");
        } else {
            panic!("Expected Error");
        }
    }

    #[test]
    fn frame_builder_ping_pong() {
        let ping = FrameBuilder::ping(100);
        let pong = FrameBuilder::pong(100);

        if let Message::Ping { seq } = ping {
            assert_eq!(seq, 100);
        } else {
            panic!("Expected Ping");
        }

        if let Message::Pong { seq } = pong {
            assert_eq!(seq, 100);
        } else {
            panic!("Expected Pong");
        }
    }

    #[test]
    fn frame_builder_default() {
        let builder = FrameBuilder::default();
        let msg = builder.signature_request();
        assert_eq!(msg.msg_type(), MessageType::SignatureRequest);
    }

    // ==========================================================================
    // EDGE CASES
    // ==========================================================================

    #[test]
    fn empty_signature_response() {
        let codec = Codec::new();
        let sig = Signature::new(1024, 0);
        let msg = Message::SignatureResponse {
            file_id: 1,
            signature: sig,
        };

        let mut buf = Vec::new();
        codec.write_message(&mut buf, &msg).unwrap();

        let mut codec2 = Codec::new();
        let mut cursor = Cursor::new(buf);
        let read_msg = codec2.read_message(&mut cursor).unwrap();

        assert_eq!(msg, read_msg);
    }

    #[test]
    fn ack_with_no_message() {
        let msg = Message::Ack {
            file_id: 1,
            success: false,
            message: None,
        };

        let encoded = msg.encode().unwrap();
        let decoded = Message::decode(&encoded).unwrap();
        assert_eq!(msg, decoded);
    }

    #[test]
    fn error_with_empty_message() {
        let msg = Message::Error {
            code: 0,
            message: String::new(),
        };

        let encoded = msg.encode().unwrap();
        let decoded = Message::decode(&encoded).unwrap();
        assert_eq!(msg, decoded);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use std::io::Cursor;

    proptest! {
        /// Frame header encode/decode roundtrip
        #[test]
        fn frame_header_roundtrip(
            msg_type in 1u8..=7,
            length in 0u32..MAX_PAYLOAD_SIZE
        ) {
            let msg_type = MessageType::from_u8(msg_type).unwrap();
            let header = FrameHeader::new(msg_type, length);
            let encoded = header.encode();
            let decoded = FrameHeader::decode(&encoded).unwrap();
            prop_assert_eq!(header, decoded);
        }

        /// Signature request roundtrip
        #[test]
        fn signature_request_roundtrip(
            file_id in any::<u64>(),
            block_size in 512u32..65536
        ) {
            let msg = Message::SignatureRequest { file_id, block_size };
            let encoded = msg.encode().unwrap();
            let decoded = Message::decode(&encoded).unwrap();
            prop_assert_eq!(msg, decoded);
        }

        /// Ack message roundtrip
        #[test]
        fn ack_roundtrip(
            file_id in any::<u64>(),
            success in any::<bool>(),
            message in proptest::option::of(any::<String>())
        ) {
            let msg = Message::Ack { file_id, success, message };
            let encoded = msg.encode().unwrap();
            let decoded = Message::decode(&encoded).unwrap();
            prop_assert_eq!(msg, decoded);
        }

        /// Error message roundtrip
        #[test]
        fn error_roundtrip(
            code in any::<u32>(),
            message in any::<String>()
        ) {
            let msg = Message::Error { code, message };
            let encoded = msg.encode().unwrap();
            let decoded = Message::decode(&encoded).unwrap();
            prop_assert_eq!(msg, decoded);
        }

        /// Ping/pong roundtrip
        #[test]
        fn ping_pong_roundtrip(seq in any::<u64>()) {
            let ping = Message::Ping { seq };
            let pong = Message::Pong { seq };

            let ping_encoded = ping.encode().unwrap();
            let pong_encoded = pong.encode().unwrap();

            let ping_decoded = Message::decode(&ping_encoded).unwrap();
            let pong_decoded = Message::decode(&pong_encoded).unwrap();

            prop_assert_eq!(ping, ping_decoded);
            prop_assert_eq!(pong, pong_decoded);
        }

        /// Codec roundtrip
        #[test]
        fn codec_roundtrip(
            file_id in any::<u64>(),
            block_size in 512u32..65536
        ) {
            let codec = Codec::new();
            let msg = Message::SignatureRequest { file_id, block_size };

            let mut buf = Vec::new();
            codec.write_message(&mut buf, &msg).unwrap();

            let mut codec2 = Codec::new();
            let mut cursor = Cursor::new(buf);
            let read_msg = codec2.read_message(&mut cursor).unwrap();

            prop_assert_eq!(msg, read_msg);
        }
    }
}
