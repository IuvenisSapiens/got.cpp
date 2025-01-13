use std::fmt;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Mode {
    /// Run once with the image input
    Once,
    /// Interactively input multiple images like chat
    Chat,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum OcrType {
    /// Corresponds to "chat" in the GOT OCR.
    Ocr,
    /// Corresponds to "chat" in the GOT OCR with type = "format".
    Format,
    /// Corresponds to "chat_crop" in the GOT OCR.
    CropOcr,
    /// Corresponds to "chat_crop" in the GOT OCR with type = "format".
    CropFormat,
}

impl fmt::Display for OcrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OcrType::Ocr => write!(f, "Ocr"),
            OcrType::Format => write!(f, "Format"),
            OcrType::CropOcr => write!(f, "CropOcr"),
            OcrType::CropFormat => write!(f, "CropFormat"),
        }
    }
}

fn path_encoder_parser(s: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(s);
    if path.exists() {
        Ok(path)
    }else {
        let path = PathBuf::from(r#".\encoder_single.onnx"#);
        if path.exists() {Ok(path)} else { Err("cant find decoder in path!".to_string()) }
    }
}

fn path_decoder_parser(s: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(s);
    if path.exists() {
        Ok(path)
    }else {
        let path = PathBuf::from(r#".\got_decoder-q4_k_m.gguf"#);
        if path.exists() {Ok(path)} else { Err("cant find decoder in path!".to_string()) }
    }
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// Path of encoder onnx , default = .\encoder_single.onnx
    #[arg(short,long,value_parser = path_encoder_parser , value_name = "FILE")]
    pub encoder: PathBuf,
    /// Path of decoder gguf , default = .\got_decoder-q4_k_m.gguf
    #[arg(short,long,value_parser = path_decoder_parser , value_name = "FILE")]
    pub decoder: PathBuf,
    /// Got ocr type
    #[arg(value_enum,short, long, default_value_t = OcrType::Ocr)]
    pub r#type: OcrType,
    /// Cli mode
    #[arg(value_enum,short, long, default_value_t = Mode::Chat)]
    pub mode: Mode,
    /// (for mode::once) image to ocr
    pub image_path:Option<PathBuf>
}

pub fn parse_args() -> Cli {
    Cli::parse()
}

#[test]
fn verify_cli() {
    use clap::CommandFactory;
    Cli::command().debug_assert();
}
