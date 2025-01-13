mod args;
mod ffi;
mod got_encoder;
mod got_image_processor;
mod vit_image_processor;
use std::fmt::{self, Formatter};

pub use crate::args::{parse_args, Cli, Mode, OcrType};
use crate::got_encoder::GoTEncoder;
use crate::got_image_processor::{read_from_file, read_from_file_crop};
use anyhow::{anyhow, Result};
use ndarray::{concatenate, Axis};
use std::ffi::{c_int, CStr, CString};
use std::path::Path;

#[derive(Debug)]
pub enum GotType {
    Ocr = 1,
    Format = 2,
    CropOcr = 3,
    CropFormat = 4,
}

impl From<OcrType> for GotType {
    fn from(ocr_type: OcrType) -> Self {
        match ocr_type {
            OcrType::Ocr => GotType::Ocr,
            OcrType::Format => GotType::Format,
            OcrType::CropOcr => GotType::CropOcr,
            OcrType::CropFormat => GotType::CropFormat,
        }
    }
}

pub struct GotOcr {
    ctx: *mut std::ffi::c_void,
    encoder: GoTEncoder,
}

impl GotOcr {
    pub fn from_encoder_decoder(
        encoder_path: impl AsRef<Path>,
        decoder_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let decoder_path = decoder_path.as_ref();
        if !decoder_path.exists() {
            return Err(anyhow!("{} not a valid model!", decoder_path.display()));
        }
        let args = vec![
            "got".to_string(),
            "-m".to_string(),
            decoder_path.display().to_string(),
            // "-p".to_string(),
            // "PROMPT".to_string(),
            "-ngl".to_string(),
            "100".to_string(),
            "--log-verbosity".to_string(),
            "-1".to_string(),
        ];
        Ok(Self::new(args, encoder_path)?)
    }

    pub fn new(args: Vec<String>, encoder_path: impl AsRef<Path>) -> Result<Self> {
        let argc = args.len() as i32;
        let argv: Vec<*mut i8> = args
            .into_iter()
            .map(|arg| CString::new(arg).unwrap().into_raw())
            .collect();
        let argv_ptr = argv.as_ptr() as *mut *mut i8;
        let ctx = unsafe { ffi::ocr_init(argc, argv_ptr) };
        if ctx.is_null() {
            Err(anyhow!("Failed to initialize OCR context"))
        } else {
            Ok(GotOcr {
                ctx,
                encoder: GoTEncoder::from_file(encoder_path)?,
            })
        }
    }

    pub fn run_decode(
        &self,
        image_embeds: &[f32],
        n_embeds: i32,
        got_type: GotType,
    ) -> OcrResultWrapper {
        assert_eq!(image_embeds.len() as i32, n_embeds * 1024 * 256);
        let result = unsafe {
            ffi::ocr_run(
                self.ctx,
                image_embeds.as_ptr(),
                256 * n_embeds as c_int,
                got_type as c_int,
            )
        };
        result.into()
    }

    pub fn generate(
        &self,
        image_path: impl AsRef<Path>,
        got_type: GotType,
    ) -> Result<OcrResultWrapper> {
        match got_type {
            GotType::Ocr | GotType::Format => {
                let img = read_from_file(image_path)?;
                let img_embed = self.encoder.call(img)?;
                let img_embed_slice: &'static [f32] =
                    unsafe { std::slice::from_raw_parts(img_embed.as_ptr(), 256 * 1024) };
                Ok(self.run_decode(img_embed_slice, 1, got_type))
            }
            GotType::CropOcr | GotType::CropFormat => {
                let imgs = read_from_file_crop(image_path)?;
                let img_len = imgs.len();
                let mut img_embeds = Vec::with_capacity(img_len);
                for img in imgs {
                    img_embeds.push(self.encoder.call(img)?)
                }
                let img_embeds = concatenate(
                    Axis(0),
                    &(img_embeds.iter().map(|v| v.view()).collect::<Vec<_>>()),
                )?;
                let img_embed_slice: &'static [f32] = unsafe {
                    std::slice::from_raw_parts(img_embeds.as_ptr(), img_len * 256 * 1024)
                };
                Ok(self.run_decode(img_embed_slice, img_len as i32, got_type))
            }
        }
    }

    pub fn cleanup_ctx(&self) -> Result<()> {
        if unsafe { ffi::ocr_cleanup_ctx(self.ctx) } != 0 {
            Err(anyhow!("fail to reset ctx"))
        } else {
            Ok(())
        }
    }
}

impl Drop for GotOcr {
    fn drop(&mut self) {
        unsafe {
            ffi::ocr_free(self.ctx);
        }
    }
}

pub struct OcrResultWrapper {
    result: *mut ffi::OcrResult,
}

impl OcrResultWrapper {
    pub fn get_result(&self) -> Result<String> {
        unsafe {
            if (*self.result).error.is_null() {
                let c_str = CStr::from_ptr((*self.result).result);
                Ok(c_str.to_string_lossy().into_owned())
            } else {
                let c_str = CStr::from_ptr((*self.result).error);
                Err(anyhow!("OcrError:{}", c_str.to_string_lossy().into_owned()))
            }
        }
    }
}

impl std::fmt::Display for OcrResultWrapper {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.get_result() {
            Ok(result) => write!(f, "{}", result),
            Err(e) => write!(f, "Error: {}", e),
        }
    }
}

impl From<*mut ffi::OcrResult> for OcrResultWrapper {
    fn from(value: *mut ffi::OcrResult) -> Self {
        OcrResultWrapper { result: value }
    }
}

impl Drop for OcrResultWrapper {
    fn drop(&mut self) {
        unsafe {
            ffi::ocr_free_result(self.result);
        }
    }
}



// #[cfg(test)]
// mod tests {
//     use super::*;
// }
