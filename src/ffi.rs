use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
#[allow(unused)]
pub struct OcrResult {
    pub(crate) result: *mut c_char,
    pub(crate) error: *mut c_char,
}

#[repr(C)]
#[allow(unused)]
pub struct OcrContext {
    params: *mut c_void,
    model: *mut c_void,
    ctx: *mut c_void,
}


extern "C" {
    pub fn ocr_init(argc: c_int, argv: *mut *mut c_char) -> *mut c_void;
    pub fn ocr_free(ctx: *mut c_void) -> c_int;
    pub fn ocr_run(ctx: *mut c_void, image_embeds: *const f32, n_embeds: c_int, got_type: c_int) -> *mut OcrResult;
    pub fn ocr_cleanup_ctx(ctx: *mut c_void) -> c_int;
    pub fn ocr_free_result(result: *mut OcrResult) -> c_int;
}