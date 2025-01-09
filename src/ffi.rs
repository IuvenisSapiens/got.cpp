use std::os::raw::{c_char, c_int, c_void};

#[cfg(feature = "vl")]
// 定义不透明的结构体指针
#[repr(C)]
pub struct LlavaInner(c_void);
#[cfg(feature = "vl")]
#[repr(C)]
pub struct VlResult(c_void);

#[cfg(feature = "vl")]
// 定义 FFI 函数
extern "C" {
    pub fn vl_init(argc: c_int, argv: *mut *mut c_char) -> *mut LlavaInner;
    pub fn vl_run(
        ctx: *mut LlavaInner,
        image_bytes: *const u8,
        image_bytes_length: c_int,
    ) -> *mut VlResult;
    pub fn vl_release(ctx: *mut LlavaInner) -> c_int;
    pub fn vl_release_result(result: *mut VlResult) -> c_int;
    pub fn vl_reload_context(ctx: *mut LlavaInner) -> c_int;
}

#[cfg(feature = "got")]
#[repr(C)]
pub struct OcrResult {
    pub(crate) result: *mut c_char,
    pub(crate) error: *mut c_char,
}
#[cfg(feature = "got")]
#[repr(C)]
pub struct OcrContext {
    params: *mut c_void,
    model: *mut c_void,
    ctx: *mut c_void,
}

#[cfg(feature = "got")]
extern "C" {
    pub fn ocr_init(argc: c_int, argv: *mut *mut c_char) -> *mut c_void;
    pub fn ocr_free(ctx: *mut c_void) -> c_int;
    pub fn ocr_run(ctx: *mut c_void, image_embeds: *const f32, n_embeds: c_int, got_type: c_int) -> *mut OcrResult;
    pub fn ocr_cleanup_ctx(ctx: *mut c_void) -> c_int;
    pub fn ocr_free_result(result: *mut OcrResult) -> c_int;
}