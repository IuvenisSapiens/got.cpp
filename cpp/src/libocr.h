//
// Created by whl on 2025/1/9.
//

#ifndef LIBOCR_H
#define LIBOCR_H
#include "arg.h"
#include "base64.hpp"
#include "clip.h"
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "llava.h"
#include "log.h"
#include "sampling.h"
#ifdef GGML_USE_CUDA
#    include "ggml-cuda.h"
#endif
#ifdef NDEBUG
#    include "ggml-alloc.h"
#    include "ggml-backend.h"
#endif

// #define OCR_SHARED
// #define LLAMA_BUILD
#ifdef OCR_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define OCR_API __declspec(dllexport)
#        else
#            define OCR_API __declspec(dllimport)
#        endif
#    else
#        define OCR_API __attribute__((visibility("default")))
#    endif
#else
#    define OCR_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

using GOT_TYPE = int;
#define GOT_OCR_TYPE         1
#define GOT_FORMAT_TYPE      2
#define GOT_CROP_OCR_TYPE    3
#define GOT_CROP_FORMAT_TYPE 4

struct OCR_API ocr_result {
    char * result;
    char * error;
};

struct ocr_context {
    common_params * params;
    llama_model *   model;
    llama_context * ctx;
};

OCR_API void *       ocr_init(int argc, char ** argv);
OCR_API int          ocr_free(void * ctx);
OCR_API ocr_result * ocr_run(void * ctx, const float * image_embeds, int n_embeds, GOT_TYPE got_type);
OCR_API int          ocr_cleanup_ctx(void * ctx);
OCR_API int          ocr_free_result(ocr_result * result);
#ifdef __cplusplus
}
#endif
#endif  //LIBOCR_H
