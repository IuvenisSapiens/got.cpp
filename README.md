# GOT.CPP

使用llama.cpp和onnxruntime 加速推理 [GOT OCR 2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
### Encoder
LLama.cpp不支持GOT 魔改的 Vision Encoder，水平有限懒得改clip.cpp的计算图，
正好最近gguf那边因为精度bug问题停了clip的GPU支持，反正encoder的IO很简单，所以不如拿onnx+dml简单快速。

提取Encoder的代码在 [main.ipynb](main.ipynb) 里，这一步只提取pt文件。onnx是通过 [MS Olive](https://github.com/microsoft/Olive)
转换的（官方说dml对静态形状支持的更好）

### Decoder [huggingface](https://huggingface.co/MosRat/got_decoder-Q4_K_M-GGUF)
Decoder是Qwen0.5B，拿huggingface的QwenForCasualLM直接导入GOT权重就行，头痛的是他tokenizer用的是tiktoken，需要自己
手动改成huggingface的。代码在 [main.ipynb](main.ipynb) 里。拿到model和tokenizer后直接上传huggingface仓库白嫖在线转换。

decoder的推理在多模态llava.cpp上做个改动，编译了一个库[libocr](cpp)。CUDA运行时太大了还分版本，所以Vulkan启动！

### cli
Rust cli [src](src)

### python
python [got_ocr.py](got_ocr.py) 通过 ctypes调用 libocr 来使用gguf decoder。
