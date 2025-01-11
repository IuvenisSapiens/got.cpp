fn main() {
    println!("cargo:rerun-if-changed=E:\\WorkSpace\\CppProjects\\llama.cpp\\build-x64-windows-vulkan-release\\lib");
    // 设置 MT
    println!("cargo:rustc-cfg=feature=\"crt-static\"");

    // 添加链接搜索路径
    println!("cargo:rustc-link-search=native=E:\\WorkSpace\\CppProjects\\llama.cpp\\build-x64-windows-vulkan-release\\lib");
    println!("cargo:rustc-link-search=native=D:\\Scoop\\apps\\vulkan\\current\\Lib");

    // 链接静态库 (llama.cpp vulkan 后端)
    println!("cargo:rustc-link-lib=static=libocr");
    println!("cargo:rustc-link-lib=static=common");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-vulkan");
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=vulkan-1");

    // 链接动态库
    println!("cargo:rustc-link-lib=dylib=DirectML");
    println!("cargo:rustc-link-lib=dylib=d3d12");
    println!("cargo:rustc-link-lib=dylib=dxgi");
    println!("cargo:rustc-link-lib=dylib=dxcore");
}