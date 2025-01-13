use std::env;
use std::path::PathBuf;
fn main() {
    // 获取当前工作目录
    let current_dir = env::current_dir().expect("Failed to get current directory");

    // 构建相对路径
    let cpp_lib_path = current_dir.join("cpp/build-x64-windows-vulkan-release/lib");

    // 设置重新构建的条件
    println!("cargo:rerun-if-changed={}", cpp_lib_path.display());

    // 设置 MT
    println!("cargo:rustc-cfg=feature=\"crt-static\"");

    // 添加链接搜索路径
    println!("cargo:rustc-link-search=native={}", cpp_lib_path.display());

    // 使用环境变量获取 Vulkan 库路径
    let vulkan_lib_path = env::var("VULKAN_SDK").expect("VULKAN_SDK environment variable not set");
    let vulkan_lib_path = PathBuf::from(vulkan_lib_path).join("Lib");

    println!("cargo:rustc-link-search=native={}", vulkan_lib_path.display());

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