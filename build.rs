use std::env;
use std::path::PathBuf;
fn main() {
    let current_dir = env::current_dir().expect("Failed to get current directory");

    let cpp_lib_path = current_dir.join("cpp/build-x64-windows-vulkan-release/lib/static");

    println!("cargo:rerun-if-changed={}", cpp_lib_path.display());

    println!("cargo:rustc-cfg=feature=\"crt-static\"");

    println!("cargo:rustc-link-search=native={}", cpp_lib_path.display());

    let vulkan_lib_path = env::var("VULKAN_SDK").expect("VULKAN_SDK environment variable not set");
    let vulkan_lib_path = PathBuf::from(vulkan_lib_path).join("Lib");

    println!(
        "cargo:rustc-link-search=native={}",
        vulkan_lib_path.display()
    );

    println!("cargo:rustc-link-lib=static=libocr");
    println!("cargo:rustc-link-lib=static=common");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    println!("cargo:rustc-link-lib=static=llama");

    #[cfg(feature = "vulkan")]
    {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
        println!("cargo:rustc-link-lib=static=vulkan-1");
    }

    #[cfg(feature = "directml")]
    {
        println!("cargo:rustc-link-lib=dylib=DirectML");
        println!("cargo:rustc-link-lib=dylib=d3d12");
        println!("cargo:rustc-link-lib=dylib=dxgi");
        println!("cargo:rustc-link-lib=dylib=dxcore");
    }
}
