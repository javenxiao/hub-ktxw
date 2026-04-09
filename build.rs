// build.rs - 配置 FFI 绑定和 DLL 链接
use std::env;
use std::path::PathBuf;
use std::fs;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_path = manifest_dir.join("third_party").join("lib");
    let _include_path = manifest_dir.join("third_party").join("include");

    // ============ 自动复制 DLL 到输出目录 ============
    let profile = env::var("PROFILE").unwrap();
    let target_dir = manifest_dir.join("target").join(&profile);

    let dll_src = lib_path.join("ar8030_client.dll");
    let dll_dst = target_dir.join("ar8030_client.dll");

    if dll_src.exists() {
        let _ = fs::copy(&dll_src, &dll_dst);
        println!("cargo:rerun-if-changed={}", dll_src.display());
    }

    // ============ 生成 FFI 绑定 (可选) ============
    // 注释掉自动生成，改为手动维护 FFI 定义以获得更好的控制
    // 如果需要自动生成，请解除注释以下代码：
    /*
    let bindings = bindgen::Builder::default()
        .header(include_path.join("bb_api.h").to_string_lossy().to_string())
        .clang_arg(format!("-I{}", include_path.display()))
        .generate()
        .expect("bindgen failed");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
    */
}