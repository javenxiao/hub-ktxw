// build.rs - 配置 FFI 绑定和 DLL 链接
use std::fs;
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_path = manifest_dir.join("third_party").join("lib");
    let _include_path = manifest_dir.join("third_party").join("include");

    // ============ 自动复制 DLL 到输出目录 ============
    let profile = env::var("PROFILE").unwrap();
    let target_dir = manifest_dir.join("target").join(&profile);

    copy_runtime_dependency(&lib_path.join("ar8030_client.dll"), &target_dir);
    copy_runtime_dependency(&lib_path.join("pthread.dll"), &target_dir);

    for debug_runtime in debug_runtime_candidates() {
        copy_runtime_dependency(&debug_runtime, &target_dir);
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

fn copy_runtime_dependency(source: &Path, target_dir: &Path) {
    if !source.exists() {
        return;
    }

    let Some(file_name) = source.file_name() else {
        return;
    };

    let target_path = target_dir.join(file_name);
    let _ = fs::copy(source, &target_path);
    println!("cargo:rerun-if-changed={}", source.display());
}

fn debug_runtime_candidates() -> Vec<PathBuf> {
    vec![
        PathBuf::from(r"C:\Windows\System32\vcruntime140d.dll"),
        PathBuf::from(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC\14.44.35112\debug_nonredist\x64\Microsoft.VC143.DebugCRT\vcruntime140d.dll"),
        PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\ucrt\ucrtbased.dll"),
        PathBuf::from(r"C:\Program Files (x86)\Microsoft SDKs\Windows Kits\10\ExtensionSDKs\Microsoft.UniversalCRT.Debug\10.0.26100.0\Redist\Debug\x64\ucrtbased.dll"),
    ]
}