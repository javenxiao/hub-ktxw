// build.rs - 配置 FFI 绑定和 DLL 链接
use std::fs;
use std::env;
use std::path::{Path, PathBuf};

const REGENERATE_BINDINGS_ENV: &str = "RSHTML_REGENERATE_BINDINGS";
const BINDINGS_OUTPUT_ENV: &str = "RSHTML_BINDINGS_OUT";
const BINDINGS_LIBCLANG_ENV: &str = "RSHTML_LIBCLANG_PATH";

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_path = manifest_dir.join("third_party").join("lib");
    let include_path = manifest_dir.join("third_party").join("include");

    // ============ 自动复制 DLL 到输出目录 ============
    let profile = env::var("PROFILE").unwrap();
    let target_dir = manifest_dir.join("target").join(&profile);

    copy_runtime_dependency(&lib_path.join("ar8030_client.dll"), &target_dir);
    copy_runtime_dependency(&lib_path.join("pthread.dll"), &target_dir);

    for debug_runtime in debug_runtime_candidates() {
        copy_runtime_dependency(&debug_runtime, &target_dir);
    }

    emit_bindings_watchers(&include_path);
    maybe_regenerate_bindings(&manifest_dir, &include_path);
}

fn emit_bindings_watchers(include_path: &Path) {
    println!(
        "cargo:rerun-if-changed={}",
        include_path.join("bb_api.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        include_path.join("bb_config.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        include_path.join("prj_rpc.h").display()
    );
    println!("cargo:rerun-if-env-changed={}", REGENERATE_BINDINGS_ENV);
    println!("cargo:rerun-if-env-changed={}", BINDINGS_OUTPUT_ENV);
    println!("cargo:rerun-if-env-changed={}", BINDINGS_LIBCLANG_ENV);
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
}

fn maybe_regenerate_bindings(manifest_dir: &Path, include_path: &Path) {
    if !env_flag_enabled(REGENERATE_BINDINGS_ENV) {
        return;
    }

    configure_libclang_path();

    let output_path = resolved_bindings_output_path(manifest_dir);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("failed to create bindings output directory {}: {}", parent.display(), error));
    }

    let bb_api_header = include_path.join("bb_api.h");
    let prj_rpc_header = include_path.join("prj_rpc.h");

    let bindings = bindgen::Builder::default()
        .header(bb_api_header.to_string_lossy().into_owned())
        .header(prj_rpc_header.to_string_lossy().into_owned())
        .clang_arg(format!("-I{}", include_path.display()))
        .clang_arg("-D")
        .clang_arg("PACK(...)=__attribute__((packed)) __VA_ARGS__")
        .generate_comments(true)
        .derive_default(true)
        .generate()
        .unwrap_or_else(|error| {
            panic!(
                "bindgen failed for {} and {}: {}",
                bb_api_header.display(),
                prj_rpc_header.display(),
                error
            )
        });

    bindings
        .write_to_file(&output_path)
        .unwrap_or_else(|error| panic!("failed to write bindings to {}: {}", output_path.display(), error));

    println!(
        "cargo:warning=Regenerated SDK FFI bindings at {}",
        output_path.display()
    );
}

fn resolved_bindings_output_path(manifest_dir: &Path) -> PathBuf {
    let configured = env::var(BINDINGS_OUTPUT_ENV).ok().map(|value| value.trim().to_string());

    let path = match configured {
        Some(value) if !value.is_empty() => PathBuf::from(value),
        _ => PathBuf::from("target/generated/ffi_bindings.rs"),
    };

    if path.is_absolute() {
        path
    } else {
        manifest_dir.join(path)
    }
}

fn env_flag_enabled(key: &str) -> bool {
    let Ok(value) = env::var(key) else {
        return false;
    };

    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn configure_libclang_path() {
    if env::var("LIBCLANG_PATH")
        .ok()
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
    {
        return;
    }

    if let Some(configured) = env::var(BINDINGS_LIBCLANG_ENV)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        let path = PathBuf::from(configured);
        if path_contains_libclang(&path) {
            env::set_var("LIBCLANG_PATH", &path);
            println!(
                "cargo:warning=Using LIBCLANG_PATH from {} -> {}",
                BINDINGS_LIBCLANG_ENV,
                path.display()
            );
            return;
        }

        panic!(
            "{} is set to {}, but clang.dll/libclang.dll was not found there",
            BINDINGS_LIBCLANG_ENV,
            path.display()
        );
    }

    for candidate in libclang_path_candidates() {
        if path_contains_libclang(&candidate) {
            env::set_var("LIBCLANG_PATH", &candidate);
            println!(
                "cargo:warning=Auto-detected LIBCLANG_PATH={} for bindgen",
                candidate.display()
            );
            return;
        }
    }

    panic!(
        "Unable to find clang.dll/libclang.dll for bindgen. Set LIBCLANG_PATH or {} before enabling {}.",
        BINDINGS_LIBCLANG_ENV,
        REGENERATE_BINDINGS_ENV
    );
}

fn libclang_path_candidates() -> Vec<PathBuf> {
    vec![
        PathBuf::from(r"C:\Program Files\LLVM\bin"),
        PathBuf::from(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin"),
        PathBuf::from(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\Llvm\x64\bin"),
        PathBuf::from(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\Llvm\x64\bin"),
    ]
}

fn path_contains_libclang(path: &Path) -> bool {
    path.join("clang.dll").exists() || path.join("libclang.dll").exists()
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