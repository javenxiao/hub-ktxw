# FFI Boundary And Bindings Workflow

This repository keeps a stable FFI facade in [src/ffi.rs](src/ffi.rs) and moves adapter-facing summary types into [src/ffi_adapter_types.rs](src/ffi_adapter_types.rs).

The goal is to isolate third-party header changes from business logic in [src/bb_api.rs](src/bb_api.rs) and [src/main.rs](src/main.rs).

## Boundary Rules

1. Keep raw C ABI structs/constants/function signatures in the FFI raw area.
2. Keep business-friendly summary types in [src/ffi_adapter_types.rs](src/ffi_adapter_types.rs).
3. Keep `bb_api` and `main` dependent only on the stable exported facade from [src/ffi.rs](src/ffi.rs).

## Controlled Bindgen Regeneration

By default, build does not run bindgen.

Enable explicit regeneration with environment variables:

1. `RSHTML_REGENERATE_BINDINGS=1`
2. Optional output override: `RSHTML_BINDINGS_OUT=target/generated/ffi_bindings.rs`
3. Optional libclang path: `RSHTML_LIBCLANG_PATH=<path containing clang.dll or libclang.dll>`

Or use the helper script:

1. `./scripts/regenerate-ffi-bindings.ps1`
2. Optional output: `./scripts/regenerate-ffi-bindings.ps1 -Output target/generated/ffi_bindings.rs`
3. Optional libclang path: `./scripts/regenerate-ffi-bindings.ps1 -LibclangPath "C:/Program Files/LLVM/bin"`

## SDK Update Checklist

1. Replace files under `third_party/include` and `third_party/lib`.
2. Run the regeneration script.
3. Run `cargo check` and critical tests.
4. Review generated binding diff and adapt raw parsing only where needed.
5. Keep adapter-facing API unchanged unless there is a confirmed ABI break.