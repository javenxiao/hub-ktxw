param(
    [string]$Output = "target/generated/ffi_bindings.rs",
    [string]$LibclangPath = ""
)

$ErrorActionPreference = "Stop"

$cargoPath = Join-Path $HOME ".cargo\bin\cargo.exe"
if (-not (Test-Path $cargoPath)) {
    throw "cargo.exe not found at $cargoPath"
}

$prevRegenerate = $env:RSHTML_REGENERATE_BINDINGS
$prevOutput = $env:RSHTML_BINDINGS_OUT
$prevLibclang = $env:RSHTML_LIBCLANG_PATH

try {
    $env:RSHTML_REGENERATE_BINDINGS = "1"
    $env:RSHTML_BINDINGS_OUT = $Output

    if ([string]::IsNullOrWhiteSpace($LibclangPath)) {
        if (Test-Path Env:RSHTML_LIBCLANG_PATH) {
            Remove-Item Env:RSHTML_LIBCLANG_PATH
        }
    } else {
        $env:RSHTML_LIBCLANG_PATH = $LibclangPath
    }

    & $cargoPath check
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    Write-Output "FFI bindings regenerated to: $Output"
} finally {
    if ($null -eq $prevRegenerate) {
        if (Test-Path Env:RSHTML_REGENERATE_BINDINGS) {
            Remove-Item Env:RSHTML_REGENERATE_BINDINGS
        }
    } else {
        $env:RSHTML_REGENERATE_BINDINGS = $prevRegenerate
    }

    if ($null -eq $prevOutput) {
        if (Test-Path Env:RSHTML_BINDINGS_OUT) {
            Remove-Item Env:RSHTML_BINDINGS_OUT
        }
    } else {
        $env:RSHTML_BINDINGS_OUT = $prevOutput
    }

    if ($null -eq $prevLibclang) {
        if (Test-Path Env:RSHTML_LIBCLANG_PATH) {
            Remove-Item Env:RSHTML_LIBCLANG_PATH
        }
    } else {
        $env:RSHTML_LIBCLANG_PATH = $prevLibclang
    }
}