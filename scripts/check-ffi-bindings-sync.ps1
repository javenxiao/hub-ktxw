param(
    [string]$BindingsPath = "src/generated/ffi_bindings.rs",
    [string]$LibclangPath = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BindingsFullPath = Join-Path $ProjectRoot $BindingsPath

if (-not (Test-Path $BindingsFullPath)) {
    Write-Error "Bindings file not found: $BindingsFullPath"
    exit 1
}

$beforeHash = (Get-FileHash -Path $BindingsFullPath -Algorithm SHA256).Hash

$prevRegenerate = $env:RSHTML_REGENERATE_BINDINGS
$prevOutput = $env:RSHTML_BINDINGS_OUT
$prevLibclang = $env:RSHTML_LIBCLANG_PATH

try {
    $env:RSHTML_REGENERATE_BINDINGS = "1"
    $env:RSHTML_BINDINGS_OUT = $BindingsPath

    if ([string]::IsNullOrWhiteSpace($LibclangPath)) {
        if (Test-Path Env:RSHTML_LIBCLANG_PATH) {
            Remove-Item Env:RSHTML_LIBCLANG_PATH
        }
    } else {
        $env:RSHTML_LIBCLANG_PATH = $LibclangPath
    }

    $cargoPath = Join-Path $HOME ".cargo\bin\cargo.exe"
    if (-not (Test-Path $cargoPath)) {
        $cargoPath = "cargo"
    }

    & $cargoPath check
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $afterHash = (Get-FileHash -Path $BindingsFullPath -Algorithm SHA256).Hash
    if ($beforeHash -ne $afterHash) {
        Write-Error "FFI bindings are out of date. Please run scripts/regenerate-ffi-bindings.ps1 and commit $BindingsPath"
        exit 2
    }

    Write-Output "FFI bindings are in sync: $BindingsPath"
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
