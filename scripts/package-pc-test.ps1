param(
    [string]$PackageLabel = "pc-test-package",
    [switch]$SkipBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$binName = "wireless_status_server"
$profile = "release"
$releaseDir = Join-Path $repoRoot "target\$profile"
$outputRoot = Join-Path $repoRoot "dist\pc-test-package"
$templateDir = Join-Path $repoRoot "pc-test-package"
$staticDir = Join-Path $repoRoot "static"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$packageDirName = "${binName}_${PackageLabel}_${timestamp}"
$packageDir = Join-Path $outputRoot $packageDirName
$zipPath = Join-Path $outputRoot ("$packageDirName.zip")
$exePath = Join-Path $releaseDir ("$binName.exe")
$readmeTemplatePath = Join-Path $templateDir "README.md"
$startScriptPath = Join-Path $packageDir "start.cmd"

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing ${Description}: ${Path}"
    }
}

if (-not $SkipBuild) {
    Push-Location $repoRoot
    try {
        & cargo build --release
        if ($LASTEXITCODE -ne 0) {
            throw "cargo build --release failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

Assert-PathExists -Path $exePath -Description "release executable"
Assert-PathExists -Path $staticDir -Description "static directory"
Assert-PathExists -Path $readmeTemplatePath -Description "tester README template"

New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

if (Test-Path -LiteralPath $packageDir) {
    Remove-Item -LiteralPath $packageDir -Recurse -Force
}

if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

New-Item -ItemType Directory -Path $packageDir -Force | Out-Null
Copy-Item -LiteralPath $exePath -Destination $packageDir

$runtimeDlls = Get-ChildItem -Path $releaseDir -Filter *.dll -File | Sort-Object Name
foreach ($runtimeDll in $runtimeDlls) {
    Copy-Item -LiteralPath $runtimeDll.FullName -Destination $packageDir
}

Copy-Item -LiteralPath $staticDir -Destination (Join-Path $packageDir "static") -Recurse
Copy-Item -LiteralPath $readmeTemplatePath -Destination (Join-Path $packageDir "README.md")

$startScript = @'
@echo off
chcp 65001 >nul
cd /d %~dp0
set RUST_LOG=info
set BB_HOST_ADDR=127.0.0.1
set BB_HOST_PORT=50000
set SERVER_PORT=8080

wireless_status_server.exe
pause
'@

Set-Content -Path $startScriptPath -Value $startScript -Encoding Ascii

Compress-Archive -Path (Join-Path $packageDir "*") -DestinationPath $zipPath -Force

Write-Host "PC test package created:"
Write-Host "  Folder: $packageDir"
Write-Host "  Zip:    $zipPath"
Write-Host "  DLLs:   $($runtimeDlls.Count) copied"