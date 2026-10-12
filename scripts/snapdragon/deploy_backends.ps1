# Deploy Qualcomm Hexagon NPU shim and runtime DLLs into build\bin

$ErrorActionPreference = "Stop"

$BuildBin = Join-Path $PSScriptRoot "..\..\build\bin"
if (-not (Test-Path $BuildBin)) {
    New-Item -ItemType Directory -Path $BuildBin -Force | Out-Null
}

Write-Host "Compiling ggml-hexagon entrypoint shim..." -ForegroundColor Cyan
clang -target aarch64-pc-windows-msvc -shared -O2 `
  -I (Join-Path $PSScriptRoot "..\..\ggml\include") `
  -I (Join-Path $PSScriptRoot "..\..\ggml\src") `
  (Join-Path $PSScriptRoot "ggml_hexagon_shim.c") `
  -lkernel32 `
  -o (Join-Path $BuildBin "ggml-hexagon.dll")

Write-Host "Copying Qualcomm WHQL core runtime and FastRPC drivers..." -ForegroundColor Cyan

$GeniePath = "C:\Qualcomm\geniex-cli\llama_cpp"
if (Test-Path $GeniePath) {
    Copy-Item "$GeniePath\ggml-hexagon.dll" (Join-Path $BuildBin "ggml-hexagon-core.dll") -Force
    Copy-Item "$GeniePath\libggml-htp*.so" $BuildBin -Force
    Copy-Item "$GeniePath\libggml-htp.cat" $BuildBin -Force
    Copy-Item "$GeniePath\libomp140.aarch64.dll" $BuildBin -Force
} else {
    Write-Warning "Could not find $GeniePath. Please ensure Qualcomm GenieX runtime is installed."
}

# FastRPC user-mode driver
$DriverPattern = "C:\Windows\System32\DriverStore\FileRepository\qcnspmcdm8380.inf_arm64_*\libcdsprpc.dll"
$DriverDll = Get-ChildItem -Path $DriverPattern -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
if (-not $DriverDll) {
    $DriverDll = "C:\Windows\System32\libcdsprpc.dll"
}

if (Test-Path $DriverDll) {
    Copy-Item $DriverDll (Join-Path $BuildBin "libcdsprpc.dll") -Force
} else {
    Write-Warning "Could not find libcdsprpc.dll in DriverStore or System32."
}

Write-Host "Deployment complete! Runtime libraries in ${BuildBin}:" -ForegroundColor Green
Get-ChildItem -Path $BuildBin -Filter "*hexagon*" | Select-Object Name, Length
Get-ChildItem -Path $BuildBin -Filter "*opencl*" | Select-Object Name, Length
