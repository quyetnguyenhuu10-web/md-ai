$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $root "native\build"

$env:CMAKE_TLS_VERIFY = "0"

cmake -S (Join-Path $root "native") -B $buildDir
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

cmake --build $buildDir --config Release
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
