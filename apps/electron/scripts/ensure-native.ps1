$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root "native\build\sidecar\Release\chat_core_sidecar.exe"
$sourceRoots = @(
  (Join-Path $root "native\CMakeLists.txt"),
  (Join-Path $root "native\chat_core"),
  (Join-Path $root "native\sidecar")
)

$needsBuild = -not (Test-Path -LiteralPath $exe)

if (-not $needsBuild) {
  $exeTime = (Get-Item -LiteralPath $exe).LastWriteTimeUtc
  $latestSource = $sourceRoots |
    ForEach-Object {
      if (Test-Path -LiteralPath $_ -PathType Container) {
        Get-ChildItem -LiteralPath $_ -Recurse -File
      } elseif (Test-Path -LiteralPath $_ -PathType Leaf) {
        Get-Item -LiteralPath $_
      }
    } |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1

  $needsBuild = $null -ne $latestSource -and $latestSource.LastWriteTimeUtc -gt $exeTime
}

if ($needsBuild) {
  & (Join-Path $PSScriptRoot "build-native.ps1")
} else {
  Write-Host "Native sidecar is up to date."
}
