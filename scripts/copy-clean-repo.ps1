[CmdletBinding()]
param(
    [string]$SourceRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Destination = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-AbsolutePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath((Join-Path -Path (Get-Location) -ChildPath $Path))
}

$source = Resolve-Path -LiteralPath $SourceRoot | Select-Object -ExpandProperty Path
$sourceParent = Split-Path -Parent $source
$sourceName = Split-Path -Leaf $source

if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path -Path $sourceParent -ChildPath "$sourceName-clean"
}

$destination = Resolve-AbsolutePath -Path $Destination
$sourceWithSeparator = $source.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
$destinationWithSeparator = $destination.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar

if ($destination.Equals($source, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Destination must be different from SourceRoot. Got: $destination"
}

if ($destinationWithSeparator.StartsWith($sourceWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Destination must not be inside SourceRoot. Got: $destination"
}

if ($sourceWithSeparator.StartsWith($destinationWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Destination must not contain SourceRoot. Got: $destination"
}

if (Test-Path -LiteralPath $destination) {
    Remove-Item -LiteralPath $destination -Recurse -Force
}

New-Item -ItemType Directory -Path $destination -Force | Out-Null

$excludeDirs = @(
    '.git'
    '.github'
    '.code-review-graph'
    '.claude'
    '.cursor'
    '.gemini'
    '.qoder'
    '.vscode'
    '.ipynb_checkpoints'
    '__pycache__'
    '.pytest_cache'
    '.ruff_cache'
    'build'
    'dist'
    'reports'
    'LM-checkpoints'
    'LM-artifacts'
    'checkpoints'
    'node_modules'
    'runs'
    'experiments'
    'apps'
)

$excludeFiles = @(
    '*.pyc'
    '*.pyo'
)

$robocopyArgs = @(
    $source
    $destination
    '/MIR'
    '/R:1'
    '/W:1'
    '/NFL'
    '/NDL'
    '/NJH'
    '/NJS'
    '/NP'
    '/XD'
) + $excludeDirs + @('/XF') + $excludeFiles

Write-Host "Source: $source"
Write-Host "Destination: $destination"
Write-Host "Excluded dirs: $($excludeDirs -join ', ')"
Write-Host "Excluded files: $($excludeFiles -join ', ')"

& robocopy @robocopyArgs
$exitCode = $LASTEXITCODE

if ($exitCode -ge 8) {
    throw "Robocopy failed with exit code $exitCode."
}

Write-Host "Clean copy complete."
