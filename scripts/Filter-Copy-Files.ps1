$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName Microsoft.VisualBasic

# ============================================================
# EXPORT EACH FOLDER TO ITS OWN TXT
# Chọn repo/folder -> quét đệ quy -> mỗi folder tạo 1 file .txt riêng
# Bên trong mỗi .txt có: đường dẫn file gốc + số dòng gốc + nội dung
# ============================================================

$OutputFolderPrefix = '_folder_txt_exports'

# Để trống = lấy mọi extension, trừ các rule loại trừ bên dưới.
$IncludeOnlyExtensions = @(
    '.py',
    '.yaml',
    '.yml',
    '.ts',
    '.tsx',
    '.js',
    '.jsx',
    '.html',
    '.css',
    '.md',
    '.txt',
    '.json',
    '.toml',
    '.ini',
    '.cfg'
)

$ExcludeFolderNames = @(
    '.git',
    '.svn',
    '.hg',
    'node_modules',
    '__pycache__',
    '.venv',
    'venv',
    'env',
    '.idea',
    '.vscode',
    'dist',
    'build',
    'target',
    '.next',
    '.nuxt',
    '.ruff_cache',
    '.pytest_cache',
    'coverage',
    'logs',
    'log',
    'tmp',
    'temp',
    'runs',
    'data_store',
    'behavior_locks',
    'tests'
)

$ExcludeFolderGlobPatterns = @(
    '_filtered_files_*',
    '_merged_files_*',
    '_folder_txt_exports_*'
)

$ExcludeFileNames = @(
    '.DS_Store',
    'Thumbs.db',
    'desktop.ini',
    'package-lock.json',
    'yarn.lock',
    'pnpm-lock.yaml'
)

$ExcludeExtensions = @(
    '.exe',
    '.dll',
    '.bin',
    '.zip',
    '.rar',
    '.7z',
    '.tar',
    '.gz',
    '.png',
    '.jpg',
    '.jpeg',
    '.gif',
    '.webp',
    '.mp4',
    '.mp3',
    '.wav',
    '.pdf',
    '.docx',
    '.xlsx',
    '.pptx'
)

$ExcludeGlobPatterns = @(
    '*.log',
    '*.tmp',
    '*.cache',
    '*.bak',
    '*.min.js',
    '*.map',
    '*.env',
    '.env',
    '*.secret',
    'secrets/*'
)

# Ví dụ: 5 = bỏ qua file lớn hơn 5MB
# $null = không giới hạn dung lượng
$MaxFileSizeMB = $null


function Normalize-PathText {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path
    )

    return $Path.Replace('\', '/').TrimEnd('/')
}


function Get-RelativePathCompat {
    param(
        [Parameter(Mandatory = $true)]
        [string] $BasePath,

        [Parameter(Mandatory = $true)]
        [string] $FullPath
    )

    $baseResolved = (Resolve-Path -LiteralPath $BasePath).Path.TrimEnd('\', '/')
    $fullResolved = (Resolve-Path -LiteralPath $FullPath).Path

    if (-not $fullResolved.StartsWith($baseResolved, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path không nằm trong folder gốc: $FullPath"
    }

    return $fullResolved.Substring($baseResolved.Length).TrimStart('\', '/')
}


function ConvertTo-SafeFileName {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Name
    )

    $invalidChars = [System.IO.Path]::GetInvalidFileNameChars()
    $safe = $Name

    foreach ($char in $invalidChars) {
        $safe = $safe.Replace($char, '_')
    }

    $safe = $safe.Replace('/', '_')
    $safe = $safe.Replace('\', '_')
    $safe = $safe.Replace(':', '_')
    $safe = $safe.Trim()

    if ($safe -eq '') {
        return '_EMPTY_NAME'
    }

    return $safe
}


function Get-UniqueDestinationPath {
    param(
        [Parameter(Mandatory = $true)]
        [string] $DestinationPath
    )

    if (-not (Test-Path -LiteralPath $DestinationPath)) {
        return $DestinationPath
    }

    $dir = Split-Path -Parent $DestinationPath
    $fileName = [System.IO.Path]::GetFileNameWithoutExtension($DestinationPath)
    $ext = [System.IO.Path]::GetExtension($DestinationPath)

    $i = 2

    while ($true) {
        $candidate = Join-Path $dir "$fileName`_$i$ext"

        if (-not (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }

        $i++
    }
}


function Select-Folder {
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = 'Chọn repo hoặc folder cần quét'
    $dialog.ShowNewFolderButton = $false

    $result = $dialog.ShowDialog()

    if ($result -ne [System.Windows.Forms.DialogResult]::OK) {
        throw 'Bạn chưa chọn folder nào.'
    }

    return $dialog.SelectedPath
}


function Test-ShouldExcludeFile {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.FileInfo] $File,

        [Parameter(Mandatory = $true)]
        [string] $RootPath,

        [Parameter(Mandatory = $true)]
        [string] $OutputFolderPath
    )

    $fileFull = $File.FullName
    $outputFull = (Resolve-Path -LiteralPath $OutputFolderPath).Path.TrimEnd('\', '/')

    if ($fileFull.StartsWith($outputFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        return @{
            Excluded = $true
            Reason   = 'Nằm trong folder output hiện tại'
        }
    }

    $relativePath = Get-RelativePathCompat -BasePath $RootPath -FullPath $fileFull
    $relativeNormalized = Normalize-PathText -Path $relativePath

    $fileName = $File.Name
    $extension = $File.Extension.ToLowerInvariant()

    if ($IncludeOnlyExtensions.Count -gt 0) {
        $normalizedInclude = $IncludeOnlyExtensions | ForEach-Object {
            $_.ToLowerInvariant()
        }

        if ($extension -notin $normalizedInclude) {
            return @{
                Excluded = $true
                Reason   = "Không nằm trong danh sách extension cần lấy: $extension"
            }
        }
    }

    $normalizedExcludeExts = $ExcludeExtensions | ForEach-Object {
        $_.ToLowerInvariant()
    }

    if ($extension -in $normalizedExcludeExts) {
        return @{
            Excluded = $true
            Reason   = "Bị loại theo extension: $extension"
        }
    }

    if ($fileName -in $ExcludeFileNames) {
        return @{
            Excluded = $true
            Reason   = "Bị loại theo tên file: $fileName"
        }
    }

    $directoryRelative = Split-Path -Parent $relativePath

    if ($null -ne $directoryRelative -and $directoryRelative.Trim() -ne '') {
        $folderParts = $directoryRelative -split '[\\/]'

        foreach ($part in $folderParts) {
            if ($part -in $ExcludeFolderNames) {
                return @{
                    Excluded = $true
                    Reason   = "Bị loại vì nằm trong folder: $part"
                }
            }

            foreach ($folderPattern in $ExcludeFolderGlobPatterns) {
                if ($part -like $folderPattern) {
                    return @{
                        Excluded = $true
                        Reason   = "Bị loại theo pattern folder: $folderPattern"
                    }
                }
            }
        }
    }

    foreach ($pattern in $ExcludeGlobPatterns) {
        $patternNormalized = Normalize-PathText -Path $pattern

        if ($fileName -like $pattern -or $relativeNormalized -like $patternNormalized) {
            return @{
                Excluded = $true
                Reason   = "Bị loại theo pattern: $pattern"
            }
        }
    }

    if ($null -ne $MaxFileSizeMB) {
        $maxBytes = [int64]($MaxFileSizeMB * 1024 * 1024)

        if ($File.Length -gt $maxBytes) {
            return @{
                Excluded = $true
                Reason   = "Vượt quá dung lượng giới hạn: $MaxFileSizeMB MB"
            }
        }
    }

    return @{
        Excluded = $false
        Reason   = ''
    }
}


function Read-FileLinesSafe {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path
    )

    try {
        return [System.IO.File]::ReadAllLines($Path)
    }
    catch {
        throw "Không đọc được file: $Path. Lỗi: $($_.Exception.Message)"
    }
}


function Get-FolderOutputFilePath {
    param(
        [Parameter(Mandatory = $true)]
        [string] $OutputFolderPath,

        [Parameter(Mandatory = $true)]
        [string] $RelativeFolderPath
    )

    if ($RelativeFolderPath -eq '.') {
        $baseName = '_ROOT'
    }
    else {
        $leafName = Split-Path -Leaf $RelativeFolderPath

        if ($null -eq $leafName -or $leafName.Trim() -eq '') {
            $leafName = $RelativeFolderPath
        }

        $baseName = ConvertTo-SafeFileName -Name $leafName
    }

    $candidatePath = Join-Path $OutputFolderPath "$baseName.txt"

    if (-not (Test-Path -LiteralPath $candidatePath)) {
        return $candidatePath
    }

    $safeRelative = ConvertTo-SafeFileName -Name $RelativeFolderPath
    $candidatePath = Join-Path $OutputFolderPath "$baseName`__$safeRelative.txt"

    return Get-UniqueDestinationPath -DestinationPath $candidatePath
}


try {
    Write-Host ''
    Write-Host '=== EXPORT EACH FOLDER TO ITS OWN TXT ===' -ForegroundColor Cyan
    Write-Host ''

    $rootPath = Select-Folder
    $rootPath = (Resolve-Path -LiteralPath $rootPath).Path.TrimEnd('\', '/')

    Write-Host 'Folder đã chọn:' -ForegroundColor Green
    Write-Host $rootPath
    Write-Host ''

    $extraExcludeInput = [Microsoft.VisualBasic.Interaction]::InputBox(
        "Nhập thêm rule loại trừ, cách nhau bằng dấu phẩy.`nVí dụ: *.env, secrets.*, private/*, data/*.csv`n`nCó thể để trống.",
        'Thêm rule loại trừ',
        '*.env,.env,*.secret,secrets/*'
    )

    if ($extraExcludeInput.Trim() -ne '') {
        $extraRules = $extraExcludeInput.Split(',') |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ -ne '' }

        $ExcludeGlobPatterns += $extraRules
    }

    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $outputFolderPath = Join-Path $rootPath "$OutputFolderPrefix`_$timestamp"

    New-Item -ItemType Directory -Path $outputFolderPath -Force | Out-Null

    Write-Host 'Folder output:' -ForegroundColor Green
    Write-Host $outputFolderPath
    Write-Host ''

    Write-Host 'Đang quét file...' -ForegroundColor Yellow

    $allFiles = Get-ChildItem -LiteralPath $rootPath -Recurse -Force -File

    $groups = @{}
    $includedRecords = New-Object System.Collections.Generic.List[object]
    $excludedRecords = New-Object System.Collections.Generic.List[object]
    $readErrorRecords = New-Object System.Collections.Generic.List[object]

    foreach ($file in $allFiles) {
        $relativeFilePath = Get-RelativePathCompat -BasePath $rootPath -FullPath $file.FullName
        $relativeFileNormalized = Normalize-PathText -Path $relativeFilePath

        $excludeResult = Test-ShouldExcludeFile `
            -File $file `
            -RootPath $rootPath `
            -OutputFolderPath $outputFolderPath

        if ([bool]$excludeResult.Excluded) {
            $excludedRecords.Add([pscustomobject]@{
                RelativePath = $relativeFileNormalized
                SourcePath   = $file.FullName
                SizeBytes    = $file.Length
                Reason       = [string]$excludeResult.Reason
            })

            continue
        }

        $parentFolderFullPath = Split-Path -Parent $file.FullName
        $relativeFolderPath = Get-RelativePathCompat -BasePath $rootPath -FullPath $parentFolderFullPath

        if ($relativeFolderPath.Trim() -eq '') {
            $relativeFolderPath = '.'
        }

        $relativeFolderNormalized = Normalize-PathText -Path $relativeFolderPath

        if (-not $groups.ContainsKey($relativeFolderNormalized)) {
            $groups[$relativeFolderNormalized] = New-Object System.Collections.Generic.List[object]
        }

        $entry = [pscustomobject]@{
            File                     = $file
            RelativeFilePath         = $relativeFilePath
            RelativeFileNormalized   = $relativeFileNormalized
            RelativeFolderPath       = $relativeFolderPath
            RelativeFolderNormalized = $relativeFolderNormalized
        }

        $groups[$relativeFolderNormalized].Add($entry)
        $includedRecords.Add($entry)
    }

    $groupKeys = $groups.Keys | Sort-Object
    $folderManifestRecords = New-Object System.Collections.Generic.List[object]

    $utf8NoBom = New-Object System.Text.UTF8Encoding -ArgumentList $false

    $folderIndex = 0
    $totalFolders = $groupKeys.Count
    $totalLinesWritten = 0

    foreach ($groupKey in $groupKeys) {
        $folderIndex++

        Write-Progress `
            -Activity 'Đang tạo nhiều file txt theo từng folder' `
            -Status "$folderIndex / $totalFolders : $groupKey" `
            -PercentComplete (($folderIndex / [Math]::Max($totalFolders, 1)) * 100)

        $outputTxtPath = Get-FolderOutputFilePath `
            -OutputFolderPath $outputFolderPath `
            -RelativeFolderPath $groupKey

        $items = $groups[$groupKey] | Sort-Object RelativeFileNormalized

        $writer = New-Object System.IO.StreamWriter -ArgumentList @($outputTxtPath, $false, $utf8NoBom)

        $folderLinesWritten = 0
        $folderReadErrors = 0

        try {
            $writer.WriteLine("# FOLDER EXPORT")
            $writer.WriteLine("# Root: $rootPath")
            $writer.WriteLine("# Folder: $groupKey")
            $writer.WriteLine("# Output: $outputTxtPath")
            $writer.WriteLine("# Created: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
            $writer.WriteLine("# Format: relative/path/file.ext:line_number | original line content")
            $writer.WriteLine("")

            foreach ($item in $items) {
                $file = [System.IO.FileInfo]$item.File
                $relativeFileNormalized = [string]$item.RelativeFileNormalized

                try {
                    $lines = Read-FileLinesSafe -Path $file.FullName

                    $writer.WriteLine("")
                    $writer.WriteLine("================================================================================")
                    $writer.WriteLine("FILE: $relativeFileNormalized")
                    $writer.WriteLine("SOURCE: $($file.FullName)")
                    $writer.WriteLine("SIZE_BYTES: $($file.Length)")
                    $writer.WriteLine("LINES: $($lines.Count)")
                    $writer.WriteLine("================================================================================")

                    $lineNo = 1

                    foreach ($line in $lines) {
                        $writer.WriteLine("${relativeFileNormalized}:$lineNo | $line")
                        $lineNo++
                        $folderLinesWritten++
                        $totalLinesWritten++
                    }
                }
                catch {
                    $folderReadErrors++

                    $readErrorRecords.Add([pscustomobject]@{
                        RelativePath = $relativeFileNormalized
                        SourcePath   = $file.FullName
                        Error        = $_.Exception.Message
                    })

                    $writer.WriteLine("")
                    $writer.WriteLine("================================================================================")
                    $writer.WriteLine("FILE: $relativeFileNormalized")
                    $writer.WriteLine("SOURCE: $($file.FullName)")
                    $writer.WriteLine("READ_ERROR: $($_.Exception.Message)")
                    $writer.WriteLine("================================================================================")
                }
            }
        }
        finally {
            $writer.Close()
            $writer.Dispose()
        }

        $folderManifestRecords.Add([pscustomobject]@{
            FolderRelativePath = $groupKey
            OutputTxt          = $outputTxtPath
            FileCount          = $items.Count
            LinesWritten       = $folderLinesWritten
            ReadErrors         = $folderReadErrors
        })
    }

    Write-Progress -Activity 'Đang tạo nhiều file txt theo từng folder' -Completed

    $manifestCsv = Join-Path $outputFolderPath '_folder_manifest.csv'
    $excludedCsv = Join-Path $outputFolderPath '_excluded_files.csv'
    $readErrorsCsv = Join-Path $outputFolderPath '_read_errors.csv'

    $folderManifestRecords |
        Export-Csv -LiteralPath $manifestCsv -NoTypeInformation -Encoding UTF8

    $excludedRecords |
        Export-Csv -LiteralPath $excludedCsv -NoTypeInformation -Encoding UTF8

    $readErrorRecords |
        Export-Csv -LiteralPath $readErrorsCsv -NoTypeInformation -Encoding UTF8

    Write-Host ''
    Write-Host 'HOÀN TẤT!' -ForegroundColor Green
    Write-Host "Tổng file quét:       $($allFiles.Count)"
    Write-Host "File hợp lệ:          $($includedRecords.Count)"
    Write-Host "File bị loại:         $($excludedRecords.Count)"
    Write-Host "Folder có xuất txt:   $($folderManifestRecords.Count)"
    Write-Host "Lỗi đọc file:         $($readErrorRecords.Count)"
    Write-Host "Tổng dòng đã ghi:     $totalLinesWritten"
    Write-Host ''
    Write-Host "Folder output:" -ForegroundColor Cyan
    Write-Host $outputFolderPath
    Write-Host ''
    Write-Host "Manifest:" -ForegroundColor Cyan
    Write-Host $manifestCsv
    Write-Host ''

    [System.Windows.Forms.MessageBox]::Show(
        "Xong!`n`nĐã tạo TXT cho: $($folderManifestRecords.Count) folder`nFile hợp lệ: $($includedRecords.Count)`nFile bị loại: $($excludedRecords.Count)`nLỗi đọc: $($readErrorRecords.Count)`n`nOutput:`n$outputFolderPath",
        'Hoàn tất',
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Information
    ) | Out-Null
}
catch {
    Write-Host ''
    Write-Host 'CÓ LỖI XẢY RA:' -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ''
    Read-Host 'Nhấn Enter để đóng'
    throw
}

Read-Host 'Nhấn Enter để kết thúc'